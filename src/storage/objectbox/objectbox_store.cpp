// 在整个 autumndawn_rag 静态库里，本 TU 是唯一定义 OBX_CPP_FILE 的地方，
// 负责让 objectbox.hpp 中"header-only"的实现真正 materialize 到目标文件里。
// 上层示例/应用（如 rag_basic/main.cpp）不需要再定义 OBX_CPP_FILE，也不需要
// include "objectbox.hpp"——它们通过 `IVectorStore` 抽象与 `store_factory`
// 使用本实现。
#define OBX_CPP_FILE

#include "storage/objectbox/objectbox_store.hpp"

#include <utility>

#include "objectbox.hpp"

#include "document.obx.hpp"
#include "objectbox-model.h"
#include "rag/storage/storage_error.hpp"

namespace autumndawn::rag::storage {

namespace {

/// 断言 ObjectBox 运行库启用了 VectorSearch 能力；否则 HNSW 索引不可用。
void ensureVectorSearchSupported() {
    if (!obx_has_feature(OBXFeature_VectorSearch)) {
        throw StorageError(
            "ObjectBox runtime lacks VectorSearch feature. "
            "请下载带 vector search 的 libobjectbox.so 覆盖 third_party/objectbox/lib/.");
    }
}

}  // namespace

struct ObjectBoxVectorStore::Impl {
    obx::Store store;
    obx::Box<Document> box;
    obx::Query<Document> queryByVector;

    static obx::Options makeOptions(const std::string& directory) {
        obx::Options opts(create_obx_model());  // 由 objectbox-model.h 提供 (static inline)
        opts.directory(directory);
        return opts;
    }

    explicit Impl(const std::string& directory)
        : store(makeOptions(directory)),
          box(store),
          queryByVector(box.query(Document_::embedding.nearestNeighbors({}, 1)).build()) {}
};

ObjectBoxVectorStore::ObjectBoxVectorStore(const std::string& directory,
                                           std::size_t /*embeddingDim*/) {
    // embeddingDim 目前不用于运行期校验（ObjectBox HNSW 索引维度写死在
    // document.fbs 的 hnsw-dimensions=1024）；保留形参是为了将来切换
    // embedding 模型时能在此加校验/自动 reindex。
    ensureVectorSearchSupported();
    impl_ = std::make_unique<Impl>(directory);
}

ObjectBoxVectorStore::~ObjectBoxVectorStore() = default;

std::uint64_t ObjectBoxVectorStore::put(DocumentRecord doc) {
    Document d;
    d.id = doc.id;
    d.text = std::move(doc.text);
    d.source = std::move(doc.source);
    d.embedding = std::move(doc.embedding);
    return impl_->box.put(d);
}

void ObjectBoxVectorStore::putBatch(std::vector<DocumentRecord> docs) {
    if (docs.empty()) return;
    obx::Transaction tx = impl_->store.tx(obx::TxMode::WRITE);
    for (auto& r : docs) {
        Document d;
        d.id = r.id;
        d.text = std::move(r.text);
        d.source = std::move(r.source);
        d.embedding = std::move(r.embedding);
        impl_->box.put(d);
    }
    tx.success();
}

std::uint64_t ObjectBoxVectorStore::count() const {
    return impl_->box.count();
}

std::uint64_t ObjectBoxVectorStore::clearAll() {
    return impl_->box.removeAll();
}

std::vector<IVectorStore::SearchHit>
ObjectBoxVectorStore::searchByVector(const std::vector<float>& queryEmbedding, int topK) {
    if (topK <= 0) return {};
    impl_->queryByVector.setParameter(Document_::embedding, queryEmbedding);
    impl_->queryByVector.setParameterMaxNeighbors(Document_::embedding,
                                                  static_cast<int64_t>(topK));
    auto raw = impl_->queryByVector.findWithScores();

    std::vector<SearchHit> out;
    out.reserve(raw.size());
    for (auto& p : raw) {
        SearchHit h;
        h.id = p.first.id;
        h.text = std::move(p.first.text);
        h.source = std::move(p.first.source);
        h.score = p.second;
        out.push_back(std::move(h));
    }
    return out;
}

}  // namespace autumndawn::rag::storage
