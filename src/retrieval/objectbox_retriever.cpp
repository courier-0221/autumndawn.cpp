#include "rag/objectbox_retriever.hpp"

#include "objectbox.hpp"
#include "document.obx.hpp"
#include "objectbox-model.h"

namespace autumndawn::rag {

OBX_model* createRagModel() {
    return create_obx_model();  // 由 objectbox-model.h 提供（static inline）
}

struct ObjectBoxRetriever::Impl {
    obx::Store& store;
    obx::Box<Document> box;
    obx::Query<Document> queryByVector;
    std::shared_ptr<inference::IEmbeddingClient> embedder;

    Impl(obx::Store& s, std::shared_ptr<inference::IEmbeddingClient> emb)
        : store(s),
          box(s),
          queryByVector(box.query(Document_::embedding.nearestNeighbors({}, 1)).build()),
          embedder(std::move(emb)) {}
};

ObjectBoxRetriever::ObjectBoxRetriever(obx::Store& store,
                                       std::shared_ptr<inference::IEmbeddingClient> embedder)
    : impl_(std::make_unique<Impl>(store, std::move(embedder))) {}

ObjectBoxRetriever::~ObjectBoxRetriever() = default;

void ObjectBoxRetriever::ingest(const std::vector<std::string>& chunks, const std::string& source) {
    if (chunks.empty()) return;
    auto vectors = impl_->embedder->embedBatch(chunks);

    obx::Transaction tx = impl_->store.tx(obx::TxMode::WRITE);
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        Document doc;
        doc.id = 0;
        doc.text = chunks[i];
        doc.source = source;
        doc.embedding = vectors[i];
        impl_->box.put(doc);
    }
    tx.success();
}

std::uint64_t ObjectBoxRetriever::addOne(const std::string& text, const std::string& source) {
    Document doc;
    doc.id = 0;
    doc.text = text;
    doc.source = source;
    doc.embedding = impl_->embedder->embed(text);
    return impl_->box.put(doc);
}

std::uint64_t ObjectBoxRetriever::clearAll() {
    return impl_->box.removeAll();
}

std::uint64_t ObjectBoxRetriever::count() const {
    return impl_->box.count();
}

std::vector<RetrievedChunk> ObjectBoxRetriever::retrieve(const std::string& query, int topK) {
    if (topK <= 0) return {};

    std::vector<float> qv = impl_->embedder->embed(query);
    impl_->queryByVector.setParameter(Document_::embedding, qv);
    impl_->queryByVector.setParameterMaxNeighbors(Document_::embedding,
                                                  static_cast<int64_t>(topK));
    auto raw = impl_->queryByVector.findWithScores();

    std::vector<RetrievedChunk> out;
    out.reserve(raw.size());
    for (auto& p : raw) {
        RetrievedChunk c;
        c.id = p.first.id;
        c.text = std::move(p.first.text);
        c.source = std::move(p.first.source);
        c.score = p.second;
        out.push_back(std::move(c));
    }
    return out;
}

}  // namespace autumndawn::rag
