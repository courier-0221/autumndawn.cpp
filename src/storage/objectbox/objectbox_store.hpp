#pragma once

#include <memory>
#include <string>

#include "rag/storage/vector_store.hpp"

namespace autumndawn::rag::storage {

/// ObjectBox HNSW 向量存储。私有实现，仅供 `store_factory.cpp` 装配使用；
/// 不暴露到 `include/`。上层代码只依赖 `IVectorStore` 抽象。
///
/// 构造函数会：
/// - 校验 `obx_has_feature(OBXFeature_VectorSearch)`，缺失即抛 StorageError；
/// - 打开位于 `directory` 的库（不存在则自动创建）。
class ObjectBoxVectorStore : public IVectorStore {
public:
    ObjectBoxVectorStore(const std::string& directory, std::size_t embeddingDim);
    ~ObjectBoxVectorStore() override;

    std::uint64_t put(DocumentRecord doc) override;
    void putBatch(std::vector<DocumentRecord> docs) override;
    std::uint64_t count() const override;
    std::uint64_t clearAll() override;
    std::vector<SearchHit> searchByVector(const std::vector<float>& queryEmbedding,
                                          int topK) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace autumndawn::rag::storage
