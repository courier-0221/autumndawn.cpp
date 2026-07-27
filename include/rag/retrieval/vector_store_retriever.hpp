#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rag/inference/embedding_model.hpp"
#include "rag/retriever.hpp"
#include "rag/storage/vector_store.hpp"

namespace autumndawn::rag {

/// 后端无关的向量检索器：`IEmbeddingModel` + `IVectorStore` → `IRetriever`。
///
/// 上层 pipeline / 装饰器（multi-query / HyDE / rerank ...）只依赖本类的
/// `IRetriever` 契约；换存储后端只需在配置里改 `storage.provider`。
///
/// `ingest` 是把"文本 → 批量 embedding → 批量入库"这条常用链路收成一个
/// 便捷方法——它并非 `IRetriever` 契约的一部分，但由于所需的两个依赖
/// (`embedder_` / `store_`) 都已在手边，作为便利方法暴露。
class VectorStoreRetriever : public IRetriever {
public:
    VectorStoreRetriever(std::shared_ptr<inference::IEmbeddingModel> embedder,
                         std::shared_ptr<storage::IVectorStore> store);

    /// 批量写入：对 `chunks` 做一次 `embedBatch`，然后一次性 `putBatch`。
    /// @param source 来源标签（如文件名），便于日后溯源。
    void ingest(const std::vector<std::string>& chunks, const std::string& source);

    /// 单条写入。返回后端分配的原生 id。
    std::uint64_t addOne(const std::string& text, const std::string& source);

    std::vector<RetrievedChunk> retrieve(const std::string& query, int topK) override;

    /// 暴露底层 store，方便上层做 count / clearAll 等纯存储操作，
    /// 不必再持有 store 引用。
    const std::shared_ptr<storage::IVectorStore>& store() const { return store_; }

private:
    std::shared_ptr<inference::IEmbeddingModel> embedder_;
    std::shared_ptr<storage::IVectorStore> store_;
};

}  // namespace autumndawn::rag
