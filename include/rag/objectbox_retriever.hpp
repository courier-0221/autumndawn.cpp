#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rag/inference/embedding_client.hpp"
#include "rag/retriever.hpp"

// 前向声明以避免把 objectbox.hpp 拖进公共头文件。
namespace obx {
class Store;
}
extern "C" {
struct OBX_model;
}

namespace autumndawn::rag {

/// 构建库内 `Document` 实体对应的 ObjectBox 模型。
/// 用法：
///   obx::Options opts(rag::createRagModel());
///   opts.directory("objectbox-db");
///   obx::Store store(opts);
///
/// 返回值所有权交给 `obx::Options`；调用后不要手动 free。
OBX_model* createRagModel();

/// ObjectBox HNSW 向量检索器。文本 → embedding → cosine 近邻。
class ObjectBoxRetriever : public IRetriever {
public:
    ObjectBoxRetriever(obx::Store& store,
                       std::shared_ptr<inference::IEmbeddingClient> embedder);
    ~ObjectBoxRetriever() override;

    /// 批量写入：对 chunks 做一次批量 embedding，然后事务写入。
    /// @param source 来源标签（如文件名），便于日后溯源。
    void ingest(const std::vector<std::string>& chunks, const std::string& source);

    /// 单条写入。
    std::uint64_t addOne(const std::string& text, const std::string& source);

    /// 清空所有文档。返回删除条数。
    std::uint64_t clearAll();

    /// 已存储文档数量。
    std::uint64_t count() const;

    std::vector<RetrievedChunk> retrieve(const std::string& query, int topK) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace autumndawn::rag
