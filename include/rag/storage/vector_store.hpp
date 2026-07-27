#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rag/storage/document_record.hpp"

namespace autumndawn::rag::storage {

/// 向量数据库的中性接口：只描述"存 / 数 / 清 / 按向量近邻检索"四件事，
/// 不感知具体后端（ObjectBox / Qdrant / FAISS / in-memory 等）。
/// `VectorStoreRetriever` 把两者组合成上层期望的 `IRetriever`。
class IVectorStore {
public:
    /// 单次 `searchByVector` 的一条命中。
    struct SearchHit {
        std::uint64_t id = 0;
        std::string text;
        std::string source;
        /// 后端语义下的"越靠前越相关"分数：
        ///   - ObjectBox HNSW cosine：值越小越相似（距离）；
        ///   - 未来若接 rerank/内积后端：值越大越相似。
        /// 上层只依赖"返回顺序 = 相关性从高到低"，不解释具体数值。
        double score = 0.0;
    };

    virtual ~IVectorStore() = default;

    /// 写入一条记录，返回后端分配的原生 id。
    virtual std::uint64_t put(DocumentRecord doc) = 0;

    /// 批量写入（在支持事务的后端里作为单事务提交）。
    virtual void putBatch(std::vector<DocumentRecord> docs) = 0;

    /// 当前存储的记录总数。
    virtual std::uint64_t count() const = 0;

    /// 清空所有记录，返回删除条数。
    virtual std::uint64_t clearAll() = 0;

    /// 按查询向量做近邻检索，返回按相关性排序的前 topK 条。
    /// @param queryEmbedding 与库中向量同维度的查询向量。
    /// @param topK           期望条数（>0）；后端按需向下裁剪。
    virtual std::vector<SearchHit> searchByVector(const std::vector<float>& queryEmbedding,
                                                  int topK) = 0;
};

}  // namespace autumndawn::rag::storage
