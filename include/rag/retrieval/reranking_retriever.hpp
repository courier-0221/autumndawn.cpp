#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rag/inference/rerank_model.hpp"
#include "rag/retriever.hpp"

namespace autumndawn::rag {

/// Re-ranking 装饰器（课程小节 15）。
///
/// 生产 RAG 里最常见的"两阶段检索"：
///   1) 底层 `IRetriever` 用向量近邻先召回 `candidateTopN` 条（recall 阶段，快而粗）；
///   2) 再让 cross-encoder 类的 `IRerankModel`（如 bge-reranker-v2-m3）
///      在 (query, doc) 语义层面精排（precision 阶段，慢而准）；
///   3) 取精排后的 top-K 返回。
///
/// 只依赖 `IRetriever` / `IRerankModel` 抽象，可装饰任意底层检索器
/// （plain / multi-query / hyde / routed ...），返回值仍是 `IRetriever` 契约。
///
/// 语义变化：返回的 `RetrievedChunk::score` **改写为 rerank 相关性分数（越大越相关）**，
/// 顺序依然是"越靠前越相关"，符合 `IRetriever` 契约。
class RerankingRetriever : public IRetriever {
public:
    struct Options {
        /// 底层召回条数上限。实际召回 = max(candidateTopN, topK)。
        /// 越大 recall 越充分、rerank 开销越高，一般设为 topK 的 3~5 倍。
        int candidateTopN = 20;
    };

    RerankingRetriever(std::shared_ptr<IRetriever> base,
                       std::shared_ptr<inference::IRerankModel> rerank)
        : RerankingRetriever(std::move(base), std::move(rerank), Options{}) {}
    RerankingRetriever(std::shared_ptr<IRetriever> base,
                       std::shared_ptr<inference::IRerankModel> rerank, Options options);

    std::vector<RetrievedChunk> retrieve(const std::string& query, int topK) override;

    /// 最近一次 retrieve 里，底层召回的候选条数（rerank 之前）。
    /// 供演示程序打印"从多少条候选里精排出 topK"。
    std::size_t lastCandidateCount() const { return lastCandidateCount_; }

private:
    std::shared_ptr<IRetriever> base_;
    std::shared_ptr<inference::IRerankModel> rerank_;
    Options options_;
    std::size_t lastCandidateCount_ = 0;
};

}  // namespace autumndawn::rag
