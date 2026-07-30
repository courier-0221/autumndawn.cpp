#include "rag/retrieval/reranking_retriever.hpp"

#include <algorithm>
#include <utility>

#include "rag/log.hpp"

namespace autumndawn::rag {

RerankingRetriever::RerankingRetriever(std::shared_ptr<IRetriever> base,
                                       std::shared_ptr<inference::IRerankModel> rerank,
                                       Options options)
    : base_(std::move(base)),
      rerank_(std::move(rerank)),
      options_(std::move(options)) {}

std::vector<RetrievedChunk> RerankingRetriever::retrieve(const std::string& query, int topK) {
    lastCandidateCount_ = 0;
    if (topK <= 0) return {};

    // 第一阶段：向量近邻粗召回。candidateTopN 至少要覆盖 topK。
    const int candidateN = std::max(options_.candidateTopN, topK);
    auto candidates = base_->retrieve(query, candidateN);
    lastCandidateCount_ = candidates.size();
    if (candidates.empty()) return {};

    // 若候选少于等于 topK，直接返回原顺序（rerank 只会花冤枉的 API 调用）。
    if (static_cast<int>(candidates.size()) <= topK) {
        return candidates;
    }

    // 第二阶段：让 rerank 模型在 (query, doc) 语义层面精排。
    std::vector<std::string> docs;
    docs.reserve(candidates.size());
    for (const auto& c : candidates) docs.push_back(c.text);

    std::vector<inference::RerankHit> hits;
    try {
        hits = rerank_->rerank(query, docs, topK);
    } catch (const std::exception& e) {
        LOG(WARNING) << "RerankingRetriever: rerank failed (" << e.what()
                     << "), falling back to first-stage top-K";
        candidates.resize(topK);
        return candidates;
    }

    // 依 rerank 结果重排原始候选；rerank score 覆盖到 RetrievedChunk::score。
    std::vector<RetrievedChunk> ordered;
    ordered.reserve(hits.size());
    for (const auto& hit : hits) {
        if (hit.index >= candidates.size()) continue;  // 防御
        RetrievedChunk chunk = candidates[hit.index];
        chunk.score = static_cast<double>(hit.score);
        ordered.push_back(std::move(chunk));
    }
    if (static_cast<int>(ordered.size()) > topK) ordered.resize(topK);
    return ordered;
}

}  // namespace autumndawn::rag
