#pragma once

#include <vector>

#include "rag/retriever.hpp"

namespace autumndawn::rag {

/// 多路召回结果的融合策略（v0.2：RRF）。
namespace fusion {

/// Reciprocal Rank Fusion（RRF）：把多份"各自按相关性排好序"的召回列表
/// 融合成一份综合排名。
///
///   score(d) = Σ_i  1 / (k + rank_i(d))       （rank 从 1 开始）
///
/// 只依赖每个列表内部的名次、不依赖各后端原始分数的量纲，因此可以
/// 通吃"cosine 距离 / 内积 / rerank 分"混合的多路召回。
///
/// - 同一文档以 `RetrievedChunk::id` 去重；文本/来源取首次出现的那份。
/// - 返回结果按 RRF 得分从高到低排序，`score` 字段被替换为 RRF 得分
///   （值越大越相关）。
///
/// @param rankings 多份召回列表，每份内部按"越靠前越相关"排序。
/// @param k        平滑常数，抑制榜首差距（论文与 LangChain 默认 60）。
std::vector<RetrievedChunk> rrf(const std::vector<std::vector<RetrievedChunk>>& rankings,
                                int k = 60);

}  // namespace fusion

}  // namespace autumndawn::rag
