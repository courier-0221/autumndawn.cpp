#include "rag/fusion.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace autumndawn::rag {
namespace fusion {

std::vector<RetrievedChunk> rrf(const std::vector<std::vector<RetrievedChunk>>& rankings,
                                int k) {
    if (k < 0) k = 0;

    // id → 融合结果（首次出现时记录 chunk 内容，之后只累加得分）。
    std::unordered_map<std::uint64_t, std::size_t> indexById;
    std::vector<RetrievedChunk> fused;

    for (const auto& ranking : rankings) {
        for (std::size_t r = 0; r < ranking.size(); ++r) {
            const auto& chunk = ranking[r];
            const double gain = 1.0 / (k + static_cast<double>(r) + 1.0);

            auto it = indexById.find(chunk.id);
            if (it == indexById.end()) {
                RetrievedChunk c = chunk;
                c.score = gain;
                indexById.emplace(chunk.id, fused.size());
                fused.push_back(std::move(c));
            } else {
                fused[it->second].score += gain;
            }
        }
    }

    std::stable_sort(fused.begin(), fused.end(),
                     [](const RetrievedChunk& a, const RetrievedChunk& b) {
                         return a.score > b.score;
                     });
    return fused;
}

}  // namespace fusion
}  // namespace autumndawn::rag
