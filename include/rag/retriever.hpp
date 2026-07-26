#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace autumndawn::rag {

/// 检索命中的一段文本 + 元数据 + 相关性分数。
/// 上层业务只依赖这个结构，不依赖 ObjectBox 的 Document 类型。
struct RetrievedChunk {
    std::uint64_t id = 0;   ///< 后端存储层的原生 id（ObjectBox 里就是 obx_id）
    std::string text;
    std::string source;
    double score = 0.0;     ///< 越接近 0（cosine 距离）或越大（rerank 分数）取决于实现；
                            ///< 顺序总是"越靠前越相关"。
};

/// 检索器抽象：文本查询 → Top-K 相关片段。
class IRetriever {
public:
    virtual ~IRetriever() = default;

    /// @param query    用户查询文本
    /// @param topK     期望返回的最大条数（>0）
    virtual std::vector<RetrievedChunk> retrieve(const std::string& query, int topK) = 0;
};

}  // namespace autumndawn::rag
