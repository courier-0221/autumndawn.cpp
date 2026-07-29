#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rag/inference/chat_model.hpp"
#include "rag/retriever.hpp"

namespace autumndawn::rag {

/// HyDE 检索装饰器（课程小节 9，Hypothetical Document Embeddings）。
///
/// 先让 `IChatModel` 针对问题"假答"出一段假设性文档，再拿这段文档
/// （而非原始问题）去底层 `IRetriever` 做向量检索——文档与文档在
/// embedding 空间里比"短问题 vs 长文档"更对称，往往能改善召回。
///
/// LLM 调用失败时降级为直接用原始问题检索。
class HydeRetriever : public IRetriever {
public:
    struct Options {
        /// 生成假设性文档的 prompt，占位符：{question}。
        std::string promptTemplate =
            "请写一段简短的中文说明文来回答下面的问题。即使不完全确定，"
            "也请给出最可能的内容，不要拒答、不要说明你在假设。"
            "只输出这段文字本身，不要任何前后缀。\n\n"
            "问题：{question}";
        inference::ChatOptions chat;
    };

    HydeRetriever(std::shared_ptr<IRetriever> base,
                  std::shared_ptr<inference::IChatModel> chat)
        : HydeRetriever(std::move(base), std::move(chat), Options{}) {}
    HydeRetriever(std::shared_ptr<IRetriever> base,
                  std::shared_ptr<inference::IChatModel> chat, Options options);

    std::vector<RetrievedChunk> retrieve(const std::string& query, int topK) override;

    /// 最近一次 retrieve 生成的假设性文档，供演示程序打印。
    const std::string& lastHypotheticalDoc() const { return lastHypotheticalDoc_; }

private:
    std::shared_ptr<IRetriever> base_;
    std::shared_ptr<inference::IChatModel> chat_;
    Options options_;
    std::string lastHypotheticalDoc_;
};

}  // namespace autumndawn::rag
