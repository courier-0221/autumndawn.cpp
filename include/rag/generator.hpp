#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rag/inference/chat_client.hpp"
#include "rag/retriever.hpp"

namespace autumndawn::rag {

/// Generator 抽象：把 (query, 检索上下文) 交给 LLM 生成答案。
class IGenerator {
public:
    virtual ~IGenerator() = default;

    virtual std::string generate(const std::string& query,
                                 const std::vector<RetrievedChunk>& contexts) = 0;
};

/// 基于 IChatClient 的通用 generator：用一个 prompt 模板把上下文塞进 user message。
/// 默认模板足以跑通 v0.1；后续可通过构造函数替换。
class ChatGenerator : public IGenerator {
public:
    struct Options {
        std::string systemPrompt =
            "你是一个基于给定上下文回答问题的助手。请严格根据【上下文】里的信息作答；"
            "如果上下文中没有依据，请明确回答'根据现有资料无法回答'，不要编造。";
        /// user 消息模板，占位符：{context} / {question}
        std::string userTemplate =
            "【上下文】\n{context}\n\n【问题】\n{question}\n\n请用简体中文回答，并在末尾用括号列出你引用了哪几个片段的编号（如：(#1, #3)）。";
        inference::ChatOptions chat;
    };

    explicit ChatGenerator(std::shared_ptr<inference::IChatClient> chat)
        : ChatGenerator(std::move(chat), Options{}) {}
    ChatGenerator(std::shared_ptr<inference::IChatClient> chat, Options options);

    std::string generate(const std::string& query,
                         const std::vector<RetrievedChunk>& contexts) override;

private:
    std::shared_ptr<inference::IChatClient> chat_;
    Options options_;
};

}  // namespace autumndawn::rag
