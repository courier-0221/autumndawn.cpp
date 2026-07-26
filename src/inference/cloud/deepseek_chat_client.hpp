#pragma once

// 云端实现类的内部头文件：不暴露到 include/，外部统一经 rag/model_factory.hpp 创建。

#include <memory>
#include <string>
#include <vector>

#include "rag/inference/chat_model.hpp"

namespace autumndawn::rag::inference {

/// DeepSeek Chat API 客户端。
/// OpenAI 兼容协议：POST {baseUrl}/chat/completions
class DeepSeekChatClient : public IChatModel {
public:
    DeepSeekChatClient(std::string baseUrl, std::string apiKey, std::string model);
    ~DeepSeekChatClient() override;

    std::string chat(const std::vector<ChatMessage>& messages,
                     const ChatOptions& options) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace autumndawn::rag::inference
