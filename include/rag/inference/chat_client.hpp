#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rag/inference/embedding_client.hpp"  // 复用 InferenceError

namespace autumndawn::rag::inference {

struct ChatMessage {
    std::string role;     // "system" / "user" / "assistant"
    std::string content;
};

struct ChatOptions {
    float temperature = 0.2f;
    int maxTokens = 1024;        // 0 表示由服务端默认
    // 未来可加 top_p / stop / stream，v0.1 保持简单
};

/// 与 chat 模型交互的抽象接口。
class IChatClient {
public:
    virtual ~IChatClient() = default;

    /// 传入多轮消息，返回 assistant 回复文本。
    virtual std::string chat(const std::vector<ChatMessage>& messages,
                             const ChatOptions& options) = 0;

    /// 便捷重载：使用默认 ChatOptions。
    std::string chat(const std::vector<ChatMessage>& messages) {
        return chat(messages, ChatOptions{});
    }
};

/// DeepSeek Chat API 客户端。
/// OpenAI 兼容协议：POST {baseUrl}/chat/completions
class DeepSeekChatClient : public IChatClient {
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
