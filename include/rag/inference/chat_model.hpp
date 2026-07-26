#pragma once

#include <string>
#include <vector>

namespace autumndawn::rag::inference {

struct ChatMessage {
    std::string role;     // "system" / "user" / "assistant"
    std::string content;
};

struct ChatOptions {
    float temperature = 0.2f;
    int maxTokens = 1024;        // 0 表示由服务端/引擎默认
    // 未来可加 top_p / stop / stream，v0.1 保持简单
};

/// 与 chat 模型交互的抽象接口。
/// 部署形态无关：云端 API（DeepSeek 等）与端侧进程内推理（llama.cpp，规划中）均实现本接口。
/// 具体实现类不暴露在公共头文件中，统一经 rag/model_factory.hpp 的工厂函数按配置创建。
///
/// 线程安全契约：实现不保证线程安全，调用方需串行访问。
///
/// 演进预留：端侧推理落地时预计补充流式接口
///   chatStream(messages, options, std::function<void(std::string_view)> onToken)
/// 计划以带默认实现（同步跑完后一次性回调）的方式加入，向后兼容。
class IChatModel {
public:
    virtual ~IChatModel() = default;

    /// 传入多轮消息，返回 assistant 回复文本。
    virtual std::string chat(const std::vector<ChatMessage>& messages,
                             const ChatOptions& options) = 0;

    /// 便捷重载：使用默认 ChatOptions。
    std::string chat(const std::vector<ChatMessage>& messages) {
        return chat(messages, ChatOptions{});
    }
};

}  // namespace autumndawn::rag::inference
