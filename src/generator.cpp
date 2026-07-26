#include "rag/generator.hpp"

#include "rag/prompt_template.hpp"

namespace autumndawn::rag {

ChatGenerator::ChatGenerator(std::shared_ptr<inference::IChatModel> chat, Options options)
    : chat_(std::move(chat)), options_(std::move(options)) {}

std::string ChatGenerator::generate(const std::string& query,
                                    const std::vector<RetrievedChunk>& contexts) {
    // 把每条 chunk 编号后拼成上下文块，方便 LLM 在末尾引用 (#1, #3) 等编号。
    std::string contextStr;
    for (std::size_t i = 0; i < contexts.size(); ++i) {
        contextStr += "#" + std::to_string(i + 1);
        if (!contexts[i].source.empty()) {
            contextStr += " (" + contexts[i].source + ")";
        }
        contextStr += ":\n";
        contextStr += contexts[i].text;
        contextStr += "\n\n";
    }

    std::string userMsg =
        prompt::format(options_.userTemplate, {{"context", contextStr}, {"question", query}});

    std::vector<inference::ChatMessage> messages = {
        {"system", options_.systemPrompt},
        {"user", userMsg},
    };

    return chat_->chat(messages, options_.chat);
}

}  // namespace autumndawn::rag
