#include "deepseek_chat_client.hpp"

#include <nlohmann/json.hpp>

#include "rag/inference/inference_error.hpp"

#include "../http_client.hpp"

namespace autumndawn::rag::inference {

using json = nlohmann::json;

struct DeepSeekChatClient::Impl {
    std::string baseUrl;
    std::string apiKey;
    std::string model;
    detail::HttpClient http;
};

DeepSeekChatClient::DeepSeekChatClient(std::string baseUrl, std::string apiKey, std::string model)
    : impl_(std::make_unique<Impl>()) {
    while (!baseUrl.empty() && baseUrl.back() == '/') {
        baseUrl.pop_back();
    }
    impl_->baseUrl = std::move(baseUrl);
    impl_->apiKey = std::move(apiKey);
    impl_->model = std::move(model);
}

DeepSeekChatClient::~DeepSeekChatClient() = default;

std::string DeepSeekChatClient::chat(const std::vector<ChatMessage>& messages,
                                     const ChatOptions& options) {
    if (messages.empty()) {
        throw InferenceError("chat() called with empty messages");
    }

    json requestBody;
    requestBody["model"] = impl_->model;
    requestBody["temperature"] = options.temperature;
    if (options.maxTokens > 0) {
        requestBody["max_tokens"] = options.maxTokens;
    }

    json msgs = json::array();
    for (const auto& m : messages) {
        msgs.push_back({{"role", m.role}, {"content", m.content}});
    }
    requestBody["messages"] = std::move(msgs);

    std::string url = impl_->baseUrl + "/chat/completions";
    std::vector<std::string> headers = {
        "Content-Type: application/json",
        "Authorization: Bearer " + impl_->apiKey,
    };

    std::string responseBody;
    try {
        responseBody = impl_->http.post(url, requestBody.dump(), headers);
    } catch (const detail::HttpError& e) {
        throw InferenceError(std::string("DeepSeek chat request failed: ") + e.what());
    }

    json response;
    try {
        response = json::parse(responseBody);
    } catch (const json::parse_error& e) {
        throw InferenceError(std::string("Failed to parse DeepSeek response: ") + e.what());
    }

    // LOG(INFO) << "DeepSeek chat response: " << response.dump();

    if (!response.contains("choices") || !response["choices"].is_array() ||
        response["choices"].empty()) {
        throw InferenceError("Unexpected DeepSeek response (missing 'choices'): " + responseBody);
    }

    const auto& first = response["choices"][0];
    if (!first.contains("message") || !first["message"].contains("content")) {
        throw InferenceError("Unexpected DeepSeek response (missing message.content): " +
                             responseBody);
    }

    return first["message"]["content"].get<std::string>();
}

}  // namespace autumndawn::rag::inference
