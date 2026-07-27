#include "rag/model_factory.hpp"

#include <type_traits>
#include <variant>

#include "inference/cloud/deepseek_chat_client.hpp"
#include "inference/cloud/siliconflow_embedding_client.hpp"
#include "inference/cloud/siliconflow_rerank_client.hpp"

namespace autumndawn::rag {

namespace {

[[noreturn]] void throwLocalNotImplemented(const std::string& provider) {
    throw ConfigError("provider '" + provider +
                      "' 为端侧进程内推理，本期尚未实现（规划中：llama.cpp / onnx）");
}

[[noreturn]] void throwUnsupportedCloudProvider(const char* section,
                                                const std::string& provider,
                                                const char* supported) {
    throw ConfigError(std::string("Cloud provider '") + provider + "' 不支持 " + section +
                      " 段（当前支持：" + supported + "）");
}

}  // namespace

std::shared_ptr<inference::IEmbeddingModel> createEmbeddingModel(
    const InferenceBackendConfig& cfg, std::size_t expectDim) {
    return std::visit(
        [expectDim](const auto& c) -> std::shared_ptr<inference::IEmbeddingModel> {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, CloudInferenceConfig>) {
                if (c.provider == "siliconflow") {
                    return std::make_shared<inference::SiliconFlowEmbeddingClient>(
                        c.baseUrl, c.apiKey, c.model, expectDim);
                }
                throwUnsupportedCloudProvider("embedding", c.provider, "siliconflow");
            } else {
                throwLocalNotImplemented(c.provider);
            }
        },
        cfg);
}

std::shared_ptr<inference::IChatModel> createChatModel(const InferenceBackendConfig& cfg) {
    return std::visit(
        [](const auto& c) -> std::shared_ptr<inference::IChatModel> {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, CloudInferenceConfig>) {
                if (c.provider == "deepseek") {
                    return std::make_shared<inference::DeepSeekChatClient>(c.baseUrl, c.apiKey,
                                                                           c.model);
                }
                throwUnsupportedCloudProvider("chat", c.provider, "deepseek");
            } else {
                throwLocalNotImplemented(c.provider);
            }
        },
        cfg);
}

std::shared_ptr<inference::IRerankModel> createRerankModel(const InferenceBackendConfig& cfg) {
    return std::visit(
        [](const auto& c) -> std::shared_ptr<inference::IRerankModel> {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, CloudInferenceConfig>) {
                if (c.provider == "siliconflow") {
                    return std::make_shared<inference::SiliconFlowRerankClient>(
                        c.baseUrl, c.apiKey, c.model);
                }
                throwUnsupportedCloudProvider("rerank", c.provider, "siliconflow");
            } else {
                throwLocalNotImplemented(c.provider);
            }
        },
        cfg);
}

}  // namespace autumndawn::rag
