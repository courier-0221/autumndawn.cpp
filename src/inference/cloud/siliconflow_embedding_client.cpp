#include "rag/inference/embedding_client.hpp"

#include <nlohmann/json.hpp>

#include "../http_client.hpp"

namespace autumndawn::rag::inference {

using json = nlohmann::json;

struct SiliconFlowEmbeddingClient::Impl {
    std::string baseUrl;
    std::string apiKey;
    std::string model;
    detail::HttpClient http;
};

SiliconFlowEmbeddingClient::SiliconFlowEmbeddingClient(std::string baseUrl,
                                                       std::string apiKey,
                                                       std::string model,
                                                       std::size_t dim)
    : impl_(std::make_unique<Impl>()), dim_(dim) {
    // 去掉 base_url 末尾多余的 '/'，方便拼接路径。
    while (!baseUrl.empty() && baseUrl.back() == '/') {
        baseUrl.pop_back();
    }
    impl_->baseUrl = std::move(baseUrl);
    impl_->apiKey = std::move(apiKey);
    impl_->model = std::move(model);
}

SiliconFlowEmbeddingClient::~SiliconFlowEmbeddingClient() = default;

std::vector<std::vector<float>> SiliconFlowEmbeddingClient::embedBatch(
    const std::vector<std::string>& texts) {
    if (texts.empty()) return {};

    json requestBody;
    requestBody["model"] = impl_->model;
    requestBody["input"] = texts;

    std::string url = impl_->baseUrl + "/embeddings";
    std::vector<std::string> headers = {
        "Content-Type: application/json",
        "Authorization: Bearer " + impl_->apiKey,
    };

    std::string responseBody;
    try {
        responseBody = impl_->http.post(url, requestBody.dump(), headers);
    } catch (const detail::HttpError& e) {
        throw InferenceError(std::string("SiliconFlow embedding request failed: ") + e.what());
    }

    json response;
    try {
        response = json::parse(responseBody);
    } catch (const json::parse_error& e) {
        throw InferenceError(std::string("Failed to parse SiliconFlow response: ") + e.what());
    }

    if (!response.contains("data") || !response["data"].is_array()) {
        throw InferenceError("Unexpected SiliconFlow response (missing 'data'): " + responseBody);
    }

    // data 中的顺序理论上按 index 升序返回，这里按 index 显式排列，避免顺序错乱。
    std::vector<std::vector<float>> result(texts.size());
    for (const auto& item : response["data"]) {
        std::size_t index = item.at("index").get<std::size_t>();
        if (index >= result.size()) {
            throw InferenceError("SiliconFlow response index out of range");
        }
        result[index] = item.at("embedding").get<std::vector<float>>();
    }

    // 维度校验（首次调用时最有用，防止配置里模型写错导致后续 ObjectBox 写入失败）。
    if (dim_ > 0 && !result.empty() && result[0].size() != dim_) {
        throw InferenceError("SiliconFlow embedding dimension mismatch: expected " +
                             std::to_string(dim_) + ", got " + std::to_string(result[0].size()));
    }

    return result;
}

std::vector<float> SiliconFlowEmbeddingClient::embed(const std::string& text) {
    auto vectors = embedBatch({text});
    if (vectors.empty()) {
        throw InferenceError("Empty embedding result");
    }
    return vectors[0];
}

}  // namespace autumndawn::rag::inference
