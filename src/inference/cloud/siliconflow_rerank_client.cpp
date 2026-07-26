#include "siliconflow_rerank_client.hpp"

#include <algorithm>

#include <nlohmann/json.hpp>

#include "rag/inference/inference_error.hpp"

#include "../http_client.hpp"

namespace autumndawn::rag::inference {

using json = nlohmann::json;

struct SiliconFlowRerankClient::Impl {
    std::string baseUrl;
    std::string apiKey;
    std::string model;
    detail::HttpClient http;
};

SiliconFlowRerankClient::SiliconFlowRerankClient(std::string baseUrl, std::string apiKey,
                                                 std::string model)
    : impl_(std::make_unique<Impl>()) {
    // 去掉 base_url 末尾多余的 '/'，方便拼接路径。
    while (!baseUrl.empty() && baseUrl.back() == '/') {
        baseUrl.pop_back();
    }
    impl_->baseUrl = std::move(baseUrl);
    impl_->apiKey = std::move(apiKey);
    impl_->model = std::move(model);
}

SiliconFlowRerankClient::~SiliconFlowRerankClient() = default;

std::vector<RerankHit> SiliconFlowRerankClient::rerank(const std::string& query,
                                                       const std::vector<std::string>& docs,
                                                       int topN) {
    if (docs.empty()) return {};

    // topN 截断到 docs 数量；topN <= 0 表示全部。
    const std::size_t limit =
        topN > 0 ? std::min(static_cast<std::size_t>(topN), docs.size()) : docs.size();

    json requestBody;
    requestBody["model"] = impl_->model;
    requestBody["query"] = query;
    requestBody["documents"] = docs;
    requestBody["top_n"] = limit;

    std::string url = impl_->baseUrl + "/rerank";
    std::vector<std::string> headers = {
        "Content-Type: application/json",
        "Authorization: Bearer " + impl_->apiKey,
    };

    std::string responseBody;
    try {
        responseBody = impl_->http.post(url, requestBody.dump(), headers);
    } catch (const detail::HttpError& e) {
        throw InferenceError(std::string("SiliconFlow rerank request failed: ") + e.what());
    }

    json response;
    try {
        response = json::parse(responseBody);
    } catch (const json::parse_error& e) {
        throw InferenceError(std::string("Failed to parse SiliconFlow rerank response: ") +
                             e.what());
    }

    if (!response.contains("results") || !response["results"].is_array()) {
        throw InferenceError("Unexpected SiliconFlow rerank response (missing 'results'): " +
                             responseBody);
    }

    // results 元素形如 {"index": i, "relevance_score": s}，理论上已按分数降序。
    std::vector<RerankHit> hits;
    hits.reserve(response["results"].size());
    for (const auto& item : response["results"]) {
        RerankHit hit;
        hit.index = item.at("index").get<std::size_t>();
        hit.score = item.at("relevance_score").get<float>();
        if (hit.index >= docs.size()) {
            throw InferenceError("SiliconFlow rerank response index out of range");
        }
        hits.push_back(hit);
    }

    // 防御：不假设服务端返回一定有序，显式按分数降序排一次。
    std::sort(hits.begin(), hits.end(),
              [](const RerankHit& a, const RerankHit& b) { return a.score > b.score; });
    if (hits.size() > limit) {
        hits.resize(limit);
    }
    return hits;
}

}  // namespace autumndawn::rag::inference
