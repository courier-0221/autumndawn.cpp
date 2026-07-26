#include "siliconflow_client.hpp"

#include <curl/curl.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

}  // namespace

SiliconFlowEmbeddingClient::SiliconFlowEmbeddingClient(std::string baseUrl, std::string apiKey, std::string model)
    : baseUrl_(std::move(baseUrl)), apiKey_(std::move(apiKey)), model_(std::move(model)) {
    // 去掉 base_url 末尾多余的 '/'，方便拼接路径。
    while (!baseUrl_.empty() && baseUrl_.back() == '/') {
        baseUrl_.pop_back();
    }
}

std::string SiliconFlowEmbeddingClient::post(const std::string& jsonBody) const {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw EmbeddingClientError("curl_easy_init() failed");
    }

    std::string url = baseUrl_ + "/embeddings";
    std::string responseBody;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string authHeader = "Authorization: Bearer " + apiKey_;
    headers = curl_slist_append(headers, authHeader.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonBody.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(jsonBody.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    // 保持默认的 TLS 证书校验（不关闭 CURLOPT_SSL_VERIFYPEER/HOST）。

    CURLcode res = curl_easy_perform(curl);
    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw EmbeddingClientError(std::string("HTTP request failed: ") + curl_easy_strerror(res));
    }
    if (httpCode < 200 || httpCode >= 300) {
        throw EmbeddingClientError("SiliconFlow API returned HTTP " + std::to_string(httpCode) +
                                    ", body: " + responseBody);
    }

    return responseBody;
}

std::vector<std::vector<float>> SiliconFlowEmbeddingClient::embedBatch(const std::vector<std::string>& texts) {
    if (texts.empty()) return {};

    json requestBody;
    requestBody["model"] = model_;
    requestBody["input"] = texts;

    std::string responseBody = post(requestBody.dump());

    json response;
    try {
        response = json::parse(responseBody);
    } catch (const json::parse_error& e) {
        throw EmbeddingClientError(std::string("Failed to parse SiliconFlow response: ") + e.what());
    }

    if (!response.contains("data") || !response["data"].is_array()) {
        throw EmbeddingClientError("Unexpected SiliconFlow response (missing 'data'): " + responseBody);
    }

    // data 中的顺序理论上按 index 升序返回，这里按 index 显式排列，避免顺序错乱。
    std::vector<std::vector<float>> result(texts.size());
    for (const auto& item : response["data"]) {
        size_t index = item.at("index").get<size_t>();
        if (index >= result.size()) {
            throw EmbeddingClientError("SiliconFlow response index out of range");
        }
        result[index] = item.at("embedding").get<std::vector<float>>();
    }

    return result;
}

std::vector<float> SiliconFlowEmbeddingClient::embed(const std::string& text) {
    auto vectors = embedBatch({text});
    if (vectors.empty()) {
        throw EmbeddingClientError("Empty embedding result");
    }
    return vectors[0];
}
