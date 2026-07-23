#pragma once

#include <stdexcept>
#include <string>
#include <vector>

/// 硅基流动（SiliconFlow）Embedding API 的最小封装。
/// 接口协议与 OpenAI /v1/embeddings 兼容：
///   POST {base_url}/embeddings
///   Authorization: Bearer {api_key}
///   { "model": "...", "input": ["text1", "text2"] }
class SiliconFlowEmbeddingClient {
public:
    SiliconFlowEmbeddingClient(std::string baseUrl, std::string apiKey, std::string model);

    /// 单条文本 -> 向量
    std::vector<float> embed(const std::string& text);

    /// 批量文本 -> 向量列表（一次 HTTP 请求）
    std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts);

private:
    std::string baseUrl_;
    std::string apiKey_;
    std::string model_;

    /// 向 {baseUrl_}/embeddings 发起 POST 请求，返回响应 body。
    /// 若网络失败或 HTTP 状态码非 2xx，抛出 std::runtime_error。
    std::string post(const std::string& jsonBody) const;
};

/// Embedding 请求/响应相关的运行时错误。
class EmbeddingClientError : public std::runtime_error {
public:
    explicit EmbeddingClientError(const std::string& what) : std::runtime_error(what) {}
};
