#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace autumndawn::rag::detail {

/// HTTP 请求错误：网络失败 / 超时 / 非 2xx 状态码。
class HttpError : public std::runtime_error {
public:
    HttpError(const std::string& what, long httpCode = 0)
        : std::runtime_error(what), httpCode_(httpCode) {}
    long httpCode() const noexcept { return httpCode_; }

private:
    long httpCode_;
};

/// 通用同步 HTTP 客户端（libcurl 包装）。
/// - 保留 TLS 证书校验（不关闭 VERIFYPEER/HOST）。
/// - 支持超时。
/// - 对 429 / 5xx 做指数回退重试。
class HttpClient {
public:
    struct Options {
        long timeoutSec = 60;
        long connectTimeoutSec = 10;
        int maxRetries = 3;         // 遇到 429 / 5xx 时的最大重试次数（不含首次）
        int baseBackoffMs = 500;    // 指数回退基数：500ms, 1s, 2s, ...
    };

    HttpClient() : HttpClient(Options{}) {}
    explicit HttpClient(Options options);

    /// 发起 POST 请求，返回响应 body。
    /// headers 例如 {"Content-Type: application/json", "Authorization: Bearer xxx"}
    /// 非 2xx 状态码时抛 HttpError，errorBody 保留响应体供上层解析。
    std::string post(const std::string& url,
                     const std::string& body,
                     const std::vector<std::string>& headers) const;

private:
    Options options_;
};

}  // namespace autumndawn::rag::detail
