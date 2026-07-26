#include "http_client.hpp"

#include <chrono>
#include <curl/curl.h>
#include <thread>

namespace autumndawn::rag::detail {

namespace {

size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

bool isRetriableStatus(long code) {
    return code == 429 || (code >= 500 && code < 600);
}

}  // namespace

HttpClient::HttpClient(Options options) : options_(options) {}

std::string HttpClient::post(const std::string& url,
                             const std::string& body,
                             const std::vector<std::string>& headers) const {
    std::string lastError;
    long lastHttpCode = 0;

    for (int attempt = 0; attempt <= options_.maxRetries; ++attempt) {
        if (attempt > 0) {
            int backoffMs = options_.baseBackoffMs * (1 << (attempt - 1));
            std::this_thread::sleep_for(std::chrono::milliseconds(backoffMs));
        }

        CURL* curl = curl_easy_init();
        if (!curl) {
            throw HttpError("curl_easy_init() failed");
        }

        std::string responseBody;
        struct curl_slist* headerList = nullptr;
        for (const auto& h : headers) {
            headerList = curl_slist_append(headerList, h.c_str());
        }

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, options_.timeoutSec);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, options_.connectTimeoutSec);
        // 保持默认的 TLS 证书校验：不关闭 VERIFYPEER/VERIFYHOST。

        CURLcode res = curl_easy_perform(curl);
        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

        curl_slist_free_all(headerList);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            lastError = std::string("HTTP transport error: ") + curl_easy_strerror(res);
            lastHttpCode = 0;
            // 网络级错误也重试（curl 超时、连接失败等）
            continue;
        }

        if (httpCode >= 200 && httpCode < 300) {
            return responseBody;
        }

        lastError = "HTTP " + std::to_string(httpCode) + ", body: " + responseBody;
        lastHttpCode = httpCode;

        if (!isRetriableStatus(httpCode)) {
            break;  // 4xx（非 429）不重试
        }
    }

    throw HttpError(lastError, lastHttpCode);
}

}  // namespace autumndawn::rag::detail
