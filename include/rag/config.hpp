#pragma once

#include <stdexcept>
#include <string>

namespace autumndawn::rag {

/// 配置文件的通用错误。
class ConfigError : public std::runtime_error {
public:
    explicit ConfigError(const std::string& what) : std::runtime_error(what) {}
};

/// 一个云端 API 段（embedding / chat / rerank 通用结构）。
struct RemoteApiConfig {
    std::string baseUrl;
    std::string apiKey;
    std::string model;
};

/// v0.1 及以后的统一配置：rag_config.json
///   { "embedding": {...}, "chat": {...}, "rerank": {...} }
/// - rerank 段在 v0.1 可选（未启用），加载时不做强校验。
struct RagConfig {
    RemoteApiConfig embedding;
    RemoteApiConfig chat;
    RemoteApiConfig rerank;  ///< 可能为空（api_key 为空表示未配置）
    bool hasRerank = false;
};

/// 从 JSON 文件加载 RagConfig。
/// 失败抛 ConfigError。会拒绝 api_key 为空 / 明显 placeholder 的配置。
RagConfig loadRagConfig(const std::string& path);

}  // namespace autumndawn::rag
