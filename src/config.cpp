#include "rag/config.hpp"

#include <fstream>
#include <optional>
#include <sstream>

#include <nlohmann/json.hpp>

namespace autumndawn::rag {

using json = nlohmann::json;

namespace {

bool looksLikePlaceholder(const std::string& key) {
    if (key.empty()) return true;
    // 常见占位符：sk-xxxx / sk-your-... / <fill-in> 等
    if (key.rfind("sk-xxxx", 0) == 0) return true;
    if (key.rfind("sk-your", 0) == 0) return true;
    if (key.find("fill-in") != std::string::npos) return true;
    if (key.find("<") != std::string::npos) return true;
    return false;
}

/// 受支持的 provider 列表，用于校验与报错提示。
constexpr const char* kCloudProviders = "siliconflow, deepseek";
constexpr const char* kLocalProviders = "llama_cpp, onnx";

bool isCloudProvider(const std::string& p) {
    return p == "siliconflow" || p == "deepseek";
}

bool isLocalProvider(const std::string& p) {
    return p == "llama_cpp" || p == "onnx";
}

std::string requireString(const json& s, const char* section, const char* field) {
    if (!s.contains(field) || !s[field].is_string()) {
        throw ConfigError(std::string("Config '") + section + "' missing string field '" +
                          field + "'");
    }
    return s[field].get<std::string>();
}

int optionalInt(const json& s, const char* section, const char* field, int fallback) {
    if (!s.contains(field)) return fallback;
    if (!s[field].is_number_integer()) {
        throw ConfigError(std::string("Config '") + section + "." + field +
                          "' must be an integer");
    }
    return s[field].get<int>();
}

CloudInferenceConfig parseCloud(const json& s, const char* section, std::string provider,
                                bool required) {
    CloudInferenceConfig cfg;
    cfg.provider = std::move(provider);
    cfg.baseUrl = requireString(s, section, "base_url");
    cfg.apiKey = requireString(s, section, "api_key");
    cfg.model = requireString(s, section, "model");
    if (required && looksLikePlaceholder(cfg.apiKey)) {
        throw ConfigError(std::string("Config '") + section +
                          ".api_key' looks like a placeholder. Fill in your real key.");
    }
    return cfg;
}

LocalInferenceConfig parseLocal(const json& s, const char* section, std::string provider) {
    LocalInferenceConfig cfg;
    cfg.provider = std::move(provider);
    cfg.modelPath = requireString(s, section, "model_path");
    cfg.nCtx = optionalInt(s, section, "n_ctx", cfg.nCtx);
    cfg.nThreads = optionalInt(s, section, "n_threads", cfg.nThreads);
    cfg.nGpuLayers = optionalInt(s, section, "n_gpu_layers", cfg.nGpuLayers);
    return cfg;
}

/// 解析推理段（embedding / chat / rerank）：provider 判别 cloud / local。
/// required=false 时：段缺失或 cloud api_key 为占位符返回 nullopt（视为未配置），不抛。
std::optional<InferenceBackendConfig> parseBackendSection(const json& j, const char* section,
                                                          bool required) {
    if (!j.contains(section)) {
        if (required) {
            throw ConfigError(std::string("Missing config section: ") + section);
        }
        return std::nullopt;
    }
    const auto& s = j[section];
    if (!s.is_object()) {
        throw ConfigError(std::string("Config section '") + section + "' must be an object");
    }

    std::string provider;
    if (s.contains("provider")) {
        if (!s["provider"].is_string()) {
            throw ConfigError(std::string("Config '") + section + ".provider' must be a string");
        }
        provider = s["provider"].get<std::string>();
    } else {
        throw ConfigError(std::string("Config '") + section +
                          "' missing 'provider' (supported cloud: " + kCloudProviders +
                          "; local: " + kLocalProviders + ")");
    }

    if (isCloudProvider(provider)) {
        auto cfg = parseCloud(s, section, std::move(provider), required);
        if (!required && looksLikePlaceholder(cfg.apiKey)) {
            return std::nullopt;  // 可选段的占位符 key：静默视为未启用
        }
        return cfg;
    }
    if (isLocalProvider(provider)) {
        return parseLocal(s, section, std::move(provider));
    }
    throw ConfigError(std::string("Config '") + section + ".provider' has unsupported value '" +
                      provider + "' (supported cloud: " + kCloudProviders +
                      "; local: " + kLocalProviders + ")");
}

}  // namespace

RagConfig loadRagConfig(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.good()) {
        throw ConfigError("Config file not found: " + path +
                          "\nHint: copy rag_config.example.json to rag_config.json and fill in your api keys.");
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();

    json j;
    try {
        j = json::parse(buffer.str());
    } catch (const json::parse_error& e) {
        throw ConfigError(std::string("Failed to parse ") + path + ": " + e.what());
    }

    RagConfig cfg;
    if (!j.contains("model")) {
        throw ConfigError("Missing top-level config section: model");
    }
    const auto& m = j["model"];
    if (!m.is_object()) {
        throw ConfigError("Config section 'model' must be an object");
    }
    cfg.model.embedding = *parseBackendSection(m, "embedding", /*required=*/true);
    cfg.model.chat = *parseBackendSection(m, "chat", /*required=*/true);
    cfg.model.rerank = parseBackendSection(m, "rerank", /*required=*/false);
    return cfg;
}

}  // namespace autumndawn::rag
