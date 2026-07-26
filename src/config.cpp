#include "rag/config.hpp"

#include <fstream>
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

RemoteApiConfig parseSection(const json& j, const char* section, bool require) {
    RemoteApiConfig cfg;
    if (!j.contains(section)) {
        if (require) {
            throw ConfigError(std::string("Missing config section: ") + section);
        }
        return cfg;
    }
    const auto& s = j[section];
    if (!s.is_object()) {
        throw ConfigError(std::string("Config section '") + section + "' must be an object");
    }
    for (const char* field : {"base_url", "api_key", "model"}) {
        if (!s.contains(field) || !s[field].is_string()) {
            throw ConfigError(std::string("Config '") + section + "' missing string field '" +
                              field + "'");
        }
    }
    cfg.baseUrl = s["base_url"].get<std::string>();
    cfg.apiKey = s["api_key"].get<std::string>();
    cfg.model = s["model"].get<std::string>();
    if (require && looksLikePlaceholder(cfg.apiKey)) {
        throw ConfigError(std::string("Config '") + section +
                          ".api_key' looks like a placeholder. Fill in your real key.");
    }
    return cfg;
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
    cfg.embedding = parseSection(j, "embedding", /*require=*/true);
    cfg.chat = parseSection(j, "chat", /*require=*/true);
    // rerank 可选：如果没配置或 api_key 是占位符，hasRerank = false，不抛。
    if (j.contains("rerank")) {
        auto tmp = parseSection(j, "rerank", /*require=*/false);
        if (!tmp.apiKey.empty() && !looksLikePlaceholder(tmp.apiKey)) {
            cfg.rerank = tmp;
            cfg.hasRerank = true;
        }
    }
    return cfg;
}

}  // namespace autumndawn::rag
