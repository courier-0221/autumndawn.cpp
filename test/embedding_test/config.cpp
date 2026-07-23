#include "config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

AppConfig loadConfig(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.good()) {
        throw std::runtime_error("Config file not found: " + path +
                                  "\nPlease copy config.example.json to config.json and fill in your "
                                  "base_url / api_key.");
    }

    std::stringstream buffer;
    buffer << ifs.rdbuf();

    json j;
    try {
        j = json::parse(buffer.str());
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("Failed to parse ") + path + ": " + e.what());
    }

    AppConfig config;
    if (!j.contains("base_url") || !j["base_url"].is_string()) {
        throw std::runtime_error("Config missing string field 'base_url'");
    }
    if (!j.contains("api_key") || !j["api_key"].is_string()) {
        throw std::runtime_error("Config missing string field 'api_key'");
    }
    if (!j.contains("model") || !j["model"].is_string()) {
        throw std::runtime_error("Config missing string field 'model'");
    }

    config.baseUrl = j["base_url"].get<std::string>();
    config.apiKey = j["api_key"].get<std::string>();
    config.model = j["model"].get<std::string>();

    if (config.apiKey.empty() || config.apiKey.rfind("sk-xxxx", 0) == 0) {
        throw std::runtime_error("Config 'api_key' looks like a placeholder. Please fill in your real key in " +
                                  path);
    }

    return config;
}
