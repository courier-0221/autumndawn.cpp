#pragma once

#include <string>

/// 从 JSON 配置文件加载 SiliconFlow 相关配置。
struct AppConfig {
    std::string baseUrl;
    std::string apiKey;
    std::string model;
};

/// 加载配置文件（默认 config.json）。
/// 若文件不存在，会给出提示（提醒从 config.example.json 复制一份并填入 api_key）。
/// 解析失败或缺少必要字段时抛出 std::runtime_error。
AppConfig loadConfig(const std::string& path = "config.json");
