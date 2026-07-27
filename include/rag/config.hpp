#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace autumndawn::rag {

/// 配置文件的通用错误。
class ConfigError : public std::runtime_error {
public:
    explicit ConfigError(const std::string& what) : std::runtime_error(what) {}
};

/// provider 接受具名厂商标识："siliconflow" / "deepseek"，
/// model_factory 会根据 provider 分派到具体 client。
struct CloudInferenceConfig {
    std::string provider;
    std::string baseUrl;
    std::string apiKey;
    std::string model;
};

/// 端侧进程内推理段（llama.cpp / onnx）。
/// 本期仅做配置解析；工厂会对该分支抛 ConfigError，实现待后续版本落地。
struct LocalInferenceConfig {
    std::string provider;  // "llama_cpp" / "onnx"
    std::string modelPath;
    int nCtx = 4096;       ///< chat 上下文长度（local chat 用）
    int nThreads = 0;      ///< 0 = 由引擎自动选择
    int nGpuLayers = 0;    ///< 0 = 纯 CPU
};

using InferenceBackendConfig = std::variant<CloudInferenceConfig, LocalInferenceConfig>;

/// ObjectBox Document schema 中 HNSW 索引写死的向量维度（bge-m3）。
/// 切换 embedding 模型时若维度不同，需要同步修改 schema 并重建库。
inline constexpr std::size_t kEmbeddingDim = 1024;

/// 推理模型段：embedding / chat / rerank。
/// 三个段结构一致，均由各自的 "provider" 字段判别 cloud / local。
struct ModelConfig {
    InferenceBackendConfig embedding;
    InferenceBackendConfig chat;
    std::optional<InferenceBackendConfig> rerank;  ///< nullopt = 未启用 rerank
};

/// v0.1 及以后的统一配置：rag_config.json
///   { "model": { "embedding": {...}, "chat": {...}, "rerank": {...} }, ... }
/// - "model" 段包含三类推理后端；rerank 可选：段缺失或 cloud api_key
///   为占位符时视为未配置（nullopt），不抛错。
/// - 后续其他顶层段（如 retrieval / splitter / storage）在 RagConfig 内平级新增。
struct RagConfig {
    ModelConfig model;
};

RagConfig loadRagConfig(const std::string& path);

}  // namespace autumndawn::rag
