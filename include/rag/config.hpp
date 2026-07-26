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

/// 云端推理段（OpenAI 兼容协议）。
/// provider 接受："siliconflow" / "deepseek" / "openai_compatible"。
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

/// embedding / chat / rerank 段的后端配置：由 provider 判别 cloud / local。
using InferenceBackendConfig = std::variant<CloudInferenceConfig, LocalInferenceConfig>;

/// ObjectBox Document schema 中 HNSW 索引写死的向量维度（bge-m3）。
/// 切换 embedding 模型时若维度不同，需要同步修改 schema 并重建库。
inline constexpr std::size_t kEmbeddingDim = 1024;

/// v0.1 及以后的统一配置：rag_config.json
///   { "embedding": {...}, "chat": {...}, "rerank": {...} }
/// - 三个段结构一致，均由 "provider" 字段判别 cloud / local；
///   provider 缺省且存在 base_url 时按 cloud 解析（向后兼容旧配置）。
/// - rerank 段可选：段缺失或 cloud api_key 为占位符时视为未配置（nullopt），不抛错。
struct RagConfig {
    InferenceBackendConfig embedding;
    InferenceBackendConfig chat;
    std::optional<InferenceBackendConfig> rerank;  ///< nullopt = 未启用 rerank
};

/// 从 JSON 文件加载 RagConfig。
/// 失败抛 ConfigError。cloud 段会拒绝 api_key 为空 / 明显 placeholder 的配置。
RagConfig loadRagConfig(const std::string& path);

}  // namespace autumndawn::rag
