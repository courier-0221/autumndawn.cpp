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

/// ObjectBox 向量库后端配置。
struct ObjectBoxStorageConfig {
    std::string provider;                       ///< 固定为 "objectbox"
    std::string directory = "objectbox-db";     ///< ObjectBox 数据库目录（相对/绝对均可）
};

/// 向量存储后端联合体。未来新增 Qdrant / FAISS / in_memory 等：
/// 1) 在此增加对应 *StorageConfig；2) 加入 variant；3) 在 store_factory 里分派。
using StorageBackendConfig = std::variant<ObjectBoxStorageConfig>;

/// v0.1 及以后的统一配置：rag_config.json
///   {
///     "model":   { "embedding": {...}, "chat": {...}, "rerank": {...} },
///     "storage": { "provider": "objectbox", "directory": "objectbox-db" },
///     ...
///   }
/// - "model" 段包含三类推理后端；rerank 可选：段缺失或 cloud api_key
///   为占位符时视为未配置（nullopt），不抛错。
/// - "storage" 段可选：缺失时默认使用 ObjectBox + `objectbox-db` 目录。
/// - 后续其他顶层段（如 retrieval / splitter）在 RagConfig 内平级新增。
struct RagConfig {
    ModelConfig model;
    StorageBackendConfig storage;
};

RagConfig loadRagConfig(const std::string& path);

}  // namespace autumndawn::rag
