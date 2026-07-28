#pragma once

#include <cstddef>
#include <memory>

#include "rag/config.hpp"
#include "rag/inference/chat_model.hpp"
#include "rag/inference/embedding_model.hpp"
#include "rag/inference/rerank_model.hpp"

namespace autumndawn::rag {

/// 推理模型装配工厂：根据配置里的 provider 判别 cloud / local，返回对应实现。
/// 工厂放在 rag 顶层而非 inference 层——inference 层只定义接口与实现，
/// 不感知 config 的存在（分层约定见 doc/rag_cpp_plan.md）。
///
/// 当前 provider 支持情况：
/// - cloud（siliconflow / deepseek）→ 对应云端 *Client；
/// - local（llama_cpp / onnx）→ 端侧进程内推理本期尚未实现，抛 ConfigError。

/// 创建 embedding 模型。
/// expectDim: 期望的向量维度（传 kEmbeddingDim，与 ObjectBox schema 对齐），0 表示不校验。
std::shared_ptr<inference::IEmbeddingModel> createEmbeddingModel(
    const InferenceBackendConfig& cfg, std::size_t expectDim);

/// 创建 chat 模型。
std::shared_ptr<inference::IChatModel> createChatModel(const InferenceBackendConfig& cfg);

/// 创建 rerank 模型。
/// 调用前先判断 RagConfig::model.rerank 是否有值（nullopt 表示未启用 rerank）。
std::shared_ptr<inference::IRerankModel> createRerankModel(const InferenceBackendConfig& cfg);

}  // namespace autumndawn::rag
