#pragma once

#include <cstddef>
#include <memory>

#include "rag/config.hpp"
#include "rag/storage/vector_store.hpp"

namespace autumndawn::rag {

/// 向量存储装配工厂：按 `storage.provider` 分派到具体后端实现。
/// 与 `model_factory` 对称——上层永远只拿到 `IVectorStore` 抽象，
/// 不 include 任何后端头文件（如 `objectbox.hpp`）。
///
/// @param cfg          storage 后端配置（当前仅支持 objectbox）。
/// @param embeddingDim 期望的向量维度；后端可用于打开时校验或索引构建。
///                     传入应与 `IEmbeddingModel` 输出维度一致
///                     （通常即 `rag::kEmbeddingDim`）。
std::shared_ptr<storage::IVectorStore> createVectorStore(const StorageBackendConfig& cfg,
                                                          std::size_t embeddingDim);

}  // namespace autumndawn::rag
