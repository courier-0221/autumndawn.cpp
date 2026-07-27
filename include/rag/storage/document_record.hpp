#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace autumndawn::rag::storage {

/// 向量存储层中一行"文档"的中性表示，后端无关。
///
/// 写入时 `embedding` 由上层（Indexer / VectorStoreRetriever::ingest）
/// 通过 `IEmbeddingModel` 预先算好后填入；`id` 传 0 让后端自动分配。
/// 搜索结果一般不回带 embedding（见 `IVectorStore::SearchHit`），以省流量。
struct DocumentRecord {
    std::uint64_t id = 0;
    std::string text;
    std::string source;
    std::vector<float> embedding;
};

}  // namespace autumndawn::rag::storage
