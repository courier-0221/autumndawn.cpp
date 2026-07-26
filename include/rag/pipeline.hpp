#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rag/generator.hpp"
#include "rag/retriever.hpp"

namespace autumndawn::rag {

/// RAG 流水线组合：Query → Retriever → Generator。
/// v0.1 是最直白的形式；后续版本会通过装饰 IRetriever 引入 multi-query / rerank / router 等。
class RagPipeline {
public:
    struct AskResult {
        std::string answer;
        std::vector<RetrievedChunk> contexts;
    };

    RagPipeline(std::shared_ptr<IRetriever> retriever, std::shared_ptr<IGenerator> generator);

    /// 检索 topK 上下文并交给 generator 生成答案。
    AskResult ask(const std::string& query, int topK = 5);

private:
    std::shared_ptr<IRetriever> retriever_;
    std::shared_ptr<IGenerator> generator_;
};

}  // namespace autumndawn::rag
