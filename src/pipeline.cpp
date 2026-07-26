#include "rag/pipeline.hpp"

namespace autumndawn::rag {

RagPipeline::RagPipeline(std::shared_ptr<IRetriever> retriever,
                         std::shared_ptr<IGenerator> generator)
    : retriever_(std::move(retriever)), generator_(std::move(generator)) {}

RagPipeline::AskResult RagPipeline::ask(const std::string& query, int topK) {
    AskResult r;
    r.contexts = retriever_->retrieve(query, topK);
    r.answer = generator_->generate(query, r.contexts);
    return r;
}

}  // namespace autumndawn::rag
