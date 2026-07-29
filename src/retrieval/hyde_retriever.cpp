#include "rag/retrieval/hyde_retriever.hpp"

#include <utility>

#include "rag/log.hpp"
#include "rag/prompt_template.hpp"

namespace autumndawn::rag {

HydeRetriever::HydeRetriever(std::shared_ptr<IRetriever> base,
                             std::shared_ptr<inference::IChatModel> chat, Options options)
    : base_(std::move(base)), chat_(std::move(chat)), options_(std::move(options)) {}

std::vector<RetrievedChunk> HydeRetriever::retrieve(const std::string& query, int topK) {
    if (topK <= 0) return {};

    lastHypotheticalDoc_.clear();
    try {
        const std::string userMsg =
            prompt::format(options_.promptTemplate, {{"question", query}});
        lastHypotheticalDoc_ = chat_->chat({{"user", userMsg}}, options_.chat);
    } catch (const std::exception& e) {
        LOG(WARNING) << "HydeRetriever: hypothetical answer generation failed (" << e.what()
                     << "), falling back to the raw query";
    }

    LOG(INFO) << "HydeRetriever: hypothetical answer: " << lastHypotheticalDoc_;
    const std::string& effectiveQuery =
        lastHypotheticalDoc_.empty() ? query : lastHypotheticalDoc_;
    return base_->retrieve(effectiveQuery, topK);
}

}  // namespace autumndawn::rag
