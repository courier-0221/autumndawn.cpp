#include "rag/retrieval/vector_store_retriever.hpp"

#include <utility>

namespace autumndawn::rag {

VectorStoreRetriever::VectorStoreRetriever(std::shared_ptr<inference::IEmbeddingModel> embedder,
                                           std::shared_ptr<storage::IVectorStore> store)
    : embedder_(std::move(embedder)), store_(std::move(store)) {}

void VectorStoreRetriever::ingest(const std::vector<std::string>& chunks,
                                  const std::string& source) {
    if (chunks.empty()) return;
    auto vectors = embedder_->embedBatch(chunks);

    std::vector<storage::DocumentRecord> records;
    records.reserve(chunks.size());
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        storage::DocumentRecord r;
        r.id = 0;
        r.text = chunks[i];
        r.source = source;
        r.embedding = std::move(vectors[i]);
        records.push_back(std::move(r));
    }
    store_->putBatch(std::move(records));
}

std::uint64_t VectorStoreRetriever::addOne(const std::string& text, const std::string& source) {
    storage::DocumentRecord r;
    r.id = 0;
    r.text = text;
    r.source = source;
    r.embedding = embedder_->embed(text);
    return store_->put(std::move(r));
}

std::vector<RetrievedChunk> VectorStoreRetriever::retrieve(const std::string& query, int topK) {
    if (topK <= 0) return {};
    auto vec = embedder_->embed(query);
    auto hits = store_->searchByVector(vec, topK);

    std::vector<RetrievedChunk> out;
    out.reserve(hits.size());
    for (auto& h : hits) {
        RetrievedChunk c;
        c.id = h.id;
        c.text = std::move(h.text);
        c.source = std::move(h.source);
        c.score = h.score;
        out.push_back(std::move(c));
    }
    return out;
}

}  // namespace autumndawn::rag
