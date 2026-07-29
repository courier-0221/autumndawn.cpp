#include "rag/retrieval/multi_query_retriever.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <utility>

#include "rag/fusion.hpp"
#include "rag/log.hpp"
#include "rag/prompt_template.hpp"

namespace autumndawn::rag {

namespace {

/// 去掉行首的编号 / 列表前缀（"1. " "2、" "- " "• " 等）和首尾空白。
std::string cleanLine(const std::string& raw) {
    std::size_t begin = 0;
    std::size_t end = raw.size();
    auto isSpace = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r'; };
    while (begin < end && isSpace(raw[begin])) ++begin;
    while (end > begin && isSpace(raw[end - 1])) --end;

    std::string s = raw.substr(begin, end - begin);

    // 数字编号："12." / "3、" / "4)" / "5:"
    std::size_t i = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    if (i > 0 && i < s.size()) {
        if (s[i] == '.' || s[i] == ')' || s[i] == ':') {
            s = s.substr(i + 1);
        } else if (s.compare(i, 3, "\xE3\x80\x81") == 0) {  // '、'
            s = s.substr(i + 3);
        }
    }
    // 列表符号："- " "* "
    if (!s.empty() && (s[0] == '-' || s[0] == '*')) s = s.substr(1);

    while (!s.empty() && isSpace(s.front())) s.erase(s.begin());
    while (!s.empty() && isSpace(s.back())) s.pop_back();
    return s;
}

}  // namespace

MultiQueryRetriever::MultiQueryRetriever(std::shared_ptr<IRetriever> base,
                                         std::shared_ptr<inference::IChatModel> chat,
                                         Options options)
    : base_(std::move(base)), chat_(std::move(chat)), options_(std::move(options)) {}

std::vector<std::string> MultiQueryRetriever::generateQueries(const std::string& query) {
    const std::string userMsg = prompt::format(
        options_.promptTemplate,
        {{"question", query}, {"n", std::to_string(options_.numQueries)}});

    const std::string reply = chat_->chat({{"user", userMsg}}, options_.chatOpt);

    std::vector<std::string> queries;
    std::size_t pos = 0;
    while (pos <= reply.size() && static_cast<int>(queries.size()) < options_.numQueries) {
        std::size_t nl = reply.find('\n', pos);
        const std::string line =
            reply.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        std::string q = cleanLine(line);
        if (!q.empty() && q != query) queries.push_back(std::move(q));
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    return queries;
}

std::vector<RetrievedChunk> MultiQueryRetriever::retrieve(const std::string& query, int topK) {
    if (topK <= 0) return {};

    lastQueries_.clear();
    if (options_.includeOriginal) lastQueries_.push_back(query);

    std::vector<std::string> rewrites;
    try {
        rewrites = generateQueries(query);
    } catch (const std::exception& e) {
        LOG(WARNING) << "MultiQueryRetriever: query rewriting failed (" << e.what()
                     << "), falling back to the original query only";
    }

    for (const auto& q : rewrites) {
        LOG(INFO) << "MultiQueryRetriever: rewritten query: " << q;
    }

    lastQueries_.insert(lastQueries_.end(), rewrites.begin(), rewrites.end());
    if (lastQueries_.empty()) lastQueries_.push_back(query);

    // 各路独立召回。每路仍取 topK，给合并留足候选。
    std::vector<std::vector<RetrievedChunk>> rankings;
    rankings.reserve(lastQueries_.size());
    for (const auto& q : lastQueries_) {
        rankings.push_back(base_->retrieve(q, topK));
    }

    std::vector<RetrievedChunk> merged;
    if (options_.merge == MergeStrategy::Rrf) {
        merged = fusion::rrf(rankings, options_.rrfK);
    } else {
        // Union：按名次轮转交错（第 1 名们、第 2 名们、……），以 id 去重。
        // 不比较各路原始分数（量纲可能不同），只依赖"每路内部有序"。
        std::unordered_set<std::uint64_t> seen;
        std::size_t maxLen = 0;
        for (const auto& r : rankings) maxLen = std::max(maxLen, r.size());
        for (std::size_t rank = 0; rank < maxLen; ++rank) {
            for (const auto& r : rankings) {
                if (rank >= r.size()) continue;
                if (seen.insert(r[rank].id).second) merged.push_back(r[rank]);
            }
        }
    }

    if (static_cast<int>(merged.size()) > topK) merged.resize(topK);
    return merged;
}

}  // namespace autumndawn::rag
