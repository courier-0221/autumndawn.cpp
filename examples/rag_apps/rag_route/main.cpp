// rag_route —— v0.3 (Phase C) Logical Routing + Re-ranking 演示。
//
// 关键新点：
//   1. LlmRouter：让 LLM 输出 JSON {"route","reason"} 选中一条候选 pipeline（课程 10a）；
//   2. RerankingRetriever：先向量粗召回 N 条，再用 bge-reranker 精排出 K 条（课程 15）。
//
// 四种策略（--strategy= 或 strategy 命令切换）：
//   plain   全库向量检索（v0.1 基线）
//   route   Router 选源 → 该源子检索器
//   rerank  全库向量检索 → rerank 精排
//   full    Router 选源 → 子检索器 → rerank 精排（v0.3 主打链路）
//
// 命令：
//   ingest <path> <source>   读入文本、切块、embedding、以 <source> 为域标签写入
//   route <question>         只跑一次 LLM 路由，打印 {route, reason}
//   ask <question> [K]       用当前策略检索 topK 并生成答案
//   compare <question> [K]   四种策略同题对比检索差异
//   strategy [name]          查看 / 切换当前策略
//   ls / clear / help / exit

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rag/config.hpp"
#include "rag/generator.hpp"
#include "rag/log.hpp"
#include "rag/model_factory.hpp"
#include "rag/pipeline.hpp"
#include "rag/retrieval/reranking_retriever.hpp"
#include "rag/retrieval/vector_store_retriever.hpp"
#include "rag/router.hpp"
#include "rag/store_factory.hpp"
#include "rag/text_splitter.hpp"

namespace rag = autumndawn::rag;

namespace {

constexpr const char* kStrategies[] = {"plain", "route", "rerank", "full"};

/// 演示用的域标签：ingest 时把这个字符串当作 source 写入 ObjectBox，
/// SourceFilterRetriever 也据此做 post-filter。
constexpr const char* kRouteTech = "tech";
constexpr const char* kRouteCuisine = "cuisine";

std::vector<std::string> splitCliInput(const std::string& input) {
    std::istringstream iss(input);
    std::vector<std::string> args;
    std::string token;
    while (iss >> token) args.push_back(token);
    return args;
}

std::string joinFrom(const std::vector<std::string>& args, std::size_t start) {
    std::string result;
    for (std::size_t i = start; i < args.size(); ++i) {
        if (!result.empty()) result += " ";
        result += args[i];
    }
    return result;
}

bool isPositiveInt(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
    }
    return true;
}

std::string readFileAll(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.good()) {
        throw std::runtime_error("File not found: " + path);
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

std::string preview(const std::string& text, std::size_t maxBytes = 60) {
    std::string s;
    s.reserve(text.size());
    for (char c : text) s += (c == '\n' ? ' ' : c);
    if (s.size() <= maxBytes) return s;
    std::size_t cut = maxBytes;
    while (cut > 0 && (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) --cut;
    return s.substr(0, cut) + "...";
}

/// 演示专用的"按 source 后过滤"检索器：
/// v0.3 用一个共享 store 装多域语料（source 字段区分域），Router 决策后
/// 把 query 转给某个域的子检索器时，用它做 post-filter。生产环境更合理的
/// 做法是把过滤下推到存储层（`IVectorStore::searchByVectorWithFilter`，
/// 见 rag_cpp_plan.md 的 Phase E 演进计划），本 demo 不做这一步以保持简单。
///
/// 由于是后过滤，会自动把底层 topK overfetch 到 `overfetchMul × topK`，
/// 尽量保证过滤后仍有足够候选。
class SourceFilterRetriever : public rag::IRetriever {
public:
    SourceFilterRetriever(std::shared_ptr<rag::IRetriever> base, std::string allowedSource,
                          int overfetchMul = 5)
        : base_(std::move(base)),
          allowedSource_(std::move(allowedSource)),
          overfetchMul_(overfetchMul) {}

    std::vector<rag::RetrievedChunk> retrieve(const std::string& query, int topK) override {
        if (topK <= 0) return {};
        const int overfetch = topK * std::max(1, overfetchMul_);
        auto raw = base_->retrieve(query, overfetch);
        std::vector<rag::RetrievedChunk> out;
        out.reserve(std::min<std::size_t>(raw.size(), topK));
        for (auto& c : raw) {
            if (c.source == allowedSource_) {
                out.push_back(std::move(c));
                if (static_cast<int>(out.size()) >= topK) break;
            }
        }
        return out;
    }

private:
    std::shared_ptr<rag::IRetriever> base_;
    std::string allowedSource_;
    int overfetchMul_;
};

void printHelp() {
    std::cout
        << "Available commands:\n"
        << "    ingest <path> <source>   Load a file with a domain tag (e.g. tech / cuisine)\n"
        << "    route <question>         Show LLM router decision {route, reason} only\n"
        << "    ask <question> [K]       Retrieve top-K with current strategy, then generate\n"
        << "    compare <question> [K]   Run all 4 strategies and print retrieval diff\n"
        << "    strategy [name]          Show or switch strategy (plain|route|rerank|full)\n"
        << "    ls                       Show number of stored documents\n"
        << "    clear                    Remove all stored documents\n"
        << "    help                     Print this help\n"
        << "    exit                     Quit\n";
}

/// 单个策略的一次召回结果打印。`plainIds` 非空时对不在其中的 id 前加 '*'（相对 plain 的新增）。
void printHits(const std::vector<rag::RetrievedChunk>& hits,
               const std::unordered_set<std::uint64_t>* plainIds = nullptr) {
    if (hits.empty()) {
        std::cout << "    (no hits)\n";
        return;
    }
    for (std::size_t i = 0; i < hits.size(); ++i) {
        const auto& c = hits[i];
        const bool isNew = plainIds && plainIds->count(c.id) == 0;
        std::printf("    %c#%zu  id=%-4" PRIu64 " src=%-8s score=%8.4f  %s\n",
                    isNew ? '*' : ' ', i + 1, c.id, c.source.c_str(), c.score,
                    preview(c.text).c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << "** autumndawn.cpp · rag_route (v0.3) **" << std::endl;

    std::string configPath = "rag_config.json";
    std::string strategy = "full";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--strategy=", 0) == 0) {
            strategy = arg.substr(std::string("--strategy=").size());
        } else {
            configPath = arg;
        }
    }
    {
        bool known = false;
        for (const char* s : kStrategies) known = known || strategy == s;
        if (!known) {
            LOG(ERROR) << "Unknown strategy: " << strategy
                       << " (expected plain|route|rerank|full)";
            return 1;
        }
    }

    rag::RagConfig cfg;
    try {
        cfg = rag::loadRagConfig(configPath);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to load config from " << configPath << ": " << e.what();
        LOG(ERROR) << "Hint: cp rag_config.example.json rag_config.json and fill in your API keys.";
        return 1;
    }

    if (!cfg.model.rerank.has_value()) {
        LOG(ERROR) << "rag_route requires a 'model.rerank' section (SiliconFlow bge-reranker-v2-m3).";
        LOG(ERROR) << "Fill in the rerank section in rag_config.json and try again.";
        return 1;
    }

    std::shared_ptr<rag::inference::IEmbeddingModel> embModel;
    std::shared_ptr<rag::inference::IChatModel> chatModel;
    std::shared_ptr<rag::inference::IRerankModel> rerankModel;
    std::shared_ptr<rag::storage::IVectorStore> store;
    try {
        embModel = rag::createEmbeddingModel(cfg.model.embedding, rag::kEmbeddingDim);
        chatModel = rag::createChatModel(cfg.model.chat);
        rerankModel = rag::createRerankModel(*cfg.model.rerank);
        store = rag::createVectorStore(cfg.storage, rag::kEmbeddingDim);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Startup failed: " << e.what();
        return 1;
    }

    // 基线全库检索器（不区分域）。
    auto plainRetriever = std::make_shared<rag::VectorStoreRetriever>(embModel, store);

    // 每个 route 对应一个 SourceFilterRetriever，只召回该域的 chunk。
    // 新增域只需在这里、rag::RouteChoice 列表、ingest 用的 source 三处保持同名即可。
    auto techRetriever =
        std::make_shared<SourceFilterRetriever>(plainRetriever, kRouteTech);
    auto cuisineRetriever =
        std::make_shared<SourceFilterRetriever>(plainRetriever, kRouteCuisine);

    // LlmRouter + RoutedRetriever：给上层一个"看起来还是 IRetriever"的门面。
    std::vector<rag::RouteChoice> routes = {
        {kRouteTech,    "关于计算机、编程、数据库、AI、RAG、向量检索等技术类问题"},
        {kRouteCuisine, "关于中餐做法、食材、烹饪技巧、菜谱等饮食类问题"},
    };
    auto router = std::make_shared<rag::LlmRouter>(chatModel, routes);
    std::unordered_map<std::string, std::shared_ptr<rag::IRetriever>> routeMap = {
        {kRouteTech,    techRetriever},
        {kRouteCuisine, cuisineRetriever},
    };
    auto routedRetriever = std::make_shared<rag::RoutedRetriever>(router, routeMap, kRouteTech);

    // Re-ranking 装饰器：粗召回 20 → 精排 topK。
    auto rerankPlain = std::make_shared<rag::RerankingRetriever>(plainRetriever, rerankModel);
    auto rerankFull = std::make_shared<rag::RerankingRetriever>(routedRetriever, rerankModel);

    const std::map<std::string, std::shared_ptr<rag::IRetriever>> retrievers = {
        {"plain",  plainRetriever},
        {"route",  routedRetriever},
        {"rerank", rerankPlain},
        {"full",   rerankFull},
    };

    auto generator = std::make_shared<rag::ChatGenerator>(chatModel);
    rag::RecursiveCharacterTextSplitter splitter;

    // 策略私有中间产物（选路决策 / 候选条数）。
    auto printStrategyDetails = [&](const std::string& name) {
        if (name == "route") {
            const auto& d = routedRetriever->lastDecision();
            std::cout << "    router chose: " << d.route << "  reason: " << d.reason << "\n";
        } else if (name == "rerank") {
            std::cout << "    reranked from " << rerankPlain->lastCandidateCount()
                      << " candidates\n";
        } else if (name == "full") {
            const auto& d = routedRetriever->lastDecision();
            std::cout << "    router chose: " << d.route << "  reason: " << d.reason << "\n";
            std::cout << "    reranked from " << rerankFull->lastCandidateCount()
                      << " candidates\n";
        }
    };

    std::cout << "Ready. Strategy: " << strategy
              << ". Registered routes: " << kRouteTech << " / " << kRouteCuisine << "."
              << " Type 'help' for commands." << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        auto args = splitCliInput(line);
        if (args.empty()) continue;
        const std::string& cmd = args[0];

        try {
            if (cmd == "exit" || cmd == "quit") {
                break;
            } else if (cmd == "help" || cmd == "?") {
                printHelp();
            } else if (cmd == "ls") {
                std::cout << "Stored documents: " << store->count() << std::endl;
            } else if (cmd == "clear") {
                auto removed = store->clearAll();
                std::cout << "Removed " << removed << " documents." << std::endl;
            } else if (cmd == "strategy") {
                if (args.size() < 2) {
                    std::cout << "Current strategy: " << strategy << std::endl;
                } else if (retrievers.count(args[1])) {
                    strategy = args[1];
                    std::cout << "Switched to: " << strategy << std::endl;
                } else {
                    LOG(WARNING) << "Unknown strategy: " << args[1]
                                 << " (expected plain|route|rerank|full)";
                }
            } else if (cmd == "ingest") {
                if (args.size() < 3) {
                    LOG(WARNING) << "Usage: ingest <path> <source>";
                    continue;
                }
                const std::string& path = args[1];
                const std::string& source = args[2];
                std::string raw = readFileAll(path);
                auto chunks = splitter.split(raw);
                if (chunks.empty()) {
                    std::cout << "No content found in " << path << std::endl;
                    continue;
                }
                std::cout << "Embedding " << chunks.size() << " chunk(s) from " << path
                          << " as source='" << source << "' ..." << std::endl;
                plainRetriever->ingest(chunks, source);
                std::cout << "Done. Total docs now: " << store->count() << std::endl;
            } else if (cmd == "route") {
                if (args.size() < 2) {
                    LOG(WARNING) << "Usage: route <question>";
                    continue;
                }
                const std::string question = joinFrom(args, 1);
                const auto d = router->route(question);
                std::cout << "router chose: " << d.route << "\n"
                          << "reason      : " << d.reason << std::endl;
            } else if (cmd == "ask" || cmd == "compare") {
                if (args.size() < 2) {
                    LOG(WARNING) << "Usage: " << cmd << " <question> [topK]";
                    continue;
                }
                int topK = 5;
                std::string question;
                if (args.size() >= 3 && isPositiveInt(args.back())) {
                    topK = std::stoi(args.back());
                    question = joinFrom(args, 1);
                    auto pos = question.rfind(' ');
                    if (pos != std::string::npos) question = question.substr(0, pos);
                } else {
                    question = joinFrom(args, 1);
                }

                if (cmd == "ask") {
                    rag::RagPipeline pipeline(retrievers.at(strategy), generator);
                    auto result = pipeline.ask(question, topK);

                    std::cout << "\n---- [" << strategy << "] retrieved contexts (top "
                              << result.contexts.size() << ") ----\n";
                    printHits(result.contexts);
                    printStrategyDetails(strategy);
                    std::cout << "\n---- answer ----\n" << result.answer << "\n\n";
                } else {
                    // compare：四种策略只做检索，不生成，打印召回差异。
                    std::unordered_set<std::uint64_t> plainIds;
                    for (const char* name : kStrategies) {
                        auto hits = retrievers.at(name)->retrieve(question, topK);
                        if (std::string(name) == "plain") {
                            for (const auto& h : hits) plainIds.insert(h.id);
                            std::cout << "\n==== [" << name << "] top " << hits.size()
                                      << " ====\n";
                            printHits(hits);
                        } else {
                            std::cout << "\n==== [" << name << "] top " << hits.size()
                                      << "  (* = not recalled by plain) ====\n";
                            printHits(hits, &plainIds);
                        }
                        printStrategyDetails(name);
                    }
                    std::cout << "\n";
                }
            } else {
                LOG(WARNING) << "Unknown command: " << cmd << ". Try 'help'.";
            }
        } catch (const std::exception& e) {
            LOG(ERROR) << "Error while executing \"" << line << "\": " << e.what();
        }
    }

    return 0;
}
