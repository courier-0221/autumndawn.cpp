// rag_query_translate —— v0.2 (Phase B) Query Translation 策略对比演示。
//
// 四种检索策略（--strategy= 或运行中 strategy 命令切换）：
//   plain        直接用原始问题做向量检索（v0.1 基线）
//   multi_query  LLM 改写 N 条查询 → 并联检索 → 名次轮转去重合并（小节 5）
//   rrf          同 multi_query，但用 Reciprocal Rank Fusion 合并（小节 6，RAG-Fusion）
//   hyde         LLM 先"假答"出一段文档，再拿它去检索（小节 9，HyDE）
//
// 命令行交互：
//   ingest <path>          读入文本文件、切块、embedding、写向量库
//   ask <question> [K]     用当前策略检索 topK 并生成答案
//   compare <question> [K] 四种策略同题对比，打印召回差异
//   strategy [name]        查看 / 切换当前策略
//   ls / clear / help / exit

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "rag/config.hpp"
#include "rag/generator.hpp"
#include "rag/log.hpp"
#include "rag/model_factory.hpp"
#include "rag/pipeline.hpp"
#include "rag/retrieval/hyde_retriever.hpp"
#include "rag/retrieval/multi_query_retriever.hpp"
#include "rag/retrieval/vector_store_retriever.hpp"
#include "rag/store_factory.hpp"
#include "rag/text_splitter.hpp"

namespace rag = autumndawn::rag;

namespace {

constexpr const char* kStrategies[] = {"plain", "multi_query", "rrf", "hyde"};

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

/// 截断文本做单行预览（按字节截，避免撕裂 UTF-8 尾字节继续回退）。
std::string preview(const std::string& text, std::size_t maxBytes = 60) {
    std::string s;
    s.reserve(text.size());
    for (char c : text) s += (c == '\n' ? ' ' : c);
    if (s.size() <= maxBytes) return s;
    std::size_t cut = maxBytes;
    while (cut > 0 && (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) --cut;
    return s.substr(0, cut) + "...";
}

void printHelp() {
    std::cout << "Available commands:\n"
              << "    ingest <path>            Load a text file, split into chunks, embed and store\n"
              << "    ask <question> [K]       Retrieve top-K with current strategy, then generate\n"
              << "    compare <question> [K]   Run all 4 strategies and print retrieval diff\n"
              << "    strategy [name]          Show or switch strategy (plain|multi_query|rrf|hyde)\n"
              << "    ls                       Show number of stored documents\n"
              << "    clear                    Remove all stored documents\n"
              << "    help                     Print this help\n"
              << "    exit                     Quit\n";
}

/// 单个策略的一次召回结果打印。newIds 非空时对不在其中的 id 前加 '*'（相对 plain 的新增）。
void printHits(const std::vector<rag::RetrievedChunk>& hits,
               const std::unordered_set<std::uint64_t>* plainIds = nullptr) {
    if (hits.empty()) {
        std::cout << "    (no hits)\n";
        return;
    }
    for (std::size_t i = 0; i < hits.size(); ++i) {
        const auto& c = hits[i];
        const bool isNew = plainIds && plainIds->count(c.id) == 0;
        std::printf("    %c#%zu  id=%-4" PRIu64 " score=%8.4f  %s\n",
                    isNew ? '*' : ' ', i + 1, c.id, c.score, preview(c.text).c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << "** autumndawn.cpp · rag_query_translate (v0.2) **" << std::endl;

    std::string configPath = "rag_config.json";
    std::string strategy = "plain";
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
                       << " (expected plain|multi_query|rrf|hyde)";
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

    std::shared_ptr<rag::inference::IEmbeddingModel> embModel;
    std::shared_ptr<rag::inference::IChatModel> chatModel;
    std::shared_ptr<rag::storage::IVectorStore> store;
    try {
        embModel = rag::createEmbeddingModel(cfg.model.embedding, rag::kEmbeddingDim);
        chatModel = rag::createChatModel(cfg.model.chat);
        store = rag::createVectorStore(cfg.storage, rag::kEmbeddingDim);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Startup failed: " << e.what();
        return 1;
    }

    // 基线检索器 + 三个装饰器，全部共享同一套 embedding/store/chat。
    auto plainRetriever = std::make_shared<rag::VectorStoreRetriever>(embModel, store);

    auto multiQueryRetriever =
        std::make_shared<rag::MultiQueryRetriever>(plainRetriever, chatModel);

    rag::MultiQueryRetriever::Options rrfOpts;
    rrfOpts.merge = rag::MultiQueryRetriever::MergeStrategy::Rrf;
    auto rrfRetriever =
        std::make_shared<rag::MultiQueryRetriever>(plainRetriever, chatModel, rrfOpts);

    auto hydeRetriever = std::make_shared<rag::HydeRetriever>(plainRetriever, chatModel);

    const std::map<std::string, std::shared_ptr<rag::IRetriever>> retrievers = {
        {"plain", plainRetriever},
        {"multi_query", multiQueryRetriever},
        {"rrf", rrfRetriever},
        {"hyde", hydeRetriever},
    };

    auto generator = std::make_shared<rag::ChatGenerator>(chatModel);
    rag::RecursiveCharacterTextSplitter splitter;

    // 打印策略私有的中间产物（改写出的查询 / 假设性文档）。
    auto printStrategyDetails = [&](const std::string& name) {
        if (name == "multi_query" || name == "rrf") {
            const auto& mq = name == "rrf" ? rrfRetriever : multiQueryRetriever;
            std::cout << "    queries used:\n";
            for (const auto& q : mq->lastQueries()) {
                std::cout << "      - " << q << "\n";
            }
        } else if (name == "hyde") {
            std::cout << "    hypothetical doc: "
                      << preview(hydeRetriever->lastHypotheticalDoc(), 120) << "\n";
        }
    };

    std::cout << "Ready. Strategy: " << strategy
              << ". Type 'help' for commands, 'exit' to quit." << std::endl;

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
                                 << " (expected plain|multi_query|rrf|hyde)";
                }
            } else if (cmd == "ingest") {
                if (args.size() < 2) {
                    LOG(WARNING) << "Usage: ingest <path>";
                    continue;
                }
                const std::string& path = args[1];
                std::string raw = readFileAll(path);
                auto chunks = splitter.split(raw);
                if (chunks.empty()) {
                    std::cout << "No content found in " << path << std::endl;
                    continue;
                }
                std::cout << "Embedding " << chunks.size() << " chunk(s) from " << path
                          << " ..." << std::endl;
                plainRetriever->ingest(chunks, path);
                std::cout << "Done. Total docs now: " << store->count() << std::endl;
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
