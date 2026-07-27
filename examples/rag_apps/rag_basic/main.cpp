// rag_basic —— v0.1 端到端最小 RAG 演示。
// 命令行交互：
//   ingest <path>       读入文本文件、切块、embedding、写 ObjectBox
//   ask    <question>   embedding query → HNSW 检索 topK → DeepSeek 生成答案
//   ls                  打印当前库里的文档数
//   clear               清空所有文档
//   help / exit         帮助 / 退出

#define OBX_CPP_FILE  // 让 objectbox.hpp 中的实现在本 TU materialize

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "objectbox.hpp"

#include "rag/config.hpp"
#include "rag/generator.hpp"
#include "rag/log.hpp"
#include "rag/model_factory.hpp"
#include "rag/objectbox_retriever.hpp"
#include "rag/pipeline.hpp"
#include "rag/text_splitter.hpp"

namespace rag = autumndawn::rag;

namespace {

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

void printHelp() {
    std::cout << "Available commands:\n"
              << "    ingest <path>          Load a text file, split into chunks, embed and store\n"
              << "    ask <question> [K]     Retrieve top-K contexts and ask DeepSeek\n"
              << "    ls                     Show number of stored documents\n"
              << "    clear                  Remove all stored documents\n"
              << "    help                   Print this help\n"
              << "    exit                   Quit\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << "** autumndawn.cpp · rag_basic (v0.1) **" << std::endl;

    if (!obx_has_feature(OBXFeature_VectorSearch)) {
        LOG(ERROR) << "This ObjectBox build has no vector search support.";
        return 1;
    }

    const std::string configPath = argc >= 2 ? argv[1] : "rag_config.json";

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
    try {
        embModel = rag::createEmbeddingModel(cfg.model.embedding, rag::kEmbeddingDim);
        chatModel = rag::createChatModel(cfg.model.chat);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to create inference models: " << e.what();
        return 1;
    }

    obx::Options options(rag::createRagModel());
    options.directory("objectbox-db");
    obx::Store store(options);

    auto retriever = std::make_shared<rag::ObjectBoxRetriever>(store, embModel);
    auto generator = std::make_shared<rag::ChatGenerator>(chatModel);
    rag::RagPipeline pipeline(retriever, generator);

    rag::RecursiveCharacterTextSplitter splitter;

    std::cout << "Ready. Type 'help' for commands, 'exit' to quit." << std::endl;

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
                std::cout << "Stored documents: " << retriever->count() << std::endl;
            } else if (cmd == "clear") {
                auto removed = retriever->clearAll();
                std::cout << "Removed " << removed << " documents." << std::endl;
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
                retriever->ingest(chunks, path);
                std::cout << "Done. Total docs now: " << retriever->count() << std::endl;
            } else if (cmd == "ask") {
                if (args.size() < 2) {
                    LOG(WARNING) << "Usage: ask <question> [topK]";
                    continue;
                }
                int topK = 5;
                std::string question;
                if (args.size() >= 3 && isPositiveInt(args.back())) {
                    topK = std::stoi(args.back());
                    question = joinFrom(args, 1);
                    // 剔除末尾的 topK token
                    auto pos = question.rfind(' ');
                    if (pos != std::string::npos) question = question.substr(0, pos);
                } else {
                    question = joinFrom(args, 1);
                }

                auto result = pipeline.ask(question, topK);

                std::cout << "\n---- retrieved contexts (top " << result.contexts.size()
                          << ") ----\n";
                for (std::size_t i = 0; i < result.contexts.size(); ++i) {
                    const auto& c = result.contexts[i];
                    std::printf("#%zu  id=%" PRIu64 "  src=%-16s  score=%.4f\n    %s\n",
                                i + 1, c.id, c.source.c_str(), c.score, c.text.c_str());
                }
                std::cout << "\n---- answer ----\n" << result.answer << "\n\n";
            } else {
                LOG(WARNING) << "Unknown command: " << cmd << ". Try 'help'.";
            }
        } catch (const std::exception& e) {
            LOG(ERROR) << "Error while executing \"" << line << "\": " << e.what();
        }
    }

    return 0;
}
