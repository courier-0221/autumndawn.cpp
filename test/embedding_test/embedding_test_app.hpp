#pragma once

#include <cinttypes>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "document.obx.hpp"
#include "objectbox-model.h"
#include "objectbox.hpp"
#include "siliconflow_client.hpp"

/// 交互式命令行应用：演示"文本 -> Embedding -> ObjectBox 存取/检索"的完整链路。
/// 命令风格参考 objectbox-c 官方 vectorsearch-cities 例子。
class EmbeddingTestApp {
    enum class Command { Import, Add, Search, List, RemoveAll, Exit, Help, Unknown };

    obx::Store& store;
    obx::Box<Document> box;
    SiliconFlowEmbeddingClient& client;

    /// 向量检索 query，成员变量复用，避免每次都重建 QueryBuilder。
    obx::Query<Document> queryByVector_;

public:
    EmbeddingTestApp(obx::Store& obxStore, SiliconFlowEmbeddingClient& embeddingClient)
        : store(obxStore),
          box(obxStore),
          client(embeddingClient),
          queryByVector_(box.query(Document_::embedding.nearestNeighbors({}, 1)).build()) {}

    int run() {
        std::cout << "Welcome to the ObjectBox + SiliconFlow embedding demo" << std::endl;
        printHelp();

        std::string input;
        std::vector<std::string> args;
        while (std::getline(std::cin, input)) {  // ctrl-d 退出
            if (input.empty()) continue;

            splitInput(input, args);
            try {
                Command command = getCommand(args[0]);
                if (command == Command::Exit) return 0;
                processCommand(command, args);
            } catch (const std::exception& e) {
                std::cerr << "Error executing \"" << input << "\": " << e.what() << std::endl;
            }
        }
        return 0;
    }

protected:
    void processCommand(Command command, const std::vector<std::string>& args) {
        switch (command) {
            case Command::Import: {
                if (args.size() == 2) {
                    importFile(args[1]);
                } else {
                    std::cerr << "Usage: import <path>" << std::endl;
                }
                break;
            }
            case Command::Add: {
                if (args.size() >= 2) {
                    std::string text = joinFrom(args, 1);
                    addDocument(text, "manual");
                } else {
                    std::cerr << "Usage: add <text>" << std::endl;
                }
                break;
            }
            case Command::Search: {
                if (args.size() >= 2) {
                    int64_t topK = 5;
                    std::string queryText = joinFrom(args, 1);
                    // 若最后一个 token 是数字，把它当作 topK，剩下的作为查询文本。
                    if (args.size() >= 3 && isNumber(args.back())) {
                        topK = std::atoll(args.back().c_str());
                        queryText = joinFrom(args, 1, args.size() - 1);
                    }
                    search(queryText, topK);
                } else {
                    std::cerr << "Usage: search <query text> [topK]" << std::endl;
                }
                break;
            }
            case Command::List: {
                dump(box.getAll());
                break;
            }
            case Command::RemoveAll: {
                uint64_t removed = box.removeAll();
                std::cout << "removeAll removed " << removed << " documents" << std::endl;
                break;
            }
            case Command::Help:
                printHelp();
                break;
            case Command::Exit:
                assert(false);
                break;
            case Command::Unknown:
            default:
                std::cerr << "Unknown command: " << args.at(0) << std::endl;
                printHelp();
                break;
        }
    }

    /// 读取文本文件，按空行切分为段落 chunk，批量 embedding 后写入。
    void importFile(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs.good()) {
            std::cerr << "File not found: " << path << std::endl;
            return;
        }

        std::vector<std::string> chunks;
        std::string line, buffer;
        while (std::getline(ifs, line)) {
            if (line.empty()) {
                if (!buffer.empty()) {
                    chunks.push_back(buffer);
                    buffer.clear();
                }
            } else {
                if (!buffer.empty()) buffer += " ";
                buffer += line;
            }
        }
        if (!buffer.empty()) chunks.push_back(buffer);

        if (chunks.empty()) {
            std::cout << "No paragraphs found in " << path << std::endl;
            return;
        }

        std::cout << "Embedding " << chunks.size() << " chunk(s) from " << path << " ..." << std::endl;
        std::vector<std::vector<float>> vectors = client.embedBatch(chunks);

        obx::Transaction tx = store.tx(obx::TxMode::WRITE);
        for (size_t i = 0; i < chunks.size(); i++) {
            Document doc;
            doc.id = 0;
            doc.text = chunks[i];
            doc.source = path;
            doc.embedding = vectors[i];
            box.put(doc);
        }
        tx.success();
        std::cout << "Imported " << chunks.size() << " document(s) from " << path << std::endl;
    }

    void addDocument(const std::string& text, const std::string& source) {
        Document doc;
        doc.id = 0;
        doc.text = text;
        doc.source = source;
        doc.embedding = client.embed(text);
        obx_id id = box.put(doc);
        std::cout << "Added document id=" << id << ": \"" << text << "\"" << std::endl;
    }

    void search(const std::string& queryText, int64_t topK) {
        std::vector<float> queryVector = client.embed(queryText);
        queryByVector_.setParameter(Document_::embedding, queryVector);
        queryByVector_.setParameterMaxNeighbors(Document_::embedding, topK);
        std::vector<std::pair<Document, double>> results = queryByVector_.findWithScores();
        dump(results);
    }

    static void splitInput(const std::string& input, std::vector<std::string>& outArgs) {
        outArgs.clear();
        std::istringstream iss(input);
        std::string token;
        while (iss >> token) {
            outArgs.push_back(token);
        }
        if (outArgs.empty()) outArgs.push_back(input);
    }

    static std::string joinFrom(const std::vector<std::string>& args, size_t start, size_t end = SIZE_MAX) {
        if (end == SIZE_MAX) end = args.size();
        std::string result;
        for (size_t i = start; i < end; i++) {
            if (!result.empty()) result += " ";
            result += args[i];
        }
        return result;
    }

    static bool isNumber(const std::string& s) {
        if (s.empty()) return false;
        for (char c : s) {
            if (!isdigit(static_cast<unsigned char>(c))) return false;
        }
        return true;
    }

    Command getCommand(const std::string& cmd) const {
        if (cmd == "import") return Command::Import;
        if (cmd == "add") return Command::Add;
        if (cmd == "search") return Command::Search;
        if (cmd == "ls" || cmd == "list") return Command::List;
        if (cmd == "removeAll") return Command::RemoveAll;
        if (cmd == "exit" || cmd == "quit") return Command::Exit;
        if (cmd == "help" || cmd == "?") return Command::Help;
        return Command::Unknown;
    }

    void printHelp() const {
        std::cout << "Available commands are:\n"
                  << "    import <path>              Import a text file (split by blank lines) and embed+store it\n"
                  << "    add <text>                 Embed a single piece of text and store it\n"
                  << "    search <query> [topK]      Embed the query and find topK nearest documents (default 5)\n"
                  << "    ls                         List all stored documents\n"
                  << "    removeAll                  Remove all stored documents\n"
                  << "    exit                       Close the program\n"
                  << "    help                       Display this help" << std::endl;
    }

    static void dump(const Document& doc) {
        printf("%3" PRIu64 "  [%-10s]  %s\n", doc.id, doc.source.c_str(), doc.text.c_str());
    }

    static void dump(const std::vector<std::unique_ptr<Document>>& list) {
        printf("%3s  %-12s  %s\n", "ID", "Source", "Text");
        for (const auto& doc : list) {
            dump(*doc);
        }
    }

    static void dump(const std::pair<Document, double>& pair) {
        printf("%3" PRIu64 "  [%-10s]  score=%-8.4f  %s\n", pair.first.id, pair.first.source.c_str(), pair.second,
               pair.first.text.c_str());
    }

    static void dump(const std::vector<std::pair<Document, double>>& list) {
        printf("%3s  %-12s  %-16s%s\n", "ID", "Source", "Score", "Text");
        for (const auto& pair : list) {
            dump(pair);
        }
    }
};
