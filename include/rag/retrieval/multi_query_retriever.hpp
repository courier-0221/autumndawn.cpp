#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rag/inference/chat_model.hpp"
#include "rag/retriever.hpp"

namespace autumndawn::rag {

/// Multi-Query 检索装饰器（课程小节 5 / 6）。
///
/// 用 `IChatModel` 把原始问题改写成 N 条语义相同、表述不同的查询，
/// 各自经底层 `IRetriever` 检索一遍，再把多路结果合并：
///   - `Union`：按名次轮转交错去重（近似课程里的 get_unique_union）；
///   - `Rrf`  ：Reciprocal Rank Fusion（即 RAG-Fusion，小节 6）。
///
/// 只依赖 `IRetriever` / `IChatModel` 抽象，可装饰任意底层检索器。
class MultiQueryRetriever : public IRetriever {
public:
    enum class MergeStrategy {
        Union,  ///< 名次轮转去重合并
        Rrf,    ///< Reciprocal Rank Fusion（RAG-Fusion）
    };

    struct Options {
        /// LLM 生成的改写查询条数（不含原始问题本身）。
        int numQueries = 3;
        /// 是否把原始问题也作为一路参与检索。
        bool includeOriginal = true;
        MergeStrategy merge = MergeStrategy::Union;
        /// RRF 平滑常数（仅 merge == Rrf 时生效）。
        int rrfK = 60;
        /// 改写 prompt，占位符：{question} / {n}。
        std::string promptTemplate =
            "你是一个帮助改写检索查询的助手。请针对下面的问题，生成 {n} 个语义相同"
            "但表述角度不同的中文检索查询，帮助向量检索系统召回更多相关文档。\n"
            "要求：每行输出一个查询，不要编号，不要输出任何多余内容。\n\n"
            "原始问题：{question}";
        inference::ChatOptions chatOpt;
    };

    MultiQueryRetriever(std::shared_ptr<IRetriever> base,
                        std::shared_ptr<inference::IChatModel> chat)
        : MultiQueryRetriever(std::move(base), std::move(chat), Options{}) {}
    MultiQueryRetriever(std::shared_ptr<IRetriever> base,
                        std::shared_ptr<inference::IChatModel> chat, Options options);

    std::vector<RetrievedChunk> retrieve(const std::string& query, int topK) override;

    /// 最近一次 retrieve 实际使用的全部查询（含原始问题，若启用），
    /// 供演示程序打印"改写了什么"。
    const std::vector<std::string>& lastQueries() const { return lastQueries_; }

private:
    /// 调 LLM 生成改写查询；解析按行拆分并清洗编号前缀。
    std::vector<std::string> generateQueries(const std::string& query);

    std::shared_ptr<IRetriever> base_;
    std::shared_ptr<inference::IChatModel> chat_;
    Options options_;
    std::vector<std::string> lastQueries_;
};

}  // namespace autumndawn::rag
