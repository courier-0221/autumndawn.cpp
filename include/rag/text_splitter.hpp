#pragma once

#include <string>
#include <vector>

namespace autumndawn::rag {

/// 递归字符文本切分器。
/// 按优先级尝试用一组分隔符去切文本（如："\n\n" → "\n" → "。" → "，" → " " → ""），
/// 保证每段 ≤ chunkSize 字节，相邻两段有 chunkOverlap 字节重叠以保留上下文。
/// 参考：LangChain RecursiveCharacterTextSplitter，v0.1 用最直白的字节实现（对中文来说是按字节数）。
class RecursiveCharacterTextSplitter {
public:
    struct Options {
        std::size_t chunkSize = 1200;      ///< 每块最大字节数（中文 ≈ 400 字，≈ 500 token）
        std::size_t chunkOverlap = 120;    ///< 相邻块的重叠字节数
        std::vector<std::string> separators = {"\n\n", "\n", "。", "！", "？", "，", " ", ""};
    };

    RecursiveCharacterTextSplitter() : RecursiveCharacterTextSplitter(Options{}) {}
    explicit RecursiveCharacterTextSplitter(Options options);

    std::vector<std::string> split(const std::string& text) const;

private:
    Options options_;
};

}  // namespace autumndawn::rag
