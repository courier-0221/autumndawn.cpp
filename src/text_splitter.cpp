#include "rag/text_splitter.hpp"

#include <algorithm>

namespace autumndawn::rag {

namespace {

/// 按给定分隔符把 text 切成"尽量大的块"（不去空块）。
/// 分隔符本身归到左侧那块的末尾，尽量保持文本原样。
std::vector<std::string> splitBy(const std::string& text, const std::string& sep) {
    std::vector<std::string> parts;
    if (sep.empty()) {
        // 空分隔符：按单字节切（作为最后兜底）。中文单字节切会切开 utf-8，但是最后兜底
        // 时块已经足够小，实际很少走到这个分支。
        for (char c : text) parts.emplace_back(1, c);
        return parts;
    }

    std::size_t start = 0;
    while (start < text.size()) {
        auto pos = text.find(sep, start);
        if (pos == std::string::npos) {
            parts.push_back(text.substr(start));
            break;
        }
        parts.push_back(text.substr(start, pos - start + sep.size()));
        start = pos + sep.size();
    }
    return parts;
}

/// UTF-8 续字节（continuation byte）形如 10xxxxxx。从 `pos` 开始往后跳过续字节，
/// 保证返回的位置落在字符边界上，避免把多字节字符（如中文）从中间切开。
std::size_t utf8SafeStart(const std::string& s, std::size_t pos) {
    while (pos < s.size() && (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80) {
        ++pos;
    }
    return pos;
}

/// 把"许多小片段"贪心合并成不超过 chunkSize 的块，相邻块保留 chunkOverlap 字节重叠。
std::vector<std::string> mergePieces(const std::vector<std::string>& pieces,
                                     std::size_t chunkSize, std::size_t chunkOverlap) {
    std::vector<std::string> chunks;
    std::string current;

    for (const auto& p : pieces) {
        if (current.empty()) {
            current = p;
            continue;
        }
        if (current.size() + p.size() <= chunkSize) {
            current += p;
        } else {
            chunks.push_back(current);
            // 生成 overlap 前缀（取上一块的末尾 chunkOverlap 字节，按 UTF-8 字符边界对齐，
            // 避免切开多字节字符导致后续 JSON 序列化时报 invalid UTF-8）。
            if (chunkOverlap > 0 && chunkOverlap < current.size()) {
                std::size_t start = utf8SafeStart(current, current.size() - chunkOverlap);
                current = current.substr(start) + p;
            } else {
                current = p;
            }
        }
    }
    if (!current.empty()) chunks.push_back(current);
    return chunks;
}

}  // namespace

RecursiveCharacterTextSplitter::RecursiveCharacterTextSplitter(Options options)
    : options_(std::move(options)) {
    if (options_.chunkSize == 0) options_.chunkSize = 1;
    if (options_.chunkOverlap >= options_.chunkSize) {
        options_.chunkOverlap = options_.chunkSize / 4;
    }
    if (options_.separators.empty()) options_.separators = {""};
}

std::vector<std::string> RecursiveCharacterTextSplitter::split(const std::string& text) const {
    if (text.empty()) return {};
    if (text.size() <= options_.chunkSize) return {text};

    // 递归尝试每个分隔符：越靠前的越"粗粒度"，越希望被优先使用。
    // 对每个片段：若已经 <= chunkSize 则保留，否则用下一级分隔符再切。
    std::vector<std::string> current = {text};
    for (const auto& sep : options_.separators) {
        std::vector<std::string> next;
        bool needMoreSplit = false;
        for (const auto& piece : current) {
            if (piece.size() <= options_.chunkSize) {
                next.push_back(piece);
                continue;
            }
            auto sub = splitBy(piece, sep);
            if (sub.size() <= 1) {
                // 这个分隔符没起作用，piece 原样传给下一级。
                next.push_back(piece);
                needMoreSplit = true;
                continue;
            }
            for (auto& s : sub) next.push_back(std::move(s));
            // 有的 sub 可能还 > chunkSize，标记一下下一轮继续切。
            for (const auto& s : sub) {
                if (s.size() > options_.chunkSize) {
                    needMoreSplit = true;
                    break;
                }
            }
        }
        current = std::move(next);
        if (!needMoreSplit) break;
    }

    // 现在 current 里每个片段基本 <= chunkSize（或已用尽分隔符），做贪心合并 + overlap。
    return mergePieces(current, options_.chunkSize, options_.chunkOverlap);
}

}  // namespace autumndawn::rag
