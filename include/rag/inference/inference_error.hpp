#pragma once

#include <stdexcept>
#include <string>

namespace autumndawn::rag::inference {

/// Embedding / Chat / Rerank 相关的运行时错误。
class InferenceError : public std::runtime_error {
public:
    explicit InferenceError(const std::string& what) : std::runtime_error(what) {}
};

}  // namespace autumndawn::rag::inference
