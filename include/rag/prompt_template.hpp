#pragma once

#include <string>
#include <unordered_map>

namespace autumndawn::rag {

/// 极简 Prompt 模板：把 "{key}" 占位符替换成给定值。
/// 例如 format("你好 {name}", {{"name", "世界"}}) => "你好 世界"
/// - 未提供的 key 保留原样；
/// - 不做转义 / 不支持 {{ }} 等复杂语法（v0.1 够用）。
namespace prompt {

std::string format(const std::string& tmpl,
                   const std::unordered_map<std::string, std::string>& vars);

}  // namespace prompt

}  // namespace autumndawn::rag
