#pragma once

#include <stdexcept>
#include <string>

namespace autumndawn::rag::storage {

/// 存储层通用错误：打开数据库失败、后端不支持的能力（如向量检索）、
/// 写入/查询过程中的运行期错误等，统一以此异常上抛。
class StorageError : public std::runtime_error {
public:
    explicit StorageError(const std::string& what) : std::runtime_error(what) {}
};

}  // namespace autumndawn::rag::storage
