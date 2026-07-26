#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace autumndawn::rag::inference {

/// rerank 结果：输入 docs 中某个片段与 query 的相关性。
struct RerankHit {
    std::size_t index = 0;  ///< 在输入 docs 中的下标
    float score = 0.0f;     ///< 相关性分数，越高越相关
};

/// (query, docs) → 相关性排序 的抽象接口。
/// 部署形态无关：云端 API（SiliconFlow bge-reranker 等）与端侧进程内推理（onnx，规划中）
/// 均实现本接口，上层不感知差异。具体实现类不暴露在公共头文件中，统一经
/// rag/model_factory.hpp 的工厂函数按配置创建。
///
/// 线程安全契约：实现不保证线程安全，调用方需串行访问。
class IRerankModel {
public:
    virtual ~IRerankModel() = default;

    /// 对 docs 按与 query 的相关性打分，返回按分数降序排列的结果。
    /// topN > 0 时只返回前 topN 条（超过 docs 数量按 docs 数量计）；
    /// topN <= 0 返回全部（仍按分数降序）。
    virtual std::vector<RerankHit> rerank(const std::string& query,
                                          const std::vector<std::string>& docs,
                                          int topN) = 0;
};

}  // namespace autumndawn::rag::inference
