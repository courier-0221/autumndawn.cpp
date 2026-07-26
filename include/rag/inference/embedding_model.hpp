#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace autumndawn::rag::inference {

/// 把文本转成固定维度浮点向量的抽象接口。
/// 部署形态无关：云端 API（SiliconFlow 等）与端侧进程内推理（llama.cpp / onnx，规划中）
/// 均实现本接口，上层不感知差异。具体实现类不暴露在公共头文件中，统一经
/// rag/model_factory.hpp 的工厂函数按配置创建。
///
/// 线程安全契约：实现不保证线程安全，调用方需串行访问。
/// （云端 HTTP 实现天然无所谓；端侧推理 session 大多不能并发调用，提前约定。）
class IEmbeddingModel {
public:
    virtual ~IEmbeddingModel() = default;

    /// 单条文本 → 向量
    virtual std::vector<float> embed(const std::string& text) = 0;

    /// 批量文本 → 向量列表（一次调用尽量少拆包）
    virtual std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts) = 0;

    /// 输出向量维度（用于运行时校验 ObjectBox schema 是否匹配）。
    virtual std::size_t dimension() const = 0;
};

}  // namespace autumndawn::rag::inference
