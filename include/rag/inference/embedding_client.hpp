#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace autumndawn::rag::inference {

/// Embedding / Chat / Rerank 相关的运行时错误。
class InferenceError : public std::runtime_error {
public:
    explicit InferenceError(const std::string& what) : std::runtime_error(what) {}
};

/// 把文本转成固定维度浮点向量的抽象接口。
/// 当前实现走云端 API（SiliconFlow）；未来可加本地推理（llama.cpp / onnx）实现同一接口。
class IEmbeddingClient {
public:
    virtual ~IEmbeddingClient() = default;

    /// 单条文本 → 向量
    virtual std::vector<float> embed(const std::string& text) = 0;

    /// 批量文本 → 向量列表（一次调用尽量少拆包）
    virtual std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts) = 0;

    /// 输出向量维度（用于运行时校验 ObjectBox schema 是否匹配）。
    virtual std::size_t dimension() const = 0;
};

/// 硅基流动（SiliconFlow）Embedding API 客户端。
/// OpenAI 兼容协议：POST {baseUrl}/embeddings
class SiliconFlowEmbeddingClient : public IEmbeddingClient {
public:
    /// dim: 期望的向量维度（bge-m3 = 1024）。0 表示不校验。
    SiliconFlowEmbeddingClient(std::string baseUrl,
                               std::string apiKey,
                               std::string model,
                               std::size_t dim = 1024);
    ~SiliconFlowEmbeddingClient() override;

    std::vector<float> embed(const std::string& text) override;
    std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts) override;
    std::size_t dimension() const override { return dim_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::size_t dim_;
};

}  // namespace autumndawn::rag::inference
