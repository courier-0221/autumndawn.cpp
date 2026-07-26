#pragma once

// 云端实现类的内部头文件：不暴露到 include/，外部统一经 rag/model_factory.hpp 创建。

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "rag/inference/embedding_model.hpp"

namespace autumndawn::rag::inference {

/// 硅基流动（SiliconFlow）Embedding API 客户端。
/// OpenAI 兼容协议：POST {baseUrl}/embeddings
class SiliconFlowEmbeddingClient : public IEmbeddingModel {
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
