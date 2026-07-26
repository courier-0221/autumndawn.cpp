#pragma once

// 云端实现类的内部头文件：不暴露到 include/，外部统一经 rag/model_factory.hpp 创建。

#include <memory>
#include <string>
#include <vector>

#include "rag/inference/rerank_model.hpp"

namespace autumndawn::rag::inference {

/// 硅基流动（SiliconFlow）Rerank API 客户端。
/// 协议：POST {baseUrl}/rerank（模型如 BAAI/bge-reranker-v2-m3）。
class SiliconFlowRerankClient : public IRerankModel {
public:
    SiliconFlowRerankClient(std::string baseUrl, std::string apiKey, std::string model);
    ~SiliconFlowRerankClient() override;

    std::vector<RerankHit> rerank(const std::string& query,
                                  const std::vector<std::string>& docs,
                                  int topN) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace autumndawn::rag::inference
