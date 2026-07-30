#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rag/inference/chat_model.hpp"
#include "rag/retriever.hpp"

namespace autumndawn::rag {

/// 一条可路由到的目标（检索源 / pipeline）：
///   - `name`：路由标识，Router 输出该字段作为决策；
///   - `description`：给 LLM 看的自然语言说明，用于选路依据。
struct RouteChoice {
    std::string name;
    std::string description;
};

/// Router 抽象：给一个 query，从若干候选里选出 1 个 route。
/// v0.3 只做单选（Logical Routing，课程小节 10a）；多选 / 分层路由留待后续版本。
class IRouter {
public:
    struct Decision {
        std::string route;   ///< 选中的 route 名（保证一定命中已注册的 RouteChoice::name）
        std::string reason;  ///< 供上层日志/演示打印的选路理由（可为空）
    };

    virtual ~IRouter() = default;

    virtual Decision route(const std::string& query) = 0;
};

/// LLM 路由器（课程小节 10a Logical Routing）。
///
/// 让 LLM 阅读候选路由的自然语言描述，输出结构化 JSON `{"route":..., "reason":...}`
/// 决定走哪条 pipeline。用 `nlohmann::json` 做 schema 校验：字段缺失或
/// route 名不在注册列表中时，回退到 `Options::fallbackRoute`（默认第一条路由）。
///
/// 线程安全契约：不保证；调用方串行访问。
class LlmRouter : public IRouter {
public:
    struct Options {
        /// 系统提示。占位符：{routes_list}（自动填成"- name: description"多行）。
        std::string systemPrompt =
            "你是一个路由分派器。用户会给你一个问题，请从下列候选中选出**最相关的一个**："
            "\n{routes_list}\n"
            "只输出一个 JSON 对象，格式：{\"route\": \"<name>\", \"reason\": \"<简短中文理由>\"}。"
            "route 字段必须严格等于上面某个候选的名字。不要输出多余内容、不要加代码块。";
        /// user 消息模板，占位符：{question}。
        std::string userTemplate = "问题：{question}";
        /// JSON 解析失败或 route 名非法时的回退路由；空串则用注册顺序第一条。
        std::string fallbackRoute;
        inference::ChatOptions chat;
    };

    LlmRouter(std::shared_ptr<inference::IChatModel> chat, std::vector<RouteChoice> routes)
        : LlmRouter(std::move(chat), std::move(routes), Options{}) {}
    LlmRouter(std::shared_ptr<inference::IChatModel> chat, std::vector<RouteChoice> routes,
              Options options);

    Decision route(const std::string& query) override;

    const std::vector<RouteChoice>& routes() const { return routes_; }

private:
    std::shared_ptr<inference::IChatModel> chat_;
    std::vector<RouteChoice> routes_;
    Options options_;
    std::string systemMessage_;  ///< 预填好 {routes_list} 之后的 system 提示
};

/// 路由检索装饰器：把 `IRouter` 的选择结果映射回具体的 `IRetriever`。
///
/// 上层 pipeline 只看到一个 `IRetriever`：调 `retrieve()` 时，
/// 内部先让 router 选路，再把 query 转发给对应的子检索器。
/// 结合 `RerankingRetriever` 可组成 v0.3 的完整链路：
///   `RoutedRetriever( LlmRouter → { tech: R_tech, cook: R_cook, ...})`
///     被  `RerankingRetriever(...)` 装饰，再交给 `RagPipeline`。
class RoutedRetriever : public IRetriever {
public:
    /// @param router  路由决策器
    /// @param routes  name → 子检索器；必须至少包含 router 里注册过的每个 name
    /// @param defaultRoute  当 router 决策出的 name 未在 `routes` 中登记时使用；
    ///                     空串则回退到 routes 里的任意一条（构造时校验必须非空）
    RoutedRetriever(
        std::shared_ptr<IRouter> router,
        std::unordered_map<std::string, std::shared_ptr<IRetriever>> routes,
        std::string defaultRoute = "");

    std::vector<RetrievedChunk> retrieve(const std::string& query, int topK) override;

    /// 最近一次 retrieve 里 router 给出的选路决策，供演示程序打印。
    const IRouter::Decision& lastDecision() const { return lastDecision_; }

private:
    std::shared_ptr<IRouter> router_;
    std::unordered_map<std::string, std::shared_ptr<IRetriever>> routes_;
    std::string defaultRoute_;
    IRouter::Decision lastDecision_;
};

}  // namespace autumndawn::rag
