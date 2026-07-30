#include "rag/router.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "rag/log.hpp"
#include "rag/prompt_template.hpp"

namespace autumndawn::rag {

namespace {

using json = nlohmann::json;

/// 把 routes 拼成 prompt 里插进 {routes_list} 的多行字符串。
std::string formatRoutesList(const std::vector<RouteChoice>& routes) {
    std::string out;
    for (const auto& r : routes) {
        out += "- ";
        out += r.name;
        out += ": ";
        out += r.description;
        out += '\n';
    }
    // 去掉末尾多余的换行符。
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

/// 从模型可能带 markdown 代码块或前后闲话的输出里，抠出第一段 JSON 对象。
/// 命中 `{` 后按大括号计数找匹配 `}` 收尾；找不到则返回原字符串。
std::string extractJsonObject(const std::string& text) {
    const auto start = text.find('{');
    if (start == std::string::npos) return text;
    int depth = 0;
    for (std::size_t i = start; i < text.size(); ++i) {
        char c = text[i];
        if (c == '{') ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) return text.substr(start, i - start + 1);
        }
    }
    return text.substr(start);
}

}  // namespace

// ---------------- LlmRouter --------------------------------------------------

LlmRouter::LlmRouter(std::shared_ptr<inference::IChatModel> chat, std::vector<RouteChoice> routes,
                     Options options)
    : chat_(std::move(chat)), routes_(std::move(routes)), options_(std::move(options)) {
    if (routes_.empty()) {
        throw std::invalid_argument("LlmRouter: routes must not be empty");
    }
    if (options_.fallbackRoute.empty()) {
        options_.fallbackRoute = routes_.front().name;
    } else {
        auto it = std::find_if(routes_.begin(), routes_.end(), [&](const RouteChoice& r) {
            return r.name == options_.fallbackRoute;
        });
        if (it == routes_.end()) {
            throw std::invalid_argument("LlmRouter: fallbackRoute '" + options_.fallbackRoute +
                                        "' is not a registered route name");
        }
    }
    // 系统提示里的 {routes_list} 在构造时一次性填好；每次调用只替换 {question}。
    systemMessage_ = prompt::format(options_.systemPrompt,
                                    {{"routes_list", formatRoutesList(routes_)}});
}

IRouter::Decision LlmRouter::route(const std::string& query) {
    Decision fallback{options_.fallbackRoute, "fallback: router failed"};

    std::string reply;
    try {
        const std::string userMsg = prompt::format(options_.userTemplate, {{"question", query}});
        reply = chat_->chat({{"system", systemMessage_}, {"user", userMsg}}, options_.chat);
    } catch (const std::exception& e) {
        LOG(WARNING) << "LlmRouter: chat failed (" << e.what() << "), fallback -> "
                     << fallback.route;
        return fallback;
    }

    json parsed;
    try {
        parsed = json::parse(extractJsonObject(reply));
    } catch (const json::parse_error& e) {
        LOG(WARNING) << "LlmRouter: reply is not valid JSON (" << e.what()
                     << "), raw='" << reply << "', fallback -> " << fallback.route;
        return fallback;
    }

    if (!parsed.is_object() || !parsed.contains("route") || !parsed["route"].is_string()) {
        LOG(WARNING) << "LlmRouter: reply missing 'route' string, raw='" << reply
                     << "', fallback -> " << fallback.route;
        return fallback;
    }

    Decision decision;
    decision.route = parsed["route"].get<std::string>();
    if (parsed.contains("reason") && parsed["reason"].is_string()) {
        decision.reason = parsed["reason"].get<std::string>();
    }

    // 校验 route 名字必须在注册列表里。
    const bool known = std::any_of(routes_.begin(), routes_.end(), [&](const RouteChoice& r) {
        return r.name == decision.route;
    });
    if (!known) {
        LOG(WARNING) << "LlmRouter: route '" << decision.route
                     << "' not in registered list, fallback -> " << fallback.route;
        decision.route = fallback.route;
        decision.reason = "fallback: unknown route '" + parsed["route"].get<std::string>() + "'";
    }
    return decision;
}

// ---------------- RoutedRetriever -------------------------------------------

RoutedRetriever::RoutedRetriever(
    std::shared_ptr<IRouter> router,
    std::unordered_map<std::string, std::shared_ptr<IRetriever>> routes,
    std::string defaultRoute)
    : router_(std::move(router)), routes_(std::move(routes)), defaultRoute_(std::move(defaultRoute)) {
    if (routes_.empty()) {
        throw std::invalid_argument("RoutedRetriever: routes must not be empty");
    }
    if (!defaultRoute_.empty() && !routes_.count(defaultRoute_)) {
        throw std::invalid_argument("RoutedRetriever: defaultRoute '" + defaultRoute_ +
                                    "' is not among registered routes");
    }
}

std::vector<RetrievedChunk> RoutedRetriever::retrieve(const std::string& query, int topK) {
    if (topK <= 0) return {};
    lastDecision_ = router_->route(query);

    auto it = routes_.find(lastDecision_.route);
    if (it == routes_.end()) {
        // Router 已尽力，但决策名在本层无对应子检索器：走 defaultRoute（或任意一条）。
        const std::string chosen = !defaultRoute_.empty() ? defaultRoute_ : routes_.begin()->first;
        LOG(WARNING) << "RoutedRetriever: no retriever for route '" << lastDecision_.route
                     << "', using '" << chosen << "'";
        lastDecision_.reason =
            "fallback: no retriever for route '" + lastDecision_.route + "'";
        lastDecision_.route = chosen;
        it = routes_.find(chosen);
    }
    return it->second->retrieve(query, topK);
}

}  // namespace autumndawn::rag
