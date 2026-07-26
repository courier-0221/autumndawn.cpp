#include "rag/prompt_template.hpp"

namespace autumndawn::rag::prompt {

std::string format(const std::string& tmpl,
                   const std::unordered_map<std::string, std::string>& vars) {
    std::string out;
    out.reserve(tmpl.size() + 64);

    std::size_t i = 0;
    while (i < tmpl.size()) {
        char c = tmpl[i];
        if (c == '{') {
            auto close = tmpl.find('}', i + 1);
            if (close != std::string::npos) {
                std::string key = tmpl.substr(i + 1, close - i - 1);
                auto it = vars.find(key);
                if (it != vars.end()) {
                    out += it->second;
                    i = close + 1;
                    continue;
                }
            }
        }
        out += c;
        ++i;
    }

    return out;
}

}  // namespace autumndawn::rag::prompt
