# RAG From Scratch —— C++ 实现方案与迭代计划 (v2, Review 后)

> 参考：<https://github.com/langchain-ai/rag-from-scratch>（5 份 notebook，18 小节）。
> 目标：**从入门到深入理解 RAG**。C++ 版本只亲手实现"高价值 + 有代表性"的子集，其余"了解原理即可"的能力预留清晰接口，方便后续按需补肉。
> 已有基础：[examples/tutorials/embedding_test/](../examples/tutorials/embedding_test/) 已经把
> "文本 → bge-m3 → ObjectBox HNSW → cosine Top-K" 全链路跑通，可以直接作为脚手架继续演进。

---

## 0. 学习路径优先级（针对"从入门到深入"目标）

把 18 节按 **"手写实现的教学价值 × 生产实用度 × 工程量"** 分三档：

### 🟢 P0 · 必学必写（RAG 的骨架 + 最经典的改进）
| # | 小节 | 为什么必学 |
|---|---|---|
| 1–4 | Overview / Indexing / Retrieval / Generation | RAG 三段式骨架，不实现等于没入门 |
| 5 | **Multi-Query** | Query rewriting 的入门形态，理解"用 LLM 扩展 query" |
| 6 | **RAG-Fusion (RRF)** | Reciprocal Rank Fusion 是通用融合技巧，20 行代码，通吃各种 retriever |
| 9 | **HyDE** | "让 LLM 先假答一遍再去检索"——理解语义空间的对称性，实现极简 |
| 15 | **Re-ranking** | 生产 RAG 里 recall→precision 的关键一步；bge-reranker 效果立竿见影 |
| 10a | **Logical Routing** | 引入 "LLM 输出结构化 JSON 做决策" 的思想，是后面 Router/Self-RAG 的基石 |

> 只做完 P0，你就掌握了 90% 生产 RAG 系统的常见套路。

### 🟡 P1 · 骨架实现 + 留扩展点（了解原理，代码只做最小版）
| # | 小节 | 处理方式 |
|---|---|---|
| 7 | Decomposition | `IRetriever` 允许递归组合，写一个最简"拆 2 个子问题"的版本，复杂拆分留 TODO |
| 8 | Step-back | 一个 prompt trick，实现同 HyDE 结构，代码几乎白送 |
| 10b | Semantic Routing | 预注册若干 route 描述向量+cosine 选路，20 行搞定 |
| 16 | CRAG（简化版） | 只做 "LLM 给检索结果打分 → 低于阈值时用另一个 retriever" 的降级；web search 留桩 |

### 🔴 P2 · 只读原理不实现（在文档留占位 + 接口 hook，将来能填肉）
| # | 小节 | 为什么先不写 | 未来怎么补 |
|---|---|---|---|
| 11 | Query Construction (text→SQL / self-query) | 与业务/schema 强耦合，脱离具体场景没意义 | 预留 `IStructuredQueryBuilder` 接口 |
| 12 | Multi-representation Indexing | 存储层改动大（要多 embedding 字段/多 entity） | `Document` schema 预留 `summary_embedding` 注释 |
| 13 | RAPTOR | 需要聚类 + 递归摘要，工程量大 | 预留 `IIndexer` 允许分层写入 |
| 14 | ColBERT (late-interaction) | 需要多向量索引，ObjectBox 目前不支持；bge-m3 sparse 也在这一档 | 记入 backlog |
| 17 | Self-RAG | 多轮 LLM 自评，token 花销大且状态机复杂 | Pipeline 已是可组合结构，未来加一个 `SelfRagPipeline` 即可 |
| 18 | Adaptive-RAG | 本质是 Router × Pipeline 的组合器 | P0 的 Router + P1 的 CRAG 已经把零件备齐 |

**学习节奏建议**：
1. **入门 (v0.1)**：亲手做 P0 里的 1–4，端到端能问答。
2. **进阶 (v0.2)**：加 P0 的 Multi-Query / RAG-Fusion / HyDE，感受"query 改写"对召回的影响。
3. **实用 (v0.3)**：加 P0 的 Re-ranking + Logical Routing，逼近生产可用。
4. **拓展 (v0.4)**：把 P1 四项以最小实现补上，同时把 P2 的接口/占位打好。
5. **深入 (自选)**：从 P2 里挑最感兴趣的一节（推荐 RAPTOR 或 Self-RAG）按接口填肉，作为毕业项目。

---

## 1. 课程思路 → C++ 模块映射（按优先级重排）

原课程 5 个 notebook 共 18 小节，我把它们分成 5 个阶段，逐条给出 C++ 版对应实现的着力点。
（Python 版几乎都基于 LangChain 抽象；C++ 版我们不复刻 LangChain，而是把最小必要接口抽出来，
避免过度设计。）

### 🟢 Phase A · Baseline（P0：小节 1–4）
| 课程小节 | 关键点 | C++ 版对应 |
|---|---|---|
| 1. Overview | RAG 三段式：Index → Retrieve → Generate | 定义 `IRetriever` / `IGenerator` 接口 + `RagPipeline`（组合器） |
| 2. Indexing | Chunk → Embed → Vector Store | 复用 `SiliconFlowEmbeddingClient` + 新增 `TextSplitter`（递归按段落/字符）+ ObjectBox `Box<Document>` |
| 3. Retrieval | 相似度检索（Cosine/Top-K） | 复用 `Document_::embedding.nearestNeighbors`，封装 `ObjectBoxRetriever` implements `IRetriever` |
| 4. Generation | Prompt 模板 + LLM 调用 | 新增 `DeepSeekChatClient`（OpenAI 兼容 `/v1/chat/completions`，同风格自研 HTTP）+ `PromptTemplate` |

**里程碑 v0.1**：`examples/rag_apps/rag_basic/` 支持 `ingest <file>` + `ask <question>`，端到端跑通中文问答。

### 🟢 Phase B · Query Translation 精选（P0：小节 5 / 6 / 9）
| 课程小节 | C++ 版对应 |
|---|---|
| 5. Multi-Query | `MultiQueryRetriever`：装饰任意 `IRetriever`，内部调用 `IGenerator` 生成 N 条改写 query → 并联检索 → 去重合并 |
| 6. RAG-Fusion | `fusion::rrf(std::vector<Ranking>, k=60)` 独立函数；`MultiQueryRetriever` 可切换合并策略 (union / rrf) |
| 9. HyDE | `HydeRetriever`：装饰器，先调 `IGenerator` 生成 hypothetical answer，再走 embedding 检索 |

**里程碑 v0.2**：`examples/rag_apps/rag_query_translate/ --strategy=multi_query|rrf|hyde|plain` 可对同一问题打印召回差异。

### 🟢 Phase C · Re-rank + Routing（P0：小节 10a + 15）
| 课程小节 | C++ 版对应 |
|---|---|
| 10a. Logical Routing | `LlmRouter`：让 LLM 返回 JSON `{ "route": "...", "reason": "..." }`；`nlohmann::json` 做 schema 校验；Router 是"选 pipeline"的元组件 |
| 15. Re-ranking | `SiliconFlowRerankClient` 已提前实现（POST `/v1/rerank`，模型 `BAAI/bge-reranker-v2-m3`）；v0.3 只需封装 `RerankingRetriever` 装饰器：先 top-N 召回 → rerank → top-K 返回 |

**里程碑 v0.3**：`RagPipeline` = `Router → (若干 Retriever) → Reranker → Generator`，达到"生产级最小可用"标准。

### 🟡 Phase D · P1 最小实现（小节 7 / 8 / 10b / 16）
| 课程小节 | 最小实现 |
|---|---|
| 7. Decomposition | `DecompositionPipeline`：LLM 拆 2 个子问题 → 各自跑一次 pipeline → 拼 context 再生成，递归深度硬编码为 1 |
| 8. Step-back | `StepBackRetriever`：装饰器，用固定 prompt 生成 step-back question，两次检索拼接 |
| 10b. Semantic Routing | `SemanticRouter`：初始化时对 route 描述文本做一次 embedding → 每次 query cosine 选最近 |
| 16. CRAG（简化） | `CorrectiveRetriever`：主 retriever 命中不足时（LLM 打分 < 阈值）走 fallback retriever；web search 用 `NullWebSearch` 占位 |

**里程碑 v0.4**：以上 4 项均为 `IRetriever` / `IPipeline` 装饰器，可任意串联；同时把 P2 的接口占位（`IStructuredQueryBuilder`、`IIndexer`、`IReflectiveJudge`）写好但不实现。

### 🔴 Phase E · P2 占位与未来扩展（小节 11 / 12 / 13 / 14 / 17 / 18）

**不写实现，只做以下三件事**：
1. 每个未实现能力在 `include/rag/future/` 下留一个 `xxx.hpp` 说明接口意图 + 参考 Python notebook 位置；
2. `doc/future_topics.md` 汇总原理笔记（Multi-representation / RAPTOR / ColBERT / Self-RAG / Adaptive-RAG / Query Construction 各一节，1–2 页 md）；
3. 在 `RagPipeline` / `Document` schema 上预留扩展点：
   - `Document.fbs` 里注释 `// TODO(v2): summary_embedding: [float]; hnsw-dimensions=1024`；
   - `RagPipeline` 已经是"接口组合"结构，未来加 `SelfRagPipeline` / `AdaptiveRagPipeline` 只需新增类；
   - `ObjectBoxRetriever` 支持传入 `QueryCondition`，方便未来接 `IStructuredQueryBuilder`。

未来若真去实现 RAPTOR（小节 13），聚类用手写 mini-KMeans（≤300 行），不引入 `mlpack`/`Eigen`；所有 App 保持 CLI，不做 Web UI。

**里程碑 v1.0**：P0 + P1 全部实现 + P2 占位齐备；`examples/rag_apps/rag_bench/` 产出策略对比表（plain / multi-query / rrf / hyde / rerank / rerank+multi-query），CSV + 简单 markdown 报告。

---

## 2. 架构与目录结构演进

现在 embedding_test 是一个"平铺 demo"。为了后续 5 个阶段能复用，需要把公共能力抽成静态库；
同时把"和模型推理交互"的部分单独收拢到一个 `inference/` 子模块 —— 现在虽然只有云端 API，但从架构上就把它当作一个可替换的推理层，
未来加本地推理（llama.cpp / ggml / bge-m3 onnx）时只需要在 `inference/local/` 下加实现，不影响上层。

头文件路径用单层 `rag/` 前缀（例如 `#include "rag/pipeline.hpp"`），避免过深嵌套。

```
autumndawn.cpp/
├── include/rag/
│   ├── inference/                      # 推理接入层：抽象接口（部署形态无关）
│   │   ├── inference_error.hpp         #   InferenceError
│   │   ├── embedding_model.hpp         #   IEmbeddingModel
│   │   ├── chat_model.hpp              #   IChatModel
│   │   └── rerank_model.hpp            #   IRerankModel
│   ├── retriever.hpp                   # IRetriever
│   ├── generator.hpp                   # IGenerator
│   ├── text_splitter.hpp
│   ├── prompt_template.hpp
│   ├── fusion.hpp                      # RRF / weighted
│   ├── router.hpp
│   ├── model_factory.hpp               # 推理模型装配工厂（按 provider 创建 I*Model）
│   └── pipeline.hpp                    # RagPipeline
├── src/
│   ├── inference/                      # 推理接入层：实现
│   │   ├── http_client.hpp/.cpp        #   通用 HTTP 小工具（TLS / 超时 / 429 退避）
│   │   ├── cloud/                      #   云端 API 实现（.hpp 为内部头，不暴露到 include/）
│   │   │   ├── siliconflow_embedding_client.hpp/.cpp
│   │   │   ├── deepseek_chat_client.hpp/.cpp
│   │   │   └── siliconflow_rerank_client.hpp/.cpp
│   │   └── local/                      #   预留：端侧进程内推理实现
│   │       └── .gitkeep
│   ├── retrieval/
│   │   ├── objectbox_retriever.cpp
│   │   ├── multi_query_retriever.cpp
│   │   ├── hyde_retriever.cpp
│   │   ├── stepback_retriever.cpp
│   │   ├── reranking_retriever.cpp
│   │   ├── corrective_retriever.cpp
│   │   └── semantic_router.cpp
│   ├── text_splitter.cpp
│   ├── prompt_template.cpp
│   ├── fusion.cpp
│   ├── router.cpp
│   ├── config.cpp                      # provider 判别的配置加载
│   ├── model_factory.cpp               # createEmbeddingModel / createChatModel / createRerankModel
│   └── pipeline.cpp
├── examples/
│   ├── rag_apps/               # 跟着版本迫代的 RAG 演示程序系列
│   │   ├── rag_basic/          # v0.1 (Phase A)
│   │   ├── rag_query_translate/# v0.2 (Phase B)
│   │   ├── rag_route/          # v0.3 (Phase C)
│   │   ├── rag_index_adv/      # v0.4 (Phase D + E 占位)
│   │   └── rag_bench/          # v1.0 策略对比
│   └── tutorials/              # 第三方库用法教程，不跟版本迫代
│       ├── embedding_test/     # 保留，作为 ObjectBox 向量检索最小示例
│       └── objectbox/quickstart
├── test/                   # 各模块单测（GoogleTest 或 doctest）
├── third_party/
│   ├── nlohmann/json.hpp
│   └── objectbox/
└── doc/
    ├── rag_cpp_plan.md     # 本文
    └── (每个 Phase 完成后追加设计说明)
```

CMake：顶层新增 `CMakeLists.txt`，把 `include/` + `src/` 打成 `autumndawn_rag`（先做 STATIC），
`examples/rag_apps/*` 和 `test/` 都 `target_link_libraries(... autumndawn_rag)`。
`examples/tutorials/embedding_test/` 保持独立可编译，以便日后 review 学习路径。

**分层说明**：

- **inference 层**（`inference/`）：只关心"文本 → 向量 / 消息 → 回复 / (query, docs) → 相关性分数"三件事，
  完全不知道有 ObjectBox / retriever / pipeline / config 的存在。接口按能力中性命名（`I*Model`），部署形态无关；
  云端实现叫 `*Client`、端侧实现叫 `*Model`，实现类头文件收在 `src/inference/cloud|local/` 内部，不暴露到
  `include/`，外部统一经 `rag/model_factory.hpp` 按配置里的 `provider` 字段装配。future 加端侧进程内推理
  （llama.cpp、bge-m3 onnx）时，只需在 `src/inference/local/` 下新增实现类，实现同一组 `I*Model` 接口，上层零改动。
- **retrieval 层**（`retrieval/`）：`IRetriever` 及其所有装饰器（multi-query / hyde / step-back / rerank / corrective / semantic-router）。
  只依赖 `IEmbeddingModel` / `IChatModel` / `IRerankModel` 抽象接口，不依赖任何具体供应商。
- **pipeline 层**：`RagPipeline` = `Router → Retriever → Generator`，最上层组合器。

**命名空间约定**：

C++ 语言级的分层用 `namespace` 来表达，与目录一一对应。顶层用 `autumndawn::rag`（跟库名 `libautumndawn_rag` 和 include 前缀 `rag/` 三者对齐）：

```cpp
namespace autumndawn::rag {                        // 顶层：pipeline / generator / text_splitter / prompt_template / fusion / router

    namespace inference {                          // 推理接入层
        // 接口：IEmbeddingModel / IChatModel / IRerankModel（include/rag/inference/）
        // 云端实现：SiliconFlowEmbeddingClient / DeepSeekChatClient / SiliconFlowRerankClient
        //          （src/inference/cloud/ 内部头，经 model_factory 装配）
    }

    namespace retrieval {                          // 检索层
        // IRetriever / ObjectBoxRetriever / MultiQueryRetriever / HydeRetriever / ...
    }

    namespace detail {                             // 内部工具（HttpClient、json helper 等），不对外承诺稳定
    }

    namespace future {                             // Phase E 占位接口（IStructuredQueryBuilder / IReflectiveJudge …）
    }
}
```

**使用侧惯例**：

- 库内文件顶部一律 `namespace autumndawn::rag { ... }`，禁止 `using namespace` 泄漏到头文件；
- Apps 侧允许在 `.cpp` 里 `namespace rag = autumndawn::rag;` 起短别名，避免 `autumndawn::rag::inference::SiliconFlowEmbeddingClient` 打太长；
- 命名规范：类 / 类型 `PascalCase`，函数 / 变量 `camelCase`，常量 `kPascalCase`，宏 `AUTUMNDAWN_XXX`（跟已有 `OBX_CPP_FILE` 风格一致）；
- 已有的 `SiliconFlowEmbeddingClient`（现在裸在全局）在 v0.1 迁移到新库时一并放进 `autumndawn::rag::inference`。

---

## 3. 关键技术选型

### 3.1 向量存储

**主选：ObjectBox**
- 已经在用，HNSW + Cosine 就够 v1.0；同库同时存原文 + 元数据 + 向量，Phase 3 的 metadata filter 很自然。
- 缺点：只有 HNSW（无 IVF/PQ），闭源二进制无法自定义，schema 变更要跑 generator；单机、不支持水平扩展。

**备选（按需引入，不进 v1.0）：**
| 备选 | 场景 | 说明 |
|---|---|---|
| **hnswlib**（header-only） | 想要更透明的 HNSW 参数调优 / benchmark | 只做向量，元数据要另存 SQLite |
| **FAISS** | 大规模、GPU、多种索引（IVF-PQ） | 依赖较重，暂无必要 |
| **SQLite + vss/sqlite-vec 扩展** | 想用一个进程内 SQL | 生态偏 Python；C++ 接入需要自己包 |
| **Qdrant / Milvus / LanceDB (远端)** | 服务化、多进程共享 | 引入 gRPC/HTTP 客户端；只有多进程共享向量库时才值得 |

结论：**Phase 1~5 都用 ObjectBox；只有当我们进入 "对比索引/大规模" benchmark 场景时再引入 hnswlib 作为对照实现。**

### 3.2 Embedding / Chat / Rerank —— 三个云端 API

| 用途 | Provider | Endpoint | 模型 | 客户端 |
|---|---|---|---|---|
| Embedding | SiliconFlow | `https://api.siliconflow.cn/v1/embeddings` | `BAAI/bge-m3`（1024 维） | `SiliconFlowEmbeddingClient` ✅ 已实现 |
| Chat | **DeepSeek** | `https://api.deepseek.com/v1/chat/completions` | `deepseek-v4-pro`（已确认） | `DeepSeekChatClient` 🆕 v0.1 |
| Rerank | SiliconFlow | `https://api.siliconflow.cn/v1/rerank` | `BAAI/bge-reranker-v2-m3` | `SiliconFlowRerankClient` ✅ 已实现 |

**Chat 客户端实现方式（已定）**：

手写 `DeepSeekChatClient`，跟已有 `SiliconFlowEmbeddingClient` 完全对称（同一个 `HttpClient` 小工具，保留 TLS 证书校验、超时、429 退避），约 100 行。DeepSeek 协议 OpenAI 兼容，POST `/v1/chat/completions` 直接可用。

`third_party/openai/openai.hpp` 已从仓库删除：它默认在 `Session` 构造里 `ignoreSSL()`（`CURLOPT_SSL_VERIFYPEER=0` / `CURLOPT_SSL_VERIFYHOST=0`），走公网 HTTPS 到 DeepSeek 等同关闭中间人防护，不符合本项目安全基线。model 名不做 hard-code，统一写在 `rag_config.json` 里。

**config 合并（已确认）**：`emb_config.json` 升级为 `rag_config.json`，顶层按功能块包裹，
当前只有 `model` 一项（后续平级新增 `retrieval` / `splitter` / `storage` 等）：

```jsonc
{
  "model": {
    "embedding": {
      "provider": "siliconflow",
      "base_url": "https://api.siliconflow.cn/v1",
      "api_key": "sk-...",
      "model": "BAAI/bge-m3"
    },
    "chat": {
      "provider": "deepseek",
      "base_url": "https://api.deepseek.com/v1",
      "api_key": "sk-...",
      "model": "deepseek-v4-pro"
    },
    "rerank": {
      "provider": "siliconflow",
      "base_url": "https://api.siliconflow.cn/v1",
      "api_key": "sk-...",
      "model": "BAAI/bge-reranker-v2-m3"
    }
  }
}
```

`rag_config.json` 加入 `.gitignore`；仓库只保留 `rag_config.example.json`。日志中不打印任何 `api_key`。

**provider 判别与工厂装配（v0.1 落地）**：`model` 下三个段结构一致，由 `provider` 字段判别后端——
cloud 接受具名厂商标识（`siliconflow` / `deepseek`，不再接受 `openai_compatible` 这种协议名），
要求 `base_url`/`api_key`/`model`；local（`llama_cpp` / `onnx`，端侧进程内推理预留）要求 `model_path`
（可选 `n_ctx`/`n_threads`/`n_gpu_layers`）。配置结构为 `std::variant<CloudInferenceConfig, LocalInferenceConfig>`；
`rerank` 段为 `std::optional`，缺失或 api_key 为占位符时视为未启用；`provider` 字段必填，不接受缺省。
装配统一走 `rag/model_factory.hpp` 的 `createEmbeddingModel` / `createChatModel` / `createRerankModel`：
cloud 分支内部**按 `provider` 真分派**（embedding=`siliconflow` → `SiliconFlowEmbeddingClient`；
chat=`deepseek` → `DeepSeekChatClient`；rerank=`siliconflow` → `SiliconFlowRerankClient`），
不匹配的组合（如 embedding=`deepseek`、chat=`siliconflow`、rerank=`deepseek`）抛 `ConfigError`，
未来新接供应商时在 factory 内新增 `if` 分支即可；local 分支目前抛 `ConfigError`（实现待落地，
届时配套 CMake 条件编译选项）。
接口约定实现不保证线程安全，调用方串行访问；`IChatModel` 后续按需要以默认实现方式补 `chatStream`。

### 3.3 分词 / Chunking

- v0.1：字符长度 + 段落边界的 `RecursiveCharacterTextSplitter`（LangChain 里最常用的那种），无需外部依赖。参数：`chunk_size=512 tokens ≈ 1200 chars`, `chunk_overlap=64`（中文按字符估算 token）。
- v0.4+：如果 chunk 精度/成本要求上来，再引入 tokenizer（选项：`tiktoken-cpp`, `sentencepiece`）。

### 3.4 Prompt 模板

- 极简：`PromptTemplate::format(name, {{"question", q}, {"context", ctx}})`，用 `{key}` 占位符 + `std::string::find` 替换即可。
- 模板集中放在 `include/rag/prompts/*.txt`（编译期 `configure_file` 拷贝到运行目录）。

### 3.5 JSON / HTTP / 日志

- JSON：`nlohmann/json`（已在 third_party）。
- HTTP：`libcurl`（已装）。抽个 `HttpClient` 小工具，统一超时/重试/TLS/UA。
- 日志：v0.1 用 `iostream` 打点；后续视需要引入 `spdlog`。
- 错误处理：走异常（`RagError` 基类 + 子类 `HttpError` / `LlmError` / `IndexError`）。

### 3.6 并发 / 限流

- 硅基流动免费额度存在 QPS 限制，`embedBatch` 需要限速 + 指数回退（HTTP 429）。Phase 1 里就把它做进 `HttpClient`。

### 3.7 测试

- 单元测试：`doctest`（单头，最省事）覆盖 `TextSplitter` / `PromptTemplate` / `fusion::rrf` / `Router` schema 校验等纯逻辑。
- 集成测试：mock 一个 `IEmbeddingModel`（返回固定向量）+ 一个 `IGenerator`（返回固定字符串），用真实 ObjectBox 跑 pipeline，保证 CI 不消耗 API。

---

## 4. 迭代计划（按学习路径重排）

| 版本 | 对应课程 | 交付物 | 验收 |
|---|---|---|---|
| **v0.1** | Phase A（P0：1–4） | 抽 `libautumndawn_rag` + `DeepSeekChatClient` + `RagPipeline` + `TextSplitter` + `examples/rag_apps/rag_basic` | `ingest corpus_sample.txt` 后 `ask` 提问，能引用相关 chunk 并生成中文答案 |
| **v0.2** | Phase B（P0：5 / 6 / 9） | `MultiQueryRetriever` + `fusion::rrf` + `HydeRetriever` + `examples/rag_apps/rag_query_translate --strategy=...` | 同一问题 4 种策略（plain / multi-query / rrf / hyde）的召回对比打印在终端 |
| **v0.3** | Phase C（P0：10a + 15） | `LlmRouter` + `RerankingRetriever`（`SiliconFlowRerankClient` 已就绪） | 多语料源自动路由 + rerank 后 top-3 相关性肉眼可见提升 |
| **v0.4** | Phase D（P1：7 / 8 / 10b / 16 最小实现） + Phase E 接口占位 | 四个装饰器 + `include/.../future/` 接口 + `doc/future_topics.md` | 装饰器可以任意串联，占位接口 lint 通过 |
| **v1.0** | `examples/rag_apps/rag_bench` | 多策略对比脚本 + CSV/markdown 报告 | 一份 QA 集上各策略 latency / token / recall 对比表 |

**节奏建议**：
- **v0.1 是学习曲线最陡的一段**（要打库 + 打通 chat），做完你就有一个能问答的最小 RAG，可以直接用它读你自己的资料。
- **v0.2 → v0.3 每一版都会带来肉眼可见的效果提升**，是最有成就感的两步。
- **v0.4 是"广度"版本**：即便某些能力只是骨架实现，你也已经完整走过 RAG 所有主流套路。
- P2 的深入（RAPTOR / Self-RAG 等）留作后续独立选题，代码骨架届时已经就绪。

每个版本落地后追加对应设计文档到 `doc/`。

---

## 5. 下一步动作

无需再多讨论，我按上面结论直接进入 v0.1：

1. 顶层 `CMakeLists.txt`，`include/rag/` + `src/` 骨架；
2. 把 `SiliconFlowEmbeddingClient` 挪进新库，`examples/tutorials/embedding_test` 改成链接新库（保持它继续可跑，作为最小示例）；
3. 新增 `HttpClient`（保留 TLS 校验、带超时/429 退避）；
4. 新增 `DeepSeekChatClient`；
5. 新增 `TextSplitter`（递归字符/段落）+ `PromptTemplate` + `ObjectBoxRetriever` + `RagPipeline`；
6. 新增 `examples/rag_apps/rag_basic/`（`ingest` / `ask` 两条命令）；
7. `rag_config.example.json` 落库，README 简单更新。

完成后我再喊你 review 一遍接口签名（`IRetriever` / `IGenerator` / `Document` schema）就往 v0.2 走。
