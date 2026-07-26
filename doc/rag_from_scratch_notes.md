# RAG From Scratch —— 原课程解读（中文入门到进阶）

> 原仓库：<https://github.com/langchain-ai/rag-from-scratch>（LangChain 官方，B 站/YouTube 视频合集共 18 节）
> 配套 5 份 Jupyter Notebook：`1_to_4` / `5_to_9` / `10_and_11` / `12_to_14` / `15_to_18`
> 本文用途：**帮你（RAG 初学者）在动手写代码前，先把 18 节课的"要解决什么问题、怎么解决、为什么这么做"一次性理清**。

---

## 0. 先建立地图：RAG 到底是什么

**Retrieval-Augmented Generation（检索增强生成）** 用一句话概括：

> LLM 参数里没有的知识（私域文档、最新数据），先用**检索**从外部知识库找出来，再作为**上下文**塞进 prompt，让 LLM 基于这段上下文生成回答。

这样比 fine-tune 便宜、比 long-context 直接塞全部文档可行，还能减少幻觉、支持数据实时更新。

一个最朴素的 RAG 系统只有三段：

```
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Indexing │ → │ Retrieval│ → │ Generation│
└──────────┘   └──────────┘   └──────────┘
  (离线一次)     (每次 query)    (每次 query)
```

课程的 18 节，本质上就是围绕这三段的**痛点**依次打补丁：

| 三段式痛点 | 对应改进小节 |
|---|---|
| 用户 query 表述差 → 召回不准 | 5 Multi-Query · 6 RAG-Fusion · 7 Decomposition · 8 Step-Back · 9 HyDE |
| 一个知识库不够/需要选路 | 10 Routing · 11 Query Construction |
| 单一 embedding 表达能力有限 | 12 Multi-representation · 13 RAPTOR · 14 ColBERT |
| 召回质量参差 → 上下文噪声大 | 15 Re-ranking |
| 检索结果差时需要兜底/自省 | 16 CRAG · 17 Self-RAG · 18 Adaptive-RAG |

---

## 1. Notebook 1–4：RAG 骨架三段式

**这一节看完就能跑一个玩具 RAG。**

### Part 1 · Overview
- 用 LangChain LCEL（管道语法）把 `retriever | prompt | llm | parser` 拼成一条链。
- 端到端例子：加载 Lilian Weng 的一篇博客 → chunk → embed → Chroma 向量库 → 检索 → GPT-3.5-turbo 回答"什么是 Task Decomposition"。

### Part 2 · Indexing（索引）
关键动作：**Load → Split → Embed → Store**。

1. **Load** —— `WebBaseLoader` / `PDFLoader` 等把原始数据读进来，统一成 `Document`（`page_content` + `metadata`）。
2. **Split** —— `RecursiveCharacterTextSplitter`：按 `["\n\n", "\n", " ", ""]` 递归切，尽量保住段落/句子边界。参数只有两个：`chunk_size`、`chunk_overlap`。
3. **Embed** —— `OpenAIEmbeddings` 把每个 chunk 变成一条 float 向量（OpenAI text-embedding-3-small 是 1536 维）。同一个 embedding 模型同时给 query 用。
4. **Store** —— 存进向量库（教程用 Chroma；生产上有 FAISS / Qdrant / Milvus / pgvector / ObjectBox 等选项）。

顺带演示了 `tiktoken` 数 token、cosine similarity 手工计算，帮你**建立"query 和文档在同一个向量空间里"这个直觉**。

### Part 3 · Retrieval（检索）
- 最基础的检索 = `vectorstore.similarity_search(query, k=N)`，本质就是"把 query embed 一下，用 cosine/内积去向量库找 top-K"。
- 检索器抽象成 `retriever = vectorstore.as_retriever(search_kwargs={"k": 1})`，后面所有小节都是**在这个 retriever 外面套装饰器**。

### Part 4 · Generation（生成）
- Prompt 模板固定套路：
  ```
  Answer the question based only on the following context:
  {context}
  
  Question: {question}
  ```
- 用 LangChain Hub 上的 `rlm/rag-prompt` 是社区经过打磨的版本。
- 最终链：
  ```
  {"context": retriever, "question": RunnablePassthrough()} 
    | prompt | llm | StrOutputParser()
  ```
- **只要 retrieved context 不含答案，LLM 就该老实说不知道**（"only based on"）——这是 RAG 抗幻觉的关键防线。

> **学到这里你已经掌握**：一个能问答自定义文档的最小可用系统。后面所有内容都是在**这个骨架的某一段**上做增强。

---

## 2. Notebook 5–9：Query Translation（改写用户 query）

**为什么需要？** 用户提的问题不一定是"对检索友好"的表述——太笼统、太多子问题、和文档里的用词不一致。这五节全部围绕**"如何把原始 query 改写/扩展成更适合检索的形式"**。

### Part 5 · Multi-Query（多查询改写）
- **思想**：用 LLM 从不同角度生成 N 条改写后的 query（一般 3~5 条），每条都独立检索，最后并集去重。
- **Prompt**：
  > You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database... Provide these alternative questions separated by newlines. Original question: {question}
- **链**：`prompt | LLM | split("\n") | retriever.map() | 合并去重`
- **收益**：cosine 相似度对"表述差异"敏感，多角度提问能显著提升召回率。**代价**：多打 N 次向量检索 + 一次 LLM。

### Part 6 · RAG-Fusion（多查询 + RRF）
- 在 Multi-Query 基础上，把"简单去重"升级为 **Reciprocal Rank Fusion (RRF)**：

  $$ \text{score}(d) = \sum_{q \in Q} \frac{1}{k + \text{rank}_q(d)} \quad (k=60 \text{ 常用}) $$

- **好处**：多个 query 都把文档 d 排到前面 → d 综合排名高；只被一个 query 排到中间 → 得分低但仍有机会入选。**不需要标定**、跨数据源通用、几十行代码搞定。
- **参考**：<https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1>
- **一个小口诀**：RRF 是 RAG 里最"性价比高"的融合手段，任何多路检索都值得先加它。

### Part 7 · Decomposition（问题分解）
- **场景**：复杂问题需要多步推理，例如 *"LLM agent 系统需要哪些组件？各自解决什么问题？"*。
- **思想**：让 LLM 把复杂问题拆成若干**可独立回答**的子问题，每个子问题独立跑 RAG，最后把 QA 对喂给最终 LLM 合成答案。
- **Prompt 关键句**："...break down the input into a set of sub-problems / sub-questions that can be answered in isolation."
- **两种拼接方式**：
  1. **Answer recursively** —— 上一个子问题的答案作为下一个子问题的额外上下文（链式）。
  2. **Answer individually** —— 各子问题并行，各自得答案后一次性合并（并行）。
- **代价**：LLM 调用数 × 分解层数，token 开销上升明显。

### Part 8 · Step-Back Prompting（后退一步）
- **思想**（论文 <https://arxiv.org/pdf/2310.06117.pdf>）：把具体问题**先抽象成一个更泛化的"step-back 问题"**，同时用两条路径检索——原问题走一路，泛化后问题走一路——最后拼接上下文一起回答。
- **Few-shot 示例**：
  ```
  Q: Could the members of The Police perform lawful arrests?
  Step-back: what can the members of The Police do?
  
  Q: Jan Sindel's was born in what country?
  Step-back: what is Jan Sindel's personal history?
  ```
- **直觉**：具体问题的匹配文档可能只有零散事实；抽象问题能触发"背景性段落"，让 LLM 有更完整的知识框架。
- **实现极简**：只是一个 few-shot prompt + 两次检索。

### Part 9 · HyDE（Hypothetical Document Embeddings）
- **论文**：<https://arxiv.org/abs/2212.10496>
- **思想**：query 和"答案段落"的向量空间往往不是对称的——用**答案的样子**去检索比用问题去检索更准。所以：
  1. 让 LLM 先"假答一遍"，生成一个 hypothetical 文档；
  2. 把这个 hypothetical 文档 embed；
  3. 用 hypothetical 的 embedding 去向量库找**真实**相似文档。
- **Prompt**：
  > Please write a scientific paper passage to answer the question. Question: {question}. Passage:
- **注意**：hypothetical 内容可以是错的，但**语义/词汇分布**接近真实答案就够了——检索靠的是空间距离，不是事实正确性。
- **极简实现价值**：装饰在任何 retriever 外面都行，代码不到 30 行。

> **五节 Query Translation 的选型建议**：
> - 只挑一个上手 → **RAG-Fusion**（Multi-Query + RRF，通用性最强）
> - 追求极简高价值 → **HyDE**
> - 面向复杂问答 / agent → **Decomposition**
> - Prompt-only trick → **Step-Back**

---

## 3. Notebook 10–11：Routing 与 Query Construction

**这两节的共同思想：让 LLM 输出结构化结果做决策，而不是只让它写自由文本。**

### Part 10 · Routing（路由）

#### 10-a Logical Routing（逻辑路由）
- **场景**：有多个知识库（Python 文档 / JS 文档 / Golang 文档...），根据问题选一条。
- **手段**：LLM function-calling / structured output，输出一个 Pydantic 对象：
  ```python
  class RouteQuery(BaseModel):
      datasource: Literal["python_docs", "js_docs", "golang_docs"]
  ```
- LLM 输出的 `datasource` 字段直接作为 if-else 的分支。
- **优点**：确定性强、可枚举、易调试。**缺点**：路由类别必须提前列全。

#### 10-b Semantic Routing（语义路由）
- **场景**：路由目标是**若干 prompt 模板**（例如"物理教师风格 prompt" vs "数学教师风格 prompt"），根据问题语义挑一个。
- **手段**：
  1. 离线：对每个候选 prompt 做一次 embedding；
  2. 在线：把 query embed → cosine 相似度 → argmax 选出模板。
- **优点**：不用调 LLM，一次向量点积就出结果，延迟极低。
- **场合**：类别边界模糊、可能新增的场景。

### Part 11 · Query Construction（结构化 query 构建）

**核心痛点**：用户说 *"找找最近发布、观看量 >10k 的关于 chat_history 的教程视频"*，其中包含**文本检索 + 元数据过滤**（日期范围、view_count 阈值）。纯向量检索做不到，得转换成结构化 DB query。

**手段**：定义 Pydantic schema，例如：

```python
class TutorialSearch(BaseModel):
    content_search: str            # 走 embedding 相似度
    title_search: str              # 走 title 关键词
    min_view_count: Optional[int]  # 元数据过滤
    earliest_publish_date: Optional[date]
    latest_publish_date: Optional[date]
```

然后 `structured_llm = llm.with_structured_output(TutorialSearch)`，LLM 把自然语言拆成结构化字段，交给数据库执行。

**扩展**：同一套思路可以做
- Text-to-SQL：LLM 生成 SQL；
- Self-Query Retriever：LLM 输出 `metadata_filter + query_text`；
- Knowledge Graph：LLM 输出 Cypher。

**参考**：<https://blog.langchain.dev/query-construction/>

> 本项目 C++ 版把这两节列入 **P2（占位不实现）**，因为它高度依赖具体业务 schema——但**"让 LLM 输出结构化 JSON 做决策"这套思想在 Router / CRAG / Self-RAG 里反复出现**，务必先理解。

---

## 4. Notebook 12–14：高级 Indexing 策略

**这三节是索引侧的"升级路径"——都比朴素 chunking + 单向量索引复杂，但都在不同场景显著提升检索质量。**

### Part 12 · Multi-representation Indexing（多表征索引）
- **痛点**：一个 chunk 只有一条 embedding，覆盖不了"标题、摘要、原文、图片、表格"多个观察角度。
- **思想**：对同一个 document 生成**多个表征**：
  - 让 LLM 生成一个**摘要**（summary），用摘要做 embedding；
  - 保留**原始 chunk** 用于最终生成时给 LLM 看；
  - 图片/表格用多模态模型另做一路 embedding；
  - 用 `MultiVectorRetriever` 关联：查中的是摘要向量，返回给 LLM 的是原文。
- **收益**：摘要比原文噪声少、语义更聚焦，检索精度上升；生成时又用完整原文避免信息丢失。
- **代价**：索引期多一次 LLM 调用（贵）；存储层要能一对多。
- **参考**：<https://arxiv.org/abs/2312.06648>（Proposition Indexing）；LangChain [MultiVectorRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)。

### Part 13 · RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）
- **论文**：<https://arxiv.org/pdf/2401.18059.pdf>
- **思想**：对 chunks 做**层次聚类 + 递归摘要**，构建一棵"摘要树"：
  ```
                     [root: 全部文档的高度摘要]
                     /                        \
              [cluster A summary]      [cluster B summary]
              /       |       \             /      \
           chunk    chunk   chunk       chunk   chunk
  ```
- 索引时把**所有节点**（叶子 chunk + 各级 summary）都写入向量库。
- 检索时同一查询能同时命中"细节级 chunk"和"高层 summary"，天然支持**跨粒度问答**（既能回答"某个函数返回什么"，又能回答"这个模块整体功能"）。
- **代价**：索引期需要跑 KMeans/GMM 聚类 + 多层 LLM 摘要，非常贵；但检索延迟不变。
- **适用**：长文档、需要多跳推理的问答（论文集合、法律条文、代码库）。

### Part 14 · ColBERT（Late Interaction）
- **论文**：ColBERT-v2；LangChain 通过 [RAGatouille](https://github.com/bclavie/ragatouille) 集成。
- **痛点**：一个 chunk 压成一条向量丢失 token 级信息。词很关键但被稀释。
- **思想**：**每个 token 都保留一条 embedding**，检索时用"晚交互"（late interaction）打分：

  $$ \text{score}(q, d) = \sum_{t_q \in q} \max_{t_d \in d} \text{sim}(t_q, t_d) $$

  即：对 query 每个 token，找 doc 里最像它的 token，取最大相似度；把所有 query token 的分累加。
- **收益**：精度显著高于稠密单向量，对术语/罕见词友好。
- **代价**：索引膨胀（每 token 一条 embedding，索引大小 × 数十倍）；检索需要专用引擎（PLAID / IVF）。
- **对 ObjectBox 用户的提示**：ObjectBox 目前只支持单向量+HNSW，若想用 ColBERT 需要换 backend（FAISS / RAGatouille）。

> 三节的选型直觉：
> - **想立刻提升召回而不重构索引** → 12（多表征，加个摘要向量）
> - **文档极长/需要多层次抽象** → 13（RAPTOR）
> - **对精度极度敏感、可接受重量级 backend** → 14（ColBERT）

---

## 5. Notebook 15–18：Active RAG（主动/自省 RAG）

**核心趋势**：RAG 不再是一次性 "retrieve → generate"，而是引入**评分、判断、循环、路由**——用 LangGraph 建立状态机，让系统能"发现自己检索/生成得不好 → 走另一条路"。

### Part 15 · Re-ranking（重排序）
- **场景**：第一步 retrieval 用 embedding 追求高**召回**（recall）——把可能相关的都捞回来（top 50）；然后用**更贵但更准**的模型对这 50 篇打分，取前 5 篇给 LLM。这一步叫 rerank，是"召回→精度"的转换器。
- **常见方案**：
  - **Cohere Rerank API**（`rerank-english-v3.0`）
  - **BGE-Reranker-v2-m3**（开源，SiliconFlow / HuggingFace 都有）
  - **RRF**（本节也演示了作为 rerank 的一种）
- **辅助技巧**：**Long-Context Reordering**——把最相关的 chunk 放在 prompt 首尾，缓解 [Lost in the Middle](https://arxiv.org/abs/2307.03172) 现象。
- **实用价值极高**：几乎每一个生产 RAG 系统都会加 rerank，通常是 P0 项。

### Part 16 · CRAG（Corrective RAG）
- **论文**：<https://arxiv.org/abs/2401.15884>
- **状态机**（LangGraph 实现）：
  ```mermaid
  flowchart LR
    Q[Query] --> R[Retrieve]
    R --> G[Grade Documents]
    G -->|全部相关| GEN[Generate]
    G -->|部分相关| REW[Rewrite Query] --> WEB[Web Search] --> GEN
    G -->|全部无关| REW
  ```
- **关键组件**：`document_grader`（用小 LLM 判断"这段和 query 相关吗？yes/no"）；`query_rewriter`；`web_search_tool`（Tavily/SerpAPI）作为兜底。
- **参考实现**：[langgraph_crag.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb)
- **收益**：私有知识库覆盖不到的问题不会瞎编，会自动去网上找。

### Part 17 · Self-RAG（自省 RAG）
- **论文**：<https://arxiv.org/abs/2310.11511>
- **思想**：训练/prompt LLM 输出**四种反思 token**：
  1. `Retrieve?` —— 需不需要检索（有的问题 LLM 自己就能答）
  2. `IsRel` —— 这段检索结果和问题相关吗
  3. `IsSup` —— 生成的答案有没有被检索段落支持
  4. `IsUse` —— 这个答案对用户是否有用
- **每一步都是 LLM 自评**，不满足就回到前面重来。相比 CRAG（只对"检索质量"评分），Self-RAG **连生成过程本身都自省**。
- **代价**：LLM 调用次数最多，token 开销最大；但对高准确度场景（医疗/法律）值得。
- **参考实现**：[langgraph_self_rag.ipynb](https://github.com/langchain-ai/langgraph/tree/main/examples/rag)

### Part 18 · Adaptive RAG
- **论文**：<https://arxiv.org/abs/2403.14403>
- **思想**：不是所有 query 都值得走复杂 pipeline。用一个**小分类器**（可以是 LLM）判断 query 复杂度：
  - **Simple** → 直接 LLM 回答（不检索）
  - **Single-step** → 常规 RAG
  - **Multi-step** → Decomposition + iterative RAG
- **本质是 Router × Pipeline 的组合器**——把前面所有 pipeline 都当积木，按 query 动态挑一条最省钱的路径。
- **和 CRAG/Self-RAG 关系**：Adaptive-RAG 是**入口路由**，CRAG/Self-RAG 是**执行内部反思**，两者可以叠加。

> 15–18 学习路径：
> - **必上生产** → 15 Re-ranking
> - **想让系统"检索不好会自救"** → 16 CRAG（比 Self-RAG 好实现得多）
> - **对准确度极致要求** → 17 Self-RAG
> - **需要给不同难度 query 不同待遇** → 18 Adaptive-RAG

---

## 6. 从入门到深入的推荐学习顺序

按"付出/收获比"排序，建议这样通读：

1. **Notebook 1–4** —— 一次跑通，理解 Index/Retrieval/Generation 三段（**入门线**）
2. **Notebook 5, 6, 9** —— Multi-Query / RAG-Fusion / HyDE，感受 query 改写的效果（**进阶线**）
3. **Notebook 15** —— 加 rerank，见识 recall→precision 的质变（**实用线**）
4. **Notebook 10-a** —— Logical Routing，理解"LLM 输出结构化 JSON 做决策"这个母题（**架构线**）
5. **Notebook 16** —— CRAG，理解 LangGraph 状态机式 RAG（**主动线**）
6. **Notebook 7, 8** —— Decomposition / Step-Back（**复杂问答线**）
7. **Notebook 10-b, 11** —— Semantic Routing / Query Construction（**结构化线**）
8. **Notebook 12, 13** —— Multi-representation / RAPTOR（**索引升级线**）
9. **Notebook 14, 17, 18** —— ColBERT / Self-RAG / Adaptive-RAG（**深水区**）

- **本文**：讲 Python 原课程的**原理和 what/why**；

---

## 7. 关键参考清单

**论文（值得精读）**
- HyDE：<https://arxiv.org/abs/2212.10496>
- Step-Back Prompting：<https://arxiv.org/abs/2310.06117>
- Self-RAG：<https://arxiv.org/abs/2310.11511>
- CRAG：<https://arxiv.org/abs/2401.15884>
- RAPTOR：<https://arxiv.org/abs/2401.18059>
- Adaptive-RAG：<https://arxiv.org/abs/2403.14403>
- Proposition Indexing（多表征）：<https://arxiv.org/abs/2312.06648>
- Lost in the Middle（长上下文）：<https://arxiv.org/abs/2307.03172>

**博客/工程实践**
- RAG-Fusion 原文：<https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1>
- LangChain Query Construction：<https://blog.langchain.dev/query-construction/>
- Multi-Modal RAG：<https://blog.langchain.dev/semi-structured-multi-modal-rag/>
- LangGraph RAG 示例集：<https://github.com/langchain-ai/langgraph/tree/main/examples/rag>

**视频**
- 官方合集：<https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x>（作者 Lance Martin 讲解，每节 5–10 分钟，强烈推荐配 notebook 一起看）
