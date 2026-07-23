# ObjectBox C++ 快速上手

本文以 `quickstart/` 示例为主线，逐步讲解 ObjectBox C++ 的基本用法；
文末附录整理了 Schema 语法、Box API、常见问题等通用参考内容。

> 依赖的下载、目录组成与升级方式见 `third_party/objectbox/README.md`，本文不重复。

---

## 示例说明

这个例子演示 ObjectBox 最核心的写入/读取流程，并验证数据的持久化：

1. **写入阶段**：打开数据库，写入 3 条 `Task` 记录，然后关闭 Store；
2. **读取阶段**：重新打开同一个数据库目录，先按 ID 逐条读取，再用 `getAll()` 读出全部记录。

数据在 Store 关闭后仍然保留在磁盘上（`objectbox-db/` 目录），第二次打开时可以完整读回。

---

## 项目结构

```
quickstart/
├── CMakeLists.txt         ← 构建配置（引用 third_party/objectbox 预编译库）
├── main.cpp               ← 业务逻辑（writeData / readData 两阶段）
└── obx/                   ← ObjectBox schema 目录
    ├── task.fbs               ← 手写：FlatBuffers Schema（实体定义）
    ├── task.obx.hpp           ← Generator 生成的实体头文件
    ├── task.obx.cpp           ← Generator 生成的实体实现文件
    ├── objectbox-model.h      ← Generator 生成的数据库模型注册
    └── objectbox-model.json   ← 模型元数据（含 ID/UID，不可删除）
```

业务代码（`main.cpp`）直接 `#include "task.obx.hpp"` / `#include "objectbox-model.h"` 即可，
CMake 已将 `obx/` 加入源文件列表与 include 搜索路径。

---

## 依赖引入

ObjectBox 依赖（预编译库 + FlatBuffers 头文件）已在仓库 `third_party/objectbox` 下统一整理，
`include/` 同时提供 `objectbox*.h(pp)` 与 `flatbuffers/`，CMake 里只需要一个 `OBX_DIR` 变量：

```cmake
get_filename_component(OBX_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/objectbox" ABSOLUTE)

add_library(objectbox SHARED IMPORTED)
set_target_properties(objectbox PROPERTIES
    IMPORTED_LOCATION "${OBX_DIR}/lib/libobjectbox.so"
    INTERFACE_INCLUDE_DIRECTORIES "${OBX_DIR}/include"
)

target_link_libraries(obxdemo objectbox)
```

下载、升级等细节见 `third_party/objectbox/README.md`。

---

## 第一步：定义实体（.fbs Schema）

`obx/task.fbs` 是实体的数据定义文件，使用 FlatBuffers IDL 语法：

```flatbuffers
table Task {
    id: ulong;       // ObjectBox 要求：第一个字段 id，类型 ulong
    text: string;
}

root_type Task;
```

- 每个实体必须有 `id: ulong` 字段，ObjectBox 用它作为主键。
- `root_type` 指定序列化的根类型。
- 生成代码后得到 `Task` struct（C++ 端）。

---

## 第二步：生成绑定代码

使用 [ObjectBox Generator](https://github.com/objectbox/objectbox-generator) 根据 `.fbs` 生成绑定。
在 `test/objectbox/quickstart/` 目录下执行：

```bash
objectbox-generator -cpp obx/task.fbs
```

Generator 默认把产物写到 `.fbs` 所在目录（即 `obx/`），无需 `-out` 参数。

生成文件（均位于 `obx/`）：

| 文件 | 说明 |
|------|------|
| `task.obx.hpp` | Task struct 定义 + 元信息（`_OBX_MetaInfo`） |
| `task.obx.cpp` | FlatBuffers 序列化/反序列化实现（需加入 `add_executable()`） |
| `objectbox-model.h` | 提供 `create_obx_model()`，注册所有实体 |
| `objectbox-model.json` | 模型 ID/UID 状态文件（**必须提交版本控制，不可删除、不可手改**） |

> 已生成的文件已在工作区中，无需重新生成。

---

## 第三步：打开数据库 Store

```cpp
#define OBX_CPP_FILE   // 在且仅在一个 .cpp 文件中定义，激活 objectbox.hpp 的实现代码
#include "task.obx.hpp"
#include "objectbox-model.h"

static obx::Options makeOptions() {
    obx::Options options;
    options.model(create_obx_model());   // 注册实体模型
    options.directory("objectbox-db");   // 数据库存储目录（相对路径）
    return options;
}

// 打开 Store
obx::Store store(makeOptions());
```

- `obx::Store` 是数据库实例，析构时自动关闭。
- `directory` 指定数据文件存放位置，程序首次运行时自动创建。
- **同一时刻只能有一个 Store 实例打开同一个目录。**

---

## 第四步：写入数据

对应 `main.cpp` 中的 `writeData()`：

```cpp
obx::Box<Task> box(store);   // 获取 Task 类型的 Box

Task task;
task.id = 0;         // id = 0 表示新对象，ObjectBox 自动分配 ID
task.text = "Buy milk";

obx_id newId = box.put(task);   // 写入，返回分配的 ID
std::cout << "Inserted id=" << newId << "\n";
```

示例中循环写入 3 条记录后，`writeData()` 返回，`store` 析构，数据库关闭——
数据已持久化到 `objectbox-db/` 目录。

---

## 第五步：读取数据

对应 `main.cpp` 中的 `readData()`：重新打开 Store，按 ID 读取 + 读取全部。

```cpp
// 按 ID 读取单条
auto taskPtr = box.get(newId);   // 返回 std::unique_ptr<Task>
if (taskPtr) {
    std::cout << "text=" << taskPtr->text << "\n";
}

// 读取全部
auto all = box.getAll();   // 返回 std::vector<std::unique_ptr<Task>>
for (const auto& t : all) {
    std::cout << "id=" << t->id << "  text=" << t->text << "\n";
}
```

---

## 第六步：编译与运行

```bash
cd test/objectbox/quickstart
mkdir -p build && cd build
cmake ..
cmake --build .

# 设置运行时库路径（指向仓库内的 third_party/objectbox/lib）
export LD_LIBRARY_PATH=<仓库根>/third_party/objectbox/lib:$LD_LIBRARY_PATH

# 运行
./obxdemo
```

预期输出：

```
[write] put id=1  text="Buy milk"
[write] put id=2  text="Walk the dog"
[write] put id=3  text="Learn ObjectBox"
[write] store closed

[read] reopened store, reading 3 records:
[read] id=1  text="Buy milk"
[read] id=2  text="Walk the dog"
[read] id=3  text="Learn ObjectBox"

[read] getAll() returned 3 records total
  id=1  text="Buy milk"
  id=2  text="Walk the dog"
  id=3  text="Learn ObjectBox"
```

---
---

# 附录

## 附录 A：Schema 与代码生成参考

### Schema 语法（.fbs）

```flatbuffers
table EntityName {
    id: ulong;          // 必须：主键，类型固定为 ulong
    name: string;
    age: int;
    score: float;
    active: bool;
}
root_type EntityName;
```

字段类型对照：

| FlatBuffers 类型 | C++ 类型 |
|-----------------|---------|
| `ulong` | `uint64_t` / `obx_id` |
| `long` | `int64_t` |
| `int` | `int32_t` |
| `string` | `std::string` |
| `bool` | `bool` |
| `float` | `float` |
| `double` | `double` |
| `[float]` | `std::vector<float>` |
| `[byte]` | `std::vector<uint8_t>` |

### ObjectBox 注解

用 `/// objectbox:` 前缀添加：

```flatbuffers
table Task {
    id: ulong;

    /// objectbox:date
    date_created: long;           // 日期类型（毫秒时间戳）

    /// objectbox:index
    category: string;              // 建立索引
}
```

| 注解 | 说明 |
|------|------|
| `objectbox:id` | 标记为主键（默认第一个 ulong 字段） |
| `objectbox:date` | 日期字段（long，毫秒时间戳） |
| `objectbox:index` | 常规索引 |
| `objectbox:unique` | 唯一约束 |
| `objectbox:index=hnsw` | HNSW 向量索引（向量搜索） |
| `objectbox:hnsw-dimensions=N` | 向量维度（HNSW 必填） |
| `objectbox:hnsw-distance-type=...` | 距离类型：`Euclidean`/`Cosine`/`DotProduct`/`Geo` |

向量搜索示例：

```flatbuffers
table City {
    id: ulong;
    name: string;

    /// objectbox: index=hnsw, hnsw-dimensions=2
    /// objectbox: hnsw-distance-type=Geo
    location: [float];   // 2D 地理位置向量
}
```

### Generator 命令行

```bash
objectbox-generator -cpp task.fbs      # C++17 绑定（默认）
objectbox-generator -cpp11 task.fbs    # C++11 兼容绑定
objectbox-generator -c task.fbs        # 纯 C 接口
```

Generator 安装：从 <https://github.com/objectbox/objectbox-generator/releases> 下载，
`chmod +x` 后放到 `PATH` 中即可。

### `objectbox-model.json` 规则

- 每个实体和属性都有全局唯一的 `id:uid` 对。
- **绝对不能手动编辑或删除**，否则 ObjectBox 无法识别已有数据的 schema，可能导致数据损坏或打开数据库失败。
- 必须提交到 Git 仓库。

---

## 附录 B：Box API 速查与注意事项

### 常用 Box API

| 方法 | 说明 |
|------|------|
| `box.put(obj)` | 插入或更新（upsert）一个对象，返回 `obx_id` |
| `box.get(id)` | 按 ID 读取，返回 `unique_ptr<T>`（不存在返回 nullptr） |
| `box.getAll()` | 读取所有对象 |
| `box.remove(id)` | 按 ID 删除 |
| `box.removeAll()` | 删除所有对象 |
| `box.count()` | 返回对象总数 |
| `box.query(...)` | 构建查询（支持条件过滤） |

### 注意事项

- `#define OBX_CPP_FILE` 只能在**一个** `.cpp` 文件中定义（通常是 `main.cpp`），用于实例化 `objectbox.hpp` 中的模板实现。
- `objectbox-model.json` 记录了实体和属性的 ID/UID 映射，**必须纳入版本控制**，删除会导致数据库 schema 混乱。
- Store 不是全局单例；多线程场景可以多个线程共享同一个 `Store` 实例，每个线程各自创建 `Box`。

---

## 附录 C：常见问题

### Q: 一定要依赖 FlatBuffers 吗？

**是的。** Generator 生成的 `task.obx.hpp`/`task.obx.cpp` 会**无条件** `#include "flatbuffers/flatbuffers.h"`
并直接使用 `flatbuffers::FlatBufferBuilder` 做序列化，这不是可选依赖。

ObjectBox 官方预编译包本身不含 FlatBuffers 头文件；本仓库已把 FlatBuffers 头文件合并到
`third_party/objectbox/include/flatbuffers/`，与 `objectbox*.h(pp)` 共用一个 include 根，
CMake 里只需引入 `third_party/objectbox/include` 一个目录。

### Q: 运行时报 `error while loading shared libraries: libobjectbox.so`

编译正常但运行时找不到动态库，需要让系统能找到 `libobjectbox.so`：

```bash
# 临时方式（推荐开发调试）
export LD_LIBRARY_PATH=<仓库根>/third_party/objectbox/lib:$LD_LIBRARY_PATH

# 或在 CMake 中给可执行目标设置 BUILD_RPATH 指向该 lib 目录
```

### Q: 头文件找不到 `flatbuffers/flatbuffers.h`

确认 include 路径指向的是 `third_party/objectbox/include`（而不是其中的某个子目录），
FlatBuffers 头文件位于该目录下的 `flatbuffers/` 子目录中。

---

## 相关链接

- 官方文档：<https://cpp.objectbox.io/>
- GitHub：<https://github.com/objectbox/objectbox-c>
- API 参考：<https://objectbox.io/docfiles/c/current/>
- ObjectBox Generator：<https://github.com/objectbox/objectbox-generator>
