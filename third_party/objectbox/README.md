# objectbox

本目录是本仓库使用的 ObjectBox C/C++ 依赖，已从上游拆出并合并为单一 include 根，
供 `test/` 下的示例工程直接引用。CMake 里只需要一个 `OBX_DIR` 变量。

## 目录结构

```
objectbox/
├── include/
│   ├── objectbox.h / .hpp        # ObjectBox C/C++ API 头
│   ├── objectbox-sync.h / .hpp   # ObjectBox Sync（如未启用可忽略）
│   └── flatbuffers/              # header-only FlatBuffers C++ API
└── lib/
    └── libobjectbox.so           # 预编译动态库（Linux x64）
```

## 组件来源与版本

| 组件            | 版本 / 来源                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| `libobjectbox.so` + `include/objectbox*.h(pp)` | ObjectBox 官方 Linux x64 预编译包 `objectbox-5.3.2-linux-x64`（`5.3.2-2026-05-05`） |
| `include/flatbuffers/`                          | Google FlatBuffers C++ header-only 库，取自 `objectbox-c` 仓库的 `external/flatbuffers/` |

FlatBuffers 并不是 ObjectBox C/C++ API 的一部分，但 ObjectBox 代码生成器产出的
`*.obx.hpp/.cpp` 会 `#include "flatbuffers/flatbuffers.h"`，所以必须一并提供。
上游 `objectbox-c` 仓库的其它内容（示例、flatcc、doxygen 等）本仓库并不使用，
不再 vendored 进来。

## CMake 用法

```cmake
set(OBX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/<相对路径>/third_party/objectbox")

add_library(objectbox SHARED IMPORTED)
set_target_properties(objectbox PROPERTIES
    IMPORTED_LOCATION       "${OBX_DIR}/lib/libobjectbox.so"
    INTERFACE_INCLUDE_DIRECTORIES "${OBX_DIR}/include"
)

target_link_libraries(<your_target> PRIVATE objectbox)
```

运行时需要能找到 `libobjectbox.so`，可选方案：
- 把 `lib/` 加入 `LD_LIBRARY_PATH`
- 或在 CMake 里给可执行目标设置 `INSTALL_RPATH` / `BUILD_RPATH` 指向 `${OBX_DIR}/lib`
- 或把 `libobjectbox.so` 拷贝到可执行文件旁边

## 升级步骤

1. 从 <https://github.com/objectbox/objectbox-c/releases> 下载新版 Linux x64 包，
   覆盖 `include/objectbox*.h(pp)` 和 `lib/libobjectbox.so`。
2. 如需同步升级 FlatBuffers，从 <https://github.com/google/flatbuffers>
   或 objectbox-c 仓库 `external/flatbuffers/` 取新的头文件覆盖 `include/flatbuffers/`。
3. 更新本 README 的版本表。
