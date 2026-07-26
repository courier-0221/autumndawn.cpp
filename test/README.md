# test/

存放 `autumndawn_rag` 各模块的**自动化单元测试**（GoogleTest / Catch2 等），
用 `ctest` 驱动，目标是可以在 CI 中无人值守跑通。

跟 `examples/` 的区别：

- `examples/`：手动运行的演示程序（`ingest` / `ask` 交互式 CLI、ObjectBox quickstart 等），
  用来学习接口用法、验证端到端链路，需要人工看输出、可能要填真实 API key。
- `test/`：断言驱动的单元测试，不依赖外部网络/API，跑起来应该是秒级、可重复、绿灯/红灯分明。

## 当前状态

骨架阶段，尚未引入具体测试框架。接入时的大致步骤（以 GoogleTest 为例）：

1. 在 `test/CMakeLists.txt` 里用 `FetchContent` 拉取 GoogleTest（或改用 vendored 到 `third_party/`）；
2. 每个模块一个 `xxx_test.cpp`，`target_link_libraries(... autumndawn_rag gtest_main)`；
3. `include(GoogleTest)` + `gtest_discover_tests(...)`，让 `ctest` 能发现所有用例；
4. 根目录 `cmake -DAUTUMNDAWN_BUILD_TESTS=ON ..` 打开测试构建。

## 运行（框架接入后）

```bash
cmake -DAUTUMNDAWN_BUILD_TESTS=ON -B build
cmake --build build
ctest --test-dir build --output-on-failure
```
