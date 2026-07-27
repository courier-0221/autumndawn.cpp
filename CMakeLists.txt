cmake_minimum_required(VERSION 3.14)
project(autumndawn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- Third-party ----------------------------------------------------------
set(AUTUMNDAWN_THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(OBX_DIR "${AUTUMNDAWN_THIRD_PARTY}/objectbox")

add_library(objectbox SHARED IMPORTED GLOBAL)
set_target_properties(objectbox PROPERTIES
    IMPORTED_LOCATION "${OBX_DIR}/lib/libobjectbox.so"
    INTERFACE_INCLUDE_DIRECTORIES "${OBX_DIR}/include"
)

find_package(CURL REQUIRED)

# ---- libautumndawn_rag ----------------------------------------------------
add_library(autumndawn_rag STATIC
    # inference
    src/inference/http_client.cpp
    src/inference/cloud/siliconflow_embedding_client.cpp
    src/inference/cloud/deepseek_chat_client.cpp
    src/inference/cloud/siliconflow_rerank_client.cpp

    # storage (ObjectBox 后端 + 生成绑定；新后端在 src/storage/<backend>/ 下平级新增)
    src/storage/objectbox/document.obx.cpp
    src/storage/objectbox/objectbox_store.cpp

    # retrieval
    src/retrieval/vector_store_retriever.cpp

    # core
    src/text_splitter.cpp
    src/prompt_template.cpp
    src/generator.cpp
    src/pipeline.cpp
    src/config.cpp
    src/model_factory.cpp
    src/store_factory.cpp
)

target_include_directories(autumndawn_rag
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${AUTUMNDAWN_THIRD_PARTY}>            # nlohmann/json.hpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/storage/objectbox      # document.obx.hpp / objectbox-model.h
)

target_link_libraries(autumndawn_rag
    PUBLIC
        objectbox
    PRIVATE
        CURL::libcurl
)

target_compile_options(autumndawn_rag PRIVATE -Wall -Wextra -Wpedantic)

# ---- examples ---------------------------------------------------------
# 手动运行的演示程序（交互式 CLI、ObjectBox quickstart 等）。
add_subdirectory(examples/rag_apps/rag_basic)

# ---- test ---------------------------------------------------------------
# 自动化单元测试骨架，默认关闭；接入具体测试框架后用
#   cmake -DAUTUMNDAWN_BUILD_TESTS=ON ..
# 打开。
option(AUTUMNDAWN_BUILD_TESTS "Build autumndawn_rag unit tests" OFF)
if(AUTUMNDAWN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
