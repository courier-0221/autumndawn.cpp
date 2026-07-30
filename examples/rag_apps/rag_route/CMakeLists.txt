# rag_route —— v0.3 (Phase C) Logical Routing + Re-ranking 演示。
add_executable(rag_route main.cpp)

target_link_libraries(rag_route PRIVATE autumndawn_rag)

# 把示例配置拷到 build 目录。
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/rag_config.example.json
    ${CMAKE_CURRENT_BINARY_DIR}/rag_config.example.json COPYONLY
)
# 优先用本目录 rag_config.json；否则回退借用 rag_basic 已填好 key 的那份。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rag_config.json")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/rag_config.json
        ${CMAKE_CURRENT_BINARY_DIR}/rag_config.json COPYONLY
    )
endif()

# v0.3 专用多域语料：tech + cuisine。
if(EXISTS "${CMAKE_SOURCE_DIR}/data/corpus/corpus_route_tech.txt")
    configure_file(
        ${CMAKE_SOURCE_DIR}/data/corpus/corpus_route_tech.txt
        ${CMAKE_CURRENT_BINARY_DIR}/corpus_route_tech.txt COPYONLY
    )
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/data/corpus/corpus_route_cuisine.txt")
    configure_file(
        ${CMAKE_SOURCE_DIR}/data/corpus/corpus_route_cuisine.txt
        ${CMAKE_CURRENT_BINARY_DIR}/corpus_route_cuisine.txt COPYONLY
    )
endif()
