# rag_query_translate —— v0.2 (Phase B) Query Translation 策略对比演示。
add_executable(rag_query_translate main.cpp)

target_link_libraries(rag_query_translate PRIVATE autumndawn_rag)

# 把示例配置和语料拷到 build 目录，方便 ./rag_query_translate 直接跑。
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/rag_config.example.json
    ${CMAKE_CURRENT_BINARY_DIR}/rag_config.example.json COPYONLY
)
# 优先用本目录的 rag_config.json。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rag_config.json")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/rag_config.json
        ${CMAKE_CURRENT_BINARY_DIR}/rag_config.json COPYONLY
    )
endif()

# 从统一的 data/corpus 目录拷贝示例语料。
if(EXISTS "${CMAKE_SOURCE_DIR}/data/corpus/corpus_sample.txt")
    configure_file(
        ${CMAKE_SOURCE_DIR}/data/corpus/corpus_sample.txt
        ${CMAKE_CURRENT_BINARY_DIR}/corpus_sample.txt COPYONLY
    )
endif()
# v0.2 专用语料：为 query translation 策略对比设计（同义改写簇 + 干扰项 + 多面向答案）。
if(EXISTS "${CMAKE_SOURCE_DIR}/data/corpus/corpus_query_translate.txt")
    configure_file(
        ${CMAKE_SOURCE_DIR}/data/corpus/corpus_query_translate.txt
        ${CMAKE_CURRENT_BINARY_DIR}/corpus_query_translate.txt COPYONLY
    )
endif()
