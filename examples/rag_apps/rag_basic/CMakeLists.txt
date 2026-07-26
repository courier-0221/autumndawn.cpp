add_executable(rag_basic main.cpp)

target_link_libraries(rag_basic PRIVATE autumndawn_rag)

# 把示例配置和语料拷到 build 目录，方便 ./rag_basic 直接跑。
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/rag_config.example.json
    ${CMAKE_CURRENT_BINARY_DIR}/rag_config.example.json COPYONLY
)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rag_config.json")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/rag_config.json
        ${CMAKE_CURRENT_BINARY_DIR}/rag_config.json COPYONLY
    )
endif()

# 从统一的 data/corpus 目录拷贝一份中文示例语料。
if(EXISTS "${CMAKE_SOURCE_DIR}/data/corpus/corpus_sample.txt")
    configure_file(
        ${CMAKE_SOURCE_DIR}/data/corpus/corpus_sample.txt
        ${CMAKE_CURRENT_BINARY_DIR}/corpus_sample.txt COPYONLY
    )
endif()
