cmake_minimum_required(VERSION 3.14)

project(embedding_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

get_filename_component(AUTUMNDAWN_THIRD_PARTY
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party" ABSOLUTE)

set(OBX_DIR "${AUTUMNDAWN_THIRD_PARTY}/objectbox")

add_library(objectbox SHARED IMPORTED)
set_target_properties(objectbox PROPERTIES
    IMPORTED_LOCATION "${OBX_DIR}/lib/libobjectbox.so"
    INTERFACE_INCLUDE_DIRECTORIES "${OBX_DIR}/include"
)

find_package(CURL REQUIRED)

add_executable(embedding_test
    main.cpp
    siliconflow_client.cpp
    config.cpp
    obx/document.obx.cpp
)

target_include_directories(embedding_test PRIVATE
    "${AUTUMNDAWN_THIRD_PARTY}"
    "${CMAKE_CURRENT_SOURCE_DIR}/obx"
)

target_link_libraries(embedding_test objectbox CURL::libcurl)

# 运行时把示例配置和语料拷贝到构建目录，方便直接在 build/ 里运行。
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/emb_config.example.json
    ${CMAKE_CURRENT_BINARY_DIR}/emb_config.example.json COPYONLY)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/emb_config.json")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/emb_config.json
        ${CMAKE_CURRENT_BINARY_DIR}/emb_config.json COPYONLY)
endif()
# 语料统一存放在仓库根目录的 data/corpus/ 下，便于其他 demo 共用和后续扩展。
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../../data/corpus/corpus_sample.txt
    ${CMAKE_CURRENT_BINARY_DIR}/corpus_sample.txt COPYONLY)
