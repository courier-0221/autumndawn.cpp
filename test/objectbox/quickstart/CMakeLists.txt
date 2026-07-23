cmake_minimum_required(VERSION 3.14)

project(obxdemo)

set(CMAKE_CXX_STANDARD 17)

get_filename_component(OBX_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/objectbox" ABSOLUTE)

add_library(objectbox SHARED IMPORTED)
set_target_properties(objectbox PROPERTIES
    IMPORTED_LOCATION "${OBX_DIR}/lib/libobjectbox.so"
    INTERFACE_INCLUDE_DIRECTORIES "${OBX_DIR}/include"
)

add_executable(obxdemo
    main.cpp
    obx/task.obx.cpp
)

target_include_directories(obxdemo PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/obx"
)

target_link_libraries(obxdemo objectbox)