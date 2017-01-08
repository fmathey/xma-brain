cmake_minimum_required(VERSION 3.6)

project(XMABrainTests)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

include_directories(
    ${PROJECT_SOURCE_DIR}/../xma-core/sources
    ${PROJECT_SOURCE_DIR}/sources
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/tests/main.cpp)