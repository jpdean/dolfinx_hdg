cmake_minimum_required(VERSION 3.12)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME dolfinx_hdg)
project(${PROJECT_NAME})

find_package(DOLFINX REQUIRED)

add_library(${PROJECT_NAME} hello.h hello.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx)

# Executable
add_executable(demo main.cpp)

target_link_libraries(demo ${PROJECT_NAME})
