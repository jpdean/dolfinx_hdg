cmake_minimum_required(VERSION 3.12)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME dolfinx_hdg)
project(${PROJECT_NAME})

add_subdirectory(cpp)
add_subdirectory(python)
