cmake_minimum_required(VERSION 3.12)

set(PROJECT_NAME dolfinx_hdg)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

find_package(DOLFINX REQUIRED)
pybind11_add_module(${PROJECT_NAME} SHARED wrapper.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC pybind11::module)
target_link_libraries(${PROJECT_NAME} PUBLIC dolfinx)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
