cmake_minimum_required(VERSION 3.12)

set(PROJECT_NAME dolfinx_hdg_pybind11)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

find_package(DOLFINX REQUIRED)
pybind11_add_module(cpp SHARED dolfinx_hdg/wrapper.cpp)
target_link_libraries(cpp PUBLIC pybind11::module)
target_link_libraries(cpp PUBLIC dolfinx)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
