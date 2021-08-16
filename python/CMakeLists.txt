cmake_minimum_required(VERSION 3.12)

set(PROJECT_NAME dolfinx_hdg_pybind11)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

find_package(DOLFINX REQUIRED)
find_package(dolfinx_hdg REQUIRED)
pybind11_add_module(cpp SHARED dolfinx_hdg/wrapper.cpp)
target_link_libraries(cpp PUBLIC pybind11::module)
target_link_libraries(cpp PUBLIC dolfinx)
target_link_libraries(cpp PUBLIC dolfinx_hdg)

# Find petsc4py through python
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_NOT_FOUND
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
