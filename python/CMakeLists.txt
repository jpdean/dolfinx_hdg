set(PROJECT_NAME dolfinx_hdg_python)
project(${PROJECT_NAME})

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

pybind11_add_module(${PROJECT_NAME} SHARED wrapper.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module)
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx_hdg_cpp)
