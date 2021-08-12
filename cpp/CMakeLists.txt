set(PROJECT_NAME dolfinx_hdg_cpp)
project(${PROJECT_NAME})

find_package(DOLFINX REQUIRED)

add_library(${PROJECT_NAME} SHARED hello.h hello.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx)

target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/")
