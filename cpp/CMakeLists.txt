cmake_minimum_required(VERSION 3.12)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME dolfinx_hdg)
project(${PROJECT_NAME})

find_package(DOLFINX REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)

# Target libraries
target_link_libraries(${PROJECT_NAME} INTERFACE dolfinx)

include(GNUInstallDirs)

install(FILES hello.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} COMPONENT Development)
