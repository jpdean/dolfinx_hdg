cmake_minimum_required(VERSION 3.12)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME dolfinx_hdg)
project(${PROJECT_NAME})

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

find_package(DOLFINX REQUIRED)

add_library(${PROJECT_NAME}
        hello.h hello.cpp
        utils.h utils.cpp
        assembler.h
        petsc.h petsc.cpp
        assemble_matrix_impl.h
        assemble_vector_impl.h
        static_condensation_helpers.h)

# state that anybody linking to us needs to include the current source dir
# to find hello.h, while we don't.
target_include_directories(${PROJECT_NAME}
                           INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:include>)

# Target libraries
target_link_libraries(${PROJECT_NAME} PUBLIC dolfinx)

install(TARGETS ${PROJECT_NAME}
        DESTINATION lib
        EXPORT dolfinx_hdg_targets)
install(FILES
        hello.h
        utils.h
        assembler.h
        petsc.h
        assemble_matrix_impl.h
        assemble_vector_impl.h
        static_condensation_helpers.h
        DESTINATION include/${PROJECT_NAME})

# install the configuration targets
install(EXPORT dolfinx_hdg_targets
        FILE dolfinx_hdg_targets.cmake
        DESTINATION lib/cmake/dolfinx_hdg)

include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/dolfinx_hdgConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/example"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO)

# install the configuration file
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/dolfinx_hdgConfig.cmake
        DESTINATION lib/cmake/dolfinx_hdg)

# generate the export targets for the build tree
# needs to be after the install(TARGETS ) command
export(EXPORT dolfinx_hdg_targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/dolfinx_hdg_targets.cmake")
