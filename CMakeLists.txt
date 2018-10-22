cmake_minimum_required (VERSION 3.8)

project(Bubbles C CXX CUDA)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
include_directories(glad)


file(GLOB_RECURSE sources_files src/*)
add_executable(${PROJECT_NAME} ${sources_files} glad/glad.c)

find_package(glfw3 3.2 REQUIRED)
target_link_libraries(Bubbles glfw)

set_property(TARGET Bubbles PROPERTY CUDA_STANDARD 14)

set_property(TARGET Bubbles
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

