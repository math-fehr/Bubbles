cmake_minimum_required (VERSION 3.8)

project(Bubbles C CXX CUDA)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
include_directories(glad)


file(GLOB_RECURSE sources_files src/*)
add_executable(${PROJECT_NAME} ${sources_files} glad/glad.c)

find_package(glfw3 3.2 REQUIRED)
target_link_libraries(Bubbles glfw)

set_property(TARGET Bubbles PROPERTY CUDA_STANDARD 14)

set_property(TARGET Bubbles PROPERTY CUDA_SEPARABLE_COMPILATION ON)





#   ____    _    ____ _   _ _____  __     ___    ____  ____
#  / ___|  / \  / ___| | | | ____| \ \   / / \  |  _ \/ ___|
# | |     / _ \| |   | |_| |  _|    \ \ / / _ \ | |_) \___ \
# | |___ / ___ \ |___|  _  | |___    \ V / ___ \|  _ < ___) |
#  \____/_/   \_\____|_| |_|_____|    \_/_/   \_\_| \_\____/


set(X_BASE_SIZE 1920 CACHE INTEGER "The start x size of the window (int)")
if(X_BASE_SIZE MATCHES "^[0-9]+$")
  add_definitions(-DX_BASE_SIZE=${X_BASE_SIZE})
else()
  message( SEND_ERROR "X_BASE_SIZE must be a unsigned integer")
endif()

set(Y_BASE_SIZE 1020 CACHE INTEGER "The start y size of the window (int)")
if(Y_BASE_SIZE MATCHES "^[0-9]+$")
  add_definitions(-DY_BASE_SIZE=${Y_BASE_SIZE})
else()
  message( SEND_ERROR "Y_BASE_SIZE must be a unsigned integer")
endif()

set(FULL_SCREEN false CACHE BOOL "Decide if window is fullscreen")
if(${FULL_SCREEN})
  add_definitions(-DFULL_SCREEN=1)
else()
  add_definitions(-DFULL_SCREEN=0)
endif()


set(VSYNC false CACHE BOOL "Enable V-Sync")
if(${VSYNC})
  add_definitions(-DVSYNC=1)
else()
  add_definitions(-DVSYNC=0)
endif()

set(THREADS_PER_BLOCK 128 CACHE INTEGER "The number of threads in a CUDA block")
if(THREADS_PER_BLOCK MATCHES "^[0-9]+$")
  add_definitions(-DTHREADS_PER_BLOCK=${THREADS_PER_BLOCK})
else()
  message( SEND_ERROR "THREADS_PER_BLOCK must be a unsigned integer")
endif()

set(NUM_REFL 2 CACHE INTEGER "The number of reflexions of a ray")
if(NUM_REFL MATCHES "^[0-9]+$")
  add_definitions(-DNUM_REFL=${NUM_REFL})
else()
  message( SEND_ERROR "NUM_REFL must be a unsigned integer")
endif()
