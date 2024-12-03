cmake_minimum_required(VERSION 3.18)
project(nccl C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

file(GLOB src_files src/*.cc)
file(GLOB transport_files src/transport/*.cc)
file(GLOB misc_files src/misc/*.cc)
file(GLOB graph_files src/graph/*.cc)
file(GLOB coll_files src/collectives/*.cc src/collectives/device/*.cu)
file(GLOB ext_files ext-net/dummy/*.c)

add_executable(nccl main.cpp ${src_files} ${transport_files}
        ${misc_files} ${graph_files} ${coll_files} ${ext_files})
target_include_directories(nccl PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} src/ src/include)

message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

