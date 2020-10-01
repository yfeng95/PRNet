cmake_minimum_required(VERSION 3.5)

project(fast_render VERSION 0.0.1)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wfatal-errors -O2")

include_directories(
    "src/"
)

file(GLOB all_fast_render_src
    "src/*.cpp"
)

add_library(fast_render SHARED ${all_fast_render_src})
