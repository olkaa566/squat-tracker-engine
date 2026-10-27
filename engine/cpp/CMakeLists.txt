cmake_minimum_required(VERSION 3.14)

project(SquatVisionEngine VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)


pybind11_add_module(cpp_engine 
    src/bindings.cpp
    src/biomechanics.cpp
    src/geometry.cpp
    src/matrix.cpp
    src/multilayer_perceptron.cpp
)

if(MSVC)
    target_compile_options(cpp_engine PRIVATE /W4)
else()
    target_compile_options(cpp_engine PRIVATE -Wall -Wextra)
endif()