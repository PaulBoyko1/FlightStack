cmake_minimum_required(VERSION 3.20)
project(flightstack_embedded LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_executable(flightstack_cpp_tests tests/test_math.cpp)
target_include_directories(flightstack_cpp_tests PRIVATE include)
target_compile_definitions(
    flightstack_cpp_tests
    PRIVATE FLIGHTSTACK_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
if(MSVC)
    target_compile_options(flightstack_cpp_tests PRIVATE /W4 /WX)
else()
    target_compile_options(flightstack_cpp_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
enable_testing()
add_test(NAME flightstack_cpp_tests COMMAND flightstack_cpp_tests)
