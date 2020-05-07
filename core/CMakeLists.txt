cmake_minimum_required(VERSION 2.6)
project(cpp_image_similarity CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-g")

add_library(illust_image_similarity illust_image_similarity.cpp)

target_include_directories(illust_image_similarity
	PUBLIC ${PROJECT_SOURCE_DIR}/
)

