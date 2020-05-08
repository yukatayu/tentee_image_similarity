cmake_minimum_required(VERSION 2.6)
project(cpp_image_similarity CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-g")

add_subdirectory(core)

# Apps
add_executable(run_test app/test.cpp)
target_link_libraries(run_test illust_image_similarity)

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(run_test PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(run_test ${OpenCV_LIBS})
endif()

# Boost
find_package(Boost REQUIRED
             COMPONENTS system filesystem)
if(Boost_FOUND)
	target_include_directories(run_test PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(run_test Boost::system Boost::filesystem)
endif()

