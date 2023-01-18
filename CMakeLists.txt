cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(
  "frames"
  VERSION 0.1.0
  DESCRIPTION
    "A frames library for working with frames and coordinate system"
)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
)

find_package(Boost REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_compile_options(-fconstexpr-depth=4096 -Wall -Wextra)
add_library(${PROJECT_NAME} tests/src/tests.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

add_subdirectory(examples)
