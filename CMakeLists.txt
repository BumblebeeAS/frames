cmake_minimum_required(VERSION 3.16)
project(
  "frames"
  VERSION 0.2.0
  DESCRIPTION
  "A frames library for working with frames and coordinate system"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(include)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(bb_planner_msgs REQUIRED)
find_package(rclcpp REQUIRED)
# find_package(Boost REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} include/${PROJECT_NAME}/)
set_target_properties(${PROJECT_NAME}  PROPERTIES LINKER_LANGUAGE CXX)
add_executable(dest_pose_from_tf
src/dest_pose_from_tf.cpp)
ament_target_dependencies(dest_pose_from_tf
  PUBLIC
  bb_planner_msgs
  rclcpp
  geometry_msgs
  tf2_ros
  tf2
  tf2_geometry_msgs
)
ament_export_include_directories(include)
install(TARGETS dest_pose_from_tf DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

# ament_target_dependencies(${PROJECT_NAME} geometry_msgs )

# add_compile_options(-fconstexpr-depth=4096 -Wall -Wextra)
# add_executable(${PROJECT_NAME}_tests tests/src/tests.cpp)
# target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
# add_test(NAME ${PROJECT_NAME}_test
#          COMMAND tests)
# target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include/${PROJECT_NAME}/)

# ament_export_include_directories(include)
ament_auto_package()
