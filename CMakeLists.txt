cmake_minimum_required(VERSION 3.8)
project(frames)
include_directories(include)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(BUILD_TESTING "Build tests" ON)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)

install(TARGETS ${PROJECT_NAME}
  EXPORT "export_${PROJECT_NAME}"
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  INCLUDES DESTINATION include
)

# find dependencies
find_package(ament_cmake REQUIRED)

# Debug
ament_export_targets("export_${PROJECT_NAME}")
install(
  DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2 REQUIRED)

  set(PROJECT_LIBRARIES
    geometry_msgs
    tf2
  )
  add_executable(test_frames tests/src/tests.cpp)
  ament_target_dependencies(test_frames ${PROJECT_LIBRARIES})
  target_compile_features(test_frames PRIVATE cxx_std_17)
endif()

ament_package()