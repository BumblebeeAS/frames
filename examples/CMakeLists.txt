cmake_minimum_required(VERSION 3.14)
project(frames_examples)

add_executable(example1 src/1_change_position_coordinate_system.cpp)
target_compile_features(example1 PRIVATE cxx_std_17)

add_executable(example2 src/2_change_orientation_coordinate_system.cpp)
target_compile_features(example2 PRIVATE cxx_std_17)

add_executable(example3 src/3_customise_origin.cpp)
target_compile_features(example3 PRIVATE cxx_std_17)