# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_frames_porting_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED frames_porting_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(frames_porting_FOUND FALSE)
  elseif(NOT frames_porting_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(frames_porting_FOUND FALSE)
  endif()
  return()
endif()
set(_frames_porting_CONFIG_INCLUDED TRUE)

# output package information
if(NOT frames_porting_FIND_QUIETLY)
  message(STATUS "Found frames_porting: 0.0.0 (${frames_porting_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'frames_porting' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${frames_porting_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(frames_porting_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "ament_cmake_export_include_directories-extras.cmake")
foreach(_extra ${_extras})
  include("${frames_porting_DIR}/${_extra}")
endforeach()
