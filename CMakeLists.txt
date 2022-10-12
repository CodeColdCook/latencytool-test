cmake_minimum_required(VERSION 3.5)

# ##############################################################################
# Metadata #
# ##############################################################################

# Read the package manifest.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/package.xml" package_xml_str)

# Extract project name.
if(NOT package_xml_str MATCHES "<name>([A-Za-z0-9_]+)</name>")
  message(
    FATAL_ERROR "Could not parse project name from package manifest (aborting)")
else()
  set(extracted_name ${CMAKE_MATCH_1})
endif()

# Extract project version.
if(NOT package_xml_str MATCHES "<version>([0-9]+.[0-9]+.[0-9]+)</version>")
  message(
    FATAL_ERROR
      "Could not parse project version from package manifest (aborting)")
else()
  # At this point we either have a proper version string, or we've errored out
  # with a FATAL_ERROR above. So assume CMAKE_MATCH_1 contains our package's
  # version.
  set(extracted_version ${CMAKE_MATCH_1})
endif()

# ##############################################################################
# CMake Project #
# ##############################################################################

project(
  ${extracted_name}
  VERSION ${extracted_version}
  LANGUAGES C CXX)

# =============#
# Dependencies #
# =============#

# ----------------------#
# Third-party libraries #
# ----------------------#

find_package(Threads REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(
  Qt5
  COMPONENTS Core Gui Widgets
  REQUIRED)

find_package(jetson-utils REQUIRED)
find_package(jetson-inference REQUIRED)

find_package(CUDA REQUIRED)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# -----------------#
# Zetton libraries #
# -----------------#

# =========#
# Settings #
# =========#

# shared libraries
if(NOT DEFINED BUILD_SHARED_LIBS)
  message(STATUS "${PROJECT_NAME}: Building dynamically-linked binaries")
  option(BUILD_SHARED_LIBS "Build dynamically-linked binaries" ON)
  set(BUILD_SHARED_LIBS ON)
endif()

# build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  message(STATUS "${PROJECT_NAME}: Defaulting build type to RelWithDebInfo")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# win32
if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# global compilations
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_definitions(-O2)

# instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

# useful macros
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/util.cmake)

# ##############################################################################
# Build ##
# ##############################################################################

# ==============#
# Build targets #
# ==============#

include(GenerateExportHeader)

# ---------------#
# Library target #
# ---------------#

# find all source files
file(GLOB_RECURSE src_files ${PROJECT_SOURCE_DIR}/src/common.c)

# common library
zetton_cc_library(
  NAME
  ${PROJECT_NAME}
  SRCS
  ${src_files}
  INCLUDES
  ${OpenCV_INCLUDE_DIRS}
  DEPS
  Threads::Threads
  ${OpenCV_LIBS})

# ------------------#
# Executable targets #
# ------------------#

function(add_latency_tester)
  cmake_parse_arguments(LATENCY_EXEC "" "NAME" "SRCS;INCLUDES;DEPS" ${ARGN})
  add_executable(${LATENCY_EXEC_NAME} ${LATENCY_EXEC_SRCS})
  target_include_directories(
    ${LATENCY_EXEC_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>"
           ${LATENCY_EXEC_INCLUDES})
  target_link_libraries(${LATENCY_EXEC_NAME} PUBLIC ${LATENCY_EXEC_DEPS})
endfunction()

function(add_latency_jetson_tester)
  cmake_parse_arguments(LATENCY_EXEC "" "NAME" "SRCS;INCLUDES;DEPS" ${ARGN})
  add_executable(${LATENCY_EXEC_NAME} ${LATENCY_EXEC_SRCS})
  target_include_directories(
    ${LATENCY_EXEC_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>"
           ${LATENCY_EXEC_INCLUDES}
           ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${LATENCY_EXEC_NAME} PUBLIC ${LATENCY_EXEC_DEPS} jetson-inference)
endfunction()

add_latency_tester(
  NAME
  latency_cv_term
  SRCS
  ${PROJECT_SOURCE_DIR}/src/frontend_term.c
  ${PROJECT_SOURCE_DIR}/src/backend_opencv.cpp
  INCLUDES
  ${OpenCV_INCLUDE_DIRS}
  DEPS
  ${PROJECT_NAME}
  ${OpenCV_LIBS})

add_latency_tester(
  NAME
  latency_cv_fb
  SRCS
  ${PROJECT_SOURCE_DIR}/src/frontend_fb.c
  ${PROJECT_SOURCE_DIR}/src/backend_opencv.cpp
  INCLUDES
  ${OpenCV_INCLUDE_DIRS}
  DEPS
  ${PROJECT_NAME}
  ${OpenCV_LIBS})

add_latency_tester(
  NAME
  latency_cv_qt
  SRCS
  ${PROJECT_SOURCE_DIR}/src/frontend_qt.cpp
  ${PROJECT_SOURCE_DIR}/src/backend_opencv.cpp
  INCLUDES
  ${OpenCV_INCLUDE_DIRS}
  ${Qt5Core_INCLUDE_DIRS}
  ${Qt5Gui_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
  DEPS
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  Qt5::Core
  Qt5::Gui
  Qt5::Widgets)

add_latency_jetson_tester(
  NAME
  latency_cuda_term
  SRCS
  ${PROJECT_SOURCE_DIR}/src/frontend_term.c
  ${PROJECT_SOURCE_DIR}/src/backend_cuda.cpp
  ${PROJECT_SOURCE_DIR}/src/image_converter.cpp
  INCLUDES
  ${OpenCV_INCLUDE_DIRS}
  DEPS
  ${PROJECT_NAME}
  ${OpenCV_LIBS})

# -------------#
# Test targets #
# -------------#
