cmake_minimum_required(VERSION 2.8)
project(BembelPositTests LANGUAGES CXX)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (Bembel 1.0 REQUIRED NO_MODULE)
find_package (UNIVERSAL CONFIG REQUIRED)
find_package(EigenUniversalIntegration 0.1.0 CONFIG REQUIRED)

#   below. Then place the eigen3 directory obtained from downloading 
#   Eigen in Bembels root directory, where this CMake file sits. 
#   Afterwards, comment or remove the 
#
#   target_link_libraries(example_Quadrature.out Eigen3::Eigen)
#
#   lines in the CMake-Files of "\examples" and "\tests".

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" MARCHNATIVE)
CHECK_CXX_COMPILER_FLAG("-flto" FLTO)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "-O3 -march=native")

if (MARCHNATIVE)
message(STATUS "Found -march=native, adding option to builds of type >release< only")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native" CACHE STRING "compile flags" FORCE)
endif()
if (FLTO)
message(STATUS "Found -flto, adding option to builds of type >release< only")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto" CACHE STRING "compile flags" FORCE)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
message(STATUS "Found -fopenmp, adding option to builds of type >release< only")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
endif()

set( PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")


add_custom_target(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

add_custom_target(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )


add_subdirectory(examples)
# add_subdirectory(tests)
enable_testing()