# C tests of the C interface of openmodelica_homotopy_capi: the smoke test and the example
# of the README. The build builds the static library with cargo first. From
# the repository root, whose CMakeLists.txt includes this directory:
#
#   cmake -S . -B build_c
#   cmake --build build_c
#   ctest --test-dir build_c --output-on-failure

cmake_minimum_required(VERSION 3.16)
project(homotopy_capi_c_tests C)

set(CRATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(WORKSPACE_DIR "${CRATE_DIR}/..")
if(DEFINED ENV{CARGO_TARGET_DIR})
  set(CARGO_TARGET_DIR "$ENV{CARGO_TARGET_DIR}")
else()
  set(CARGO_TARGET_DIR "${WORKSPACE_DIR}/target")
endif()
set(HOMOTOPY_CAPI_LIBRARY "${CARGO_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}openmodelica_homotopy_capi${CMAKE_STATIC_LIBRARY_SUFFIX}")

# cargo builds incrementally, so the library is always built and is only
# compiled again when the Rust code changed
find_program(CARGO cargo HINTS "$ENV{HOME}/.cargo/bin")
if(NOT CARGO)
  message(FATAL_ERROR "cargo not found")
endif()
add_custom_target(cargo_build_homotopy_capi
  COMMAND "${CARGO}" build -p openmodelica_homotopy_capi
  WORKING_DIRECTORY "${WORKSPACE_DIR}"
  BYPRODUCTS "${HOMOTOPY_CAPI_LIBRARY}"
  COMMENT "Building openmodelica_homotopy_capi with cargo"
  USES_TERMINAL)

# The library is linked like a C host links it. The system libraries are the
# ones `cargo rustc -p openmodelica_homotopy_capi --lib --crate-type staticlib -- --print
# native-static-libs` reports.
find_package(Threads REQUIRED)
add_library(openmodelica_homotopy_capi STATIC IMPORTED)
set_target_properties(openmodelica_homotopy_capi PROPERTIES
  IMPORTED_LOCATION "${HOMOTOPY_CAPI_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CRATE_DIR}/include")
set_property(TARGET openmodelica_homotopy_capi PROPERTY
  INTERFACE_LINK_LIBRARIES Threads::Threads ${CMAKE_DL_LIBS} m)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Werror)
enable_testing()

add_executable(capi_smoke_test capi_smoke_test.c)
target_link_libraries(capi_smoke_test PRIVATE openmodelica_homotopy_capi)
add_dependencies(capi_smoke_test cargo_build_homotopy_capi)
add_test(NAME capi_smoke_test COMMAND capi_smoke_test)

# The C example of the README is compiled and run as it is, so it stays valid
set(README "${WORKSPACE_DIR}/README.md")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${README}")
file(READ "${README}" readme)
string(REGEX MATCH "```c\n([^`]*)```" readme_example "${readme}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "No C example (```c block) in ${README}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/readme_example.c" "${CMAKE_MATCH_1}")
add_executable(readme_example "${CMAKE_CURRENT_BINARY_DIR}/readme_example.c")
target_link_libraries(readme_example PRIVATE openmodelica_homotopy_capi)
add_dependencies(readme_example cargo_build_homotopy_capi)
add_test(NAME readme_example COMMAND readme_example)
