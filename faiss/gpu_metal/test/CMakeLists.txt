# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Metal backend tests. Only included when BUILD_TESTING and FAISS_ENABLE_METAL.

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(AccMetalIndexFlat AccMetalIndexFlat.mm)
target_link_libraries(AccMetalIndexFlat PRIVATE
    faiss_metal
    faiss
    GTest::gtest_main
)
target_include_directories(AccMetalIndexFlat PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
set_target_properties(AccMetalIndexFlat PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
gtest_discover_tests(AccMetalIndexFlat)

add_executable(AccMetalIndexIVFFlat AccMetalIndexIVFFlat.mm)
target_link_libraries(AccMetalIndexIVFFlat PRIVATE
    faiss_metal
    faiss
    GTest::gtest_main
)
target_include_directories(AccMetalIndexIVFFlat PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
set_target_properties(AccMetalIndexIVFFlat PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
gtest_discover_tests(AccMetalIndexIVFFlat)

add_executable(AccMetalIndexIVFSQ AccMetalIndexIVFSQ.mm)
target_link_libraries(AccMetalIndexIVFSQ PRIVATE
    faiss_metal
    faiss
    GTest::gtest_main
)
target_include_directories(AccMetalIndexIVFSQ PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
set_target_properties(AccMetalIndexIVFSQ PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
gtest_discover_tests(AccMetalIndexIVFSQ)

add_executable(AccMetalIndexIVFPQ AccMetalIndexIVFPQ.mm)
target_link_libraries(AccMetalIndexIVFPQ PRIVATE
    faiss_metal
    faiss
    GTest::gtest_main
)
target_include_directories(AccMetalIndexIVFPQ PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
set_target_properties(AccMetalIndexIVFPQ PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
gtest_discover_tests(AccMetalIndexIVFPQ)

add_executable(AccMetalIndexBinaryFlat AccMetalIndexBinaryFlat.mm)
target_link_libraries(AccMetalIndexBinaryFlat PRIVATE
    faiss_metal
    faiss
    GTest::gtest_main
)
target_include_directories(AccMetalIndexBinaryFlat PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
set_target_properties(AccMetalIndexBinaryFlat PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
gtest_discover_tests(AccMetalIndexBinaryFlat)

add_executable(BenchMetalFlatVsCpu BenchMetalVsCpu.mm)
target_link_libraries(BenchMetalFlatVsCpu PRIVATE faiss_metal faiss)
target_include_directories(BenchMetalFlatVsCpu PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
set_target_properties(BenchMetalFlatVsCpu PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)

add_executable(BenchMetalIvfVsCpu BenchMetalIvfVsCpu.mm)
target_link_libraries(BenchMetalIvfVsCpu PRIVATE faiss_metal faiss)
target_include_directories(BenchMetalIvfVsCpu PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(BenchMetalIvfVsCpu PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)

add_executable(BenchMetalIvfTilingVsCpu BenchMetalIvfTilingVsCpu.mm)
target_link_libraries(BenchMetalIvfTilingVsCpu PRIVATE faiss_metal faiss)
target_include_directories(BenchMetalIvfTilingVsCpu PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(BenchMetalIvfTilingVsCpu PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)

add_executable(BenchMetalIvfAddVsCpu BenchMetalIvfAddVsCpu.mm)
target_link_libraries(BenchMetalIvfAddVsCpu PRIVATE faiss_metal faiss)
target_include_directories(BenchMetalIvfAddVsCpu PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(BenchMetalIvfAddVsCpu PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)

add_executable(BenchMetalIvfSQVsCpu BenchMetalIvfSQVsCpu.mm)
target_link_libraries(BenchMetalIvfSQVsCpu PRIVATE faiss_metal faiss)
target_include_directories(BenchMetalIvfSQVsCpu PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(BenchMetalIvfSQVsCpu PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)

add_executable(BenchMetalIvfPQVsCpu BenchMetalIvfPQVsCpu.mm)
target_link_libraries(BenchMetalIvfPQVsCpu PRIVATE faiss_metal faiss)
target_include_directories(BenchMetalIvfPQVsCpu PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(BenchMetalIvfPQVsCpu PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)

add_executable(BenchMetalBinaryFlatVsCpu BenchMetalBinaryFlatVsCpu.mm)
target_link_libraries(BenchMetalBinaryFlatVsCpu PRIVATE faiss_metal faiss)
target_include_directories(BenchMetalBinaryFlatVsCpu PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(BenchMetalBinaryFlatVsCpu PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)

add_executable(BenchRunIvf BenchRunIvf.mm)
target_include_directories(BenchRunIvf PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(BenchRunIvf PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)

add_executable(BenchRunFlat BenchRunFlat.mm)
target_include_directories(BenchRunFlat PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(BenchRunFlat PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
