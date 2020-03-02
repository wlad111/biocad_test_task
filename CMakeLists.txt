cmake_minimum_required(VERSION 3.15)

project(cuda_test LANGUAGES CXX CUDA)


function(maybe_enable_cuda TARGET)
     set(CMAKE_CUDA_STANDARD 14)
     find_package(CUDA 10.2 REQUIRED)

     if(CUDA_FOUND)
         message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
         message(STATUS "CUDA lib: ${CUDA_LIBRARIES}")
         include(CheckLanguage)
         check_language(CUDA)
     else()
         message(STATUS "CUDA not found")
     endif()

     include_directories(${CUDA_INCLUDE_DIRS})
     target_link_libraries(${TARGET} ${CUDA_LIBRARIES})

     set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
     set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)

     if(APPLE)
         # We need to add the path to the driver (libcuda.dylib) as an rpath,
         # so that the static cuda runtime can find it at runtime.
         set_property(TARGET ${TARGET} PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
     endif()

endfunction(maybe_enable_cuda)


add_library(kernels
        sum.cu sum.cuh)

maybe_enable_cuda(kernels)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61")

add_executable(cuda_test main.cpp)

target_link_libraries(kernels ${CUDA_LIBRARIES})
target_link_libraries(cuda_test kernels ${CUDA_LIBRARIES})


include_directories(${CUDA_INCLUDE_DIRS})

target_include_directories(cuda_test INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ ${CUDA_INCLUDE_DIRS}/
        )