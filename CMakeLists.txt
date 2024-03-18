cmake_minimum_required(VERSION 3.20)

project(vyne CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall --expt-relaxed-constexpr")

find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)
find_package(CUDAToolkit 12.0 REQUIRED)

add_subdirectory(dependencies/oak)

include_directories(.
	include
	dependencies
	dependencies/oak/include
	dependencies/oak/dependencies
	dependencies/oak/dependencies/glm
	dependencies/oak/dependencies/imgui
	dependencies/optix
)

# Global project macros
add_definitions("-DVYNE_ROOT=\"${CMAKE_SOURCE_DIR}\"")

# OptiX shaders
add_library(optix_shaders_compile OBJECT shaders/optix/ssdfg.cu)

target_compile_options(optix_shaders_compile PUBLIC -optix-ir -lineinfo)

add_custom_target(optix_shaders
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_BINARY_DIR}/CMakeFiles/optix_shaders_compile.dir/shaders/optix
		${CMAKE_SOURCE_DIR}/bin
	DEPENDS optix_shaders_compile
)

# Silhouette signed-distance field gradients
add_executable(ssdfg
	source/main.cu
	source/io.cpp
	source/cuda/interop.cu
	source/ssdfg/contexts.cu
)

target_link_libraries(
	ssdfg
	oak assimp glfw SPIRV
	glslang::glslang Vulkan::Vulkan
	glslang::glslang-default-resource-limits
)

add_dependencies(ssdfg optix_shaders)