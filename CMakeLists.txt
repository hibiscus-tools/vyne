cmake_minimum_required(VERSION 3.20)

project(vyne CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall --expt-relaxed-constexpr -Xcudafe \"--diag_suppress=20012 --diag_suppress=20011 --diag_suppress=20014\"")

find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(dependencies/pybind11)
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

# Vyne python library
add_library(vyne MODULE
	source/ext/vyne.cu
	source/io.cpp
)

target_link_libraries(vyne PRIVATE
	oak assimp pybind11::module
	glfw SPIRV
	glslang::glslang Vulkan::Vulkan
	glslang::glslang-default-resource-limits
)

pybind11_extension(vyne)

# Silhouette signed-distance field gradients
add_dependencies(vyne optix_shaders)

# Copy as a python package
add_custom_command(TARGET vyne
	POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:vyne>
		"${CMAKE_SOURCE_DIR}/.venv/lib/python3.11/site-packages"
)