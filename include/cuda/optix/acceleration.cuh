#pragma once

#include <optix/optix.h>
#include <optix/optix_stubs.h>

#include <oak/mesh.hpp>

#include <microlog/microlog.h>

#include "cuda/util.cuh"

OptixTraversableHandle gas_from_mesh(OptixDeviceContext optix_context, const Mesh &mesh)
{
	constexpr uint32_t triangle_flags[] { OPTIX_PROGRAM_GROUP_FLAGS_NONE };

	OptixAccelBuildOptions options {};
	options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixBuildInput build_input {};
	build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	CUdeviceptr positions = cuda_vector_buffer(mesh.positions);
	CUdeviceptr triangles = cuda_vector_buffer(mesh.triangles);

	OptixBuildInputTriangleArray &triangle_array = build_input.triangleArray;

	triangle_array.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangle_array.numVertices = mesh.positions.size();
	triangle_array.vertexBuffers = &positions;

	triangle_array.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangle_array.numIndexTriplets = mesh.triangles.size();
	triangle_array.indexBuffer = triangles;

	triangle_array.flags = triangle_flags;

	triangle_array.numSbtRecords = 1;
	triangle_array.sbtIndexOffsetBuffer = 0;
	triangle_array.sbtIndexOffsetSizeInBytes = 0;
	triangle_array.sbtIndexOffsetStrideInBytes = 0;

	OptixAccelBufferSizes buffer_sizes;

	optixAccelComputeMemoryUsage
	(
		optix_context, &options,
		&build_input, 1, &buffer_sizes
	);

	CUdeviceptr gas_buffer = cuda_alloc(buffer_sizes.outputSizeInBytes);
	CUdeviceptr gas_temporary = cuda_alloc(buffer_sizes.tempSizeInBytes);

	OptixTraversableHandle gas;

	optixAccelBuild
	(
		optix_context, 0, &options,
		&build_input, 1,
		gas_temporary, buffer_sizes.tempSizeInBytes,
		gas_buffer, buffer_sizes.outputSizeInBytes,
		&gas, nullptr, 0
	);

	return gas;
}
