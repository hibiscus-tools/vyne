#pragma once

#include <cstdint>

__global__
void silhouette_edges
(
	const float *__restrict__,
	float *__restrict__,
	int32_t, int32_t
);

__global__
void signed_distance_field
(
	const float *__restrict__,
	const float *__restrict__,
	float *__restrict__,
	int32_t *__restrict__,
	uint32_t, uint32_t
);

__global__
void sdf_spatial_gradient(const float *__restrict__, float2 *__restrict__, uint32_t, uint32_t);

__global__
void image_space_motion_gradients(const float *__restrict__, const float2 *__restrict__, const float *__restrict__, const float2 *__restrict__, float2 *__restrict__, uint32_t, uint32_t);
//
//__global__
//void project_image_space_vectors(const float *__restrict__, const float *__restrict__, const float2 *__restrict__, float3 *__restrict__, float3, float3, uint32_t, uint32_t);

__global__
void scatter_reduce_mesh_gradient(const float3 *__restrict__, const float2 *__restrict__, const uint3 *__restrict__, const int32_t *__restrict__, float3 *__restrict__, uint *__restrict__, uint32_t, uint32_t);

__global__
void scatter_reduce_integrate(float3 *__restrict__, const uint *__restrict__, uint32_t);