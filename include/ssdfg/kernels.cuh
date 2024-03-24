#pragma once

#include <cstdint>

// Computation
__global__
void silhouette_edges(const float *__restrict__, float *__restrict__, int32_t, int32_t);

__global__
void signed_distance_field(const float *__restrict__, const float *__restrict__, float *__restrict__, uint32_t, uint32_t);

__global__
void sdf_spatial_gradient(const float *__restrict__, float2 *__restrict__, uint32_t, uint32_t);

__global__
void image_space_motion_gradients(const float *__restrict__, const float2 *__restrict__, const float *__restrict__, const float2 *__restrict__, float2 *__restrict__, uint32_t, uint32_t);

__global__
void project_image_space_vectors(const float *__restrict__, const float *__restrict__, const float2 *__restrict__, float3 *__restrict__, float3, float3, uint32_t, uint32_t);

__global__
void scatter_reduce_mesh_gradient(const float3 *__restrict__, const float2 *__restrict__, const uint3 *__restrict__, const int32_t *__restrict__, float3 *__restrict__, uint *__restrict__, uint32_t, uint32_t);

__global__
void scatter_reduce_integrate(float3 *__restrict__, const uint *__restrict__, uint32_t);

// Rendering
__global__
void render_mask(const float *__restrict__, cudaSurfaceObject_t, uint32_t, uint32_t);

__global__
void render_scalar_field(const float *__restrict__, cudaSurfaceObject_t, uint32_t, uint32_t);

__global__
void render_sdf(const float *__restrict__, cudaSurfaceObject_t, uint32_t, uint32_t);

__global__
void render_image_space_vectors(const float2 *__restrict__, cudaSurfaceObject_t, uint32_t, uint32_t);

__global__
void render_normalized_vectors(const float3 *__restrict__, cudaSurfaceObject_t, uint32_t, uint32_t);

template <typename T>
__global__
void render_vector_color(const T *__restrict__, cudaSurfaceObject_t, uint32_t, uint32_t);

__global__
void render_triangle_attribute_magnitudes(const int32_t *__restrict__, const float2 *__restrict__, const uint3 *__restrict__, const float3 *__restrict__, cudaSurfaceObject_t, uint32_t, uint32_t);
