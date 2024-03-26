#include <limits>

#include <surface_indirect_functions.h>

#include "cuda/vector_math.cuh"
#include "ssdfg/kernels.cuh"

// Computation
__global__
void silhouette_edges
(
	const float *__restrict__ visibility,
	float *__restrict__ convolved,
	int32_t width,
	int32_t height
)
{
	// Kernel width is 3x3
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < width * height; i += stride) {
		static constexpr int32_t dx[] = { 1, 1, -1, -1 };
		static constexpr int32_t dy[] = { 1, -1, 1, -1 };

		float kvalue = 4 * visibility[i];

		int32_t x = i % width;
		int32_t y = i / width;
		for (uint32_t j = 0; j < 4; j++) {
			int32_t nx = x + dx[j];
			int32_t ny = y + dy[j];

			if (nx < 0 || nx >= width)
				continue;

			if (ny < 0 || ny >= height)
				continue;

			int32_t ni = nx + ny * width;

			kvalue -= visibility[ni];
		}

		convolved[i] = fabs(kvalue);
	}
}

__global__
void signed_distance_field
(
	const float *__restrict__ visibility,
	const float *__restrict__ silhouette,
	float *__restrict__ sdf,
	int32_t *__restrict__ index,
	uint32_t width,
	uint32_t height
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	float2 extent = make_float2(width, height);
	for (int32_t i = tid; i < width * height; i += stride) {
		float2 p = make_float2(i % width, i / width)/extent;

		// Search the entire image and compute the closest point
		// TODO: try the spiral method, and then the shore waves method/wav propogation method
		float d = std::numeric_limits <float> ::max();
		int32_t idx = -1;

		for (int32_t x = 0; x < width; x++) {
			for (int32_t y = 0; y < height; y++) {
				int32_t ni = x + y * width;

				if (silhouette[ni] > 0) {
					float2 np = make_float2(x, y)/extent;
					float nd = length(p - np);
					if (nd < d) {
						d = nd;
						idx = ni;
					}
				}
			}
		}

		sdf[i] = d * (1 - 2 * (visibility[i] > 0));
		index[i] = idx;
	}
}

__global__
void sdf_spatial_gradient
(
	const float *__restrict__ sdf,
	float2 *__restrict__ gradients,
	uint32_t width,
	uint32_t height
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	float2 extent = make_float2(width, height);
	for (int32_t i = tid; i < width * height; i += stride) {
		int32_t x = i % width;
		int32_t y = i / width;

		// Horizontal gradient
		int32_t px = max(0, x - 1);
		int32_t nx = min(x + 1, width - 1);

		float sdfpx = sdf[px + y * width];
		float sdfnx = sdf[nx + y * width];
		float dx = width * (sdfnx - sdfpx)/(nx - px);

		// Vertical gradient
		int32_t py = max(0, y - 1);
		int32_t ny = min(y + 1, height - 1);

		float sdfpy = sdf[x + py * width];
		float sdfny = sdf[x + ny * width];
		float dy = height * (sdfny - sdfpy)/(ny - py);

		gradients[i] = make_float2(dx, dy);
	}
}

__global__
void image_space_motion_gradients
(
	const float *__restrict__ sdf_target,
	const float2 *__restrict__ gradients_target,
	const float *__restrict__ sdf_source,
	const float2 *__restrict__ gradients_source,
	float2 *__restrict__ motion_gradients,
	uint32_t width,
	uint32_t height
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	float2 extent = make_float2(width, height);
	for (int32_t i = tid; i < width * height; i += stride) {
		float psi = sdf_target[i];
		float phi = sdf_source[i];

		float2 psi_grad = gradients_target[i];
		float2 phi_grad = gradients_source[i];

		motion_gradients[i] = 2 * (psi - phi) * (psi_grad - phi_grad);
	}
}

//__global__
//void project_image_space_vectors
//(
//	const float *__restrict__ visibility,
//	const float *__restrict__ depth,
//	const float2 *__restrict__ vectors,
//	float3 *__restrict__ projected,
//	float3 horizontal,
//	float3 vertical,
//	uint32_t width,
//	uint32_t height
//)
//{
//	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//	int32_t stride = blockDim.x * gridDim.x;
//
//	for (int32_t i = tid; i < width * height; i += stride) {
//		if (visibility[i] <= 0) {
//			projected[i] = make_float3(0, 0, 0);
//			continue;
//		}
//
////		float2 vector = vectors[i] * depth[i];
//		float2 vector = vectors[i];
//
//		// TODO: negate the y direction?
//		projected[i] = vector.x * horizontal + vector.y * vertical;
//	}
//}

__forceinline__ __device__
float3 atomicAdd(float3 *addr, float3 val)
{
	float x = atomicAdd(&addr->x, val.x);
	float y = atomicAdd(&addr->y, val.y);
	float z = atomicAdd(&addr->z, val.z);
	return make_float3(x, y, z);
}

__global__
void scatter_reduce_mesh_gradient
(
	const float3 *__restrict__ projected,
	const float2 *__restrict__ barycentrics,
	const uint3 *__restrict__ triangles,
	const int32_t *__restrict__ primitives,
	float3 *__restrict__ gradients,
	uint *__restrict__ counts,
	uint32_t width,
	uint32_t height
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < width * height; i += stride) {
		float3 proj = projected[i];
		if (length(proj) < 1e-6)
			continue;

		uint index = primitives[i];
		uint3 triangle = triangles[index];
		float2 bary = barycentrics[i];

		float3 v0 = (1 - bary.x - bary.y) * proj;
		float3 v1 = bary.x * proj;
		float3 v2 = bary.y * proj;

		atomicAdd(&gradients[triangle.x], v0);
		atomicAdd(&gradients[triangle.y], v1);
		atomicAdd(&gradients[triangle.z], v2);

		atomicAdd(&counts[triangle.x], 1);
		atomicAdd(&counts[triangle.y], 1);
		atomicAdd(&counts[triangle.z], 1);
	}
}

__global__
void scatter_reduce_integrate
(
	float3 *__restrict__ gradients,
	const uint *__restrict__ samples,
	uint32_t size
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;

	for (int32_t i = tid; i < size; i += stride)
		gradients[i] = gradients[i] / fmax(1.0f, float(samples[i]));
}