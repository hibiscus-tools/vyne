#include <limits>

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
		for (int32_t x = 0; x < width; x++) {
			for (int32_t y = 0; y < height; y++) {
				int32_t ni = x + y * width;

				if (silhouette[ni] > 0) {
					float2 np = make_float2(x, y)/extent;
					d = fmin(d, length(p - np));
				}
			}
		}

		sdf[i] = d * (1 - 2 * (visibility[i] > 0));
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
		// Horizontal gradient
		int32_t x = i % width;
		int32_t y = i / width;

		int32_t px = max(0, x - 1);
		int32_t nx = min(x + 1, width - 1);

		float sdfpx = sdf[px + y * width];
		float sdfnx = sdf[nx + y * width];
		float dx = width * (sdfnx - sdfpx)/(nx - px);

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

// Rendering
__global__
void render_mask
(
	const float *__restrict__ visibility,
	cudaSurfaceObject_t fb,
	uint32_t width,
	uint32_t height
)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < width * height; i += stride) {
		int32_t x = i % width;
		int32_t y = i / width;
		if (visibility[i] > 0)
			surf2Dwrite(make_uchar4(0, 0, 0, 255), fb, x * sizeof(uchar4), y);
		else
			surf2Dwrite(make_uchar4(150, 150, 150, 255), fb, x * sizeof(uchar4), y);
	}
}

__global__
void render_scalar_field
(
	const float *__restrict__ field,
	cudaSurfaceObject_t fb,
	uint32_t width,
	uint32_t height
)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < width * height; i += stride) {
		int32_t x = i % width;
		int32_t y = i / width;

//		float v = fmin(1, fmax(0, field[i]));
		float v = 0.5 + 0.5 * cos(128 * field[i]);
		uint p = 255.0f * v;

		surf2Dwrite(make_uchar4(p, p, p, 0xff), fb, x * sizeof(uchar4), y);
	}
}

__global__
void render_sdf
(
	const float *__restrict__ sdf,
	cudaSurfaceObject_t fb,
	uint32_t width,
	uint32_t height
)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < width * height; i += stride) {
		int32_t x = i % width;
		int32_t y = i / width;
		float k = 0.5 + 0.5 * cos(128.0f * sdf[i]);
		float3 blue = k * make_float3(0.2, 0.5, 1.0);
		float3 red = (1 - k) * make_float3(1.0, 0.5, 0.2);
		surf2Dwrite(rgb_to_uchar4(blue + red), fb, x * sizeof(uchar4), y);
	}
}

__global__
void render_image_space_vectors
(
	const float2 *__restrict__ vectors,
	cudaSurfaceObject_t fb,
	uint32_t width,
	uint32_t height
)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < width * height; i += stride) {
		int32_t x = i % width;
		int32_t y = i / width;
		float3 color = make_float3(0.5 + 0.5 * vectors[i], 0);
		surf2Dwrite(rgb_to_uchar4(color), fb, x * sizeof(uchar4), y);
	}
}
