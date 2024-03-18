#include "ssdfg.cuh"
#include "cuda/vector_math.cuh"

extern "C" __constant__ Packet packet;

__forceinline__ __host__ __device__
static float3 ray_at_uv(float2 uv)
{
	return normalize(packet.lower_left + uv.x * packet.horizontal + uv.y * packet.vertical - packet.origin);
}

extern "C" __global__ void __raygen__()
{
	const uint3 idx = optixGetLaunchIndex();
	const uint2 extent = packet.resolution;
	const uint32_t index = idx.x + idx.y * extent.x;

	float2 uv = make_float2(idx.x + 0.5, idx.y + 0.5) / make_float2(extent.x, extent.y);
	uv = make_float2(uv.x, 1 - uv.y);
	float3 ray = ray_at_uv(uv);

	packet.visibility[index] = 0;

	unsigned int i0;
	unsigned int i1;
	pack_pointer(&packet.visibility[index], i0, i1);

	optixTrace
	(
		packet.gas,
		packet.origin, ray,
		0, 1e16, 0,
		OptixVisibilityMask(0xff),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0, 1, 0, i0, i1
	);
}

extern "C" __global__ void __closesthit__()
{
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
	float *fbptr = unpack_pointer <float> (i0, i1);
	*fbptr = 1.0f;
}

extern "C" __global__ void __miss__()
{
	unsigned int i0 = optixGetPayload_0();
	unsigned int i1 = optixGetPayload_1();
}