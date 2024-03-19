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

	float2 uv = make_float2(idx.x + 0.5, extent.y - (idx.y + 0.5)) / make_float2(extent.x, extent.y);
//	uv = make_float2(uv.x, 1 - uv.y);
	float3 ray = ray_at_uv(uv);

	uint index = idx.x + idx.y * extent.x;
	packet.visibility[index] = 0;
	packet.depth[index] = -1;

	optixTrace
	(
		packet.gas,
		packet.origin, ray,
		0, 1e16, 0,
		OptixVisibilityMask(0xff),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0, 1, 0, index
	);
}

extern "C" __global__ void __closesthit__()
{
	uint index = optixGetPayload_0();
	packet.visibility[index] = 1;
	packet.depth[index] = optixGetRayTmax();
}

extern "C" __global__ void __miss__() {}