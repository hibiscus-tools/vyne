#pragma once

#include <cstdint>

#include <optix/optix.h>

struct Packet {
	// Camera properties
	float3 origin;
	float3 lower_left;
	float3 horizontal;
	float3 vertical;

	// Handle to the surface
	OptixTraversableHandle gas;

	// Mesh properties
	uint3 *triangles;
	float3 *normals;

	// Framebuffers and resolution
	float *visibility;
	float *depth;
	int32_t *index;
	float2 *barycentrics;

	uint2 resolution;
};

__forceinline__ __host__ __device__
uint32_t rgb_to_hex(uint32_t r, uint32_t g, uint32_t b)
{
	return 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
}

__forceinline__ __host__ __device__
uint32_t rgb_to_hex(float3 c)
{
	uint32_t r = 255.0f * c.x;
	uint32_t g = 255.0f * c.y;
	uint32_t b = 255.0f * c.z;
	return 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
}

// Packing pointers for 32-bit registers
template <class T>
__forceinline__ __host__ __device__
T *unpack_pointer(uint32_t i0, uint32_t i1)
{
	const uint64_t uptr = static_cast <uint64_t> (i0) << 32 | i1;
	T *ptr = reinterpret_cast <T *> (uptr);
	return ptr;
}

template <class T>
__forceinline__ __host__ __device__
void pack_pointer(T *ptr, uint32_t &i0, uint32_t &i1)
{
	const uint64_t uptr = reinterpret_cast <uint64_t> (ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}