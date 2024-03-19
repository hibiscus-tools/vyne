#pragma once

#include <cstdint>

#include <glm/glm.hpp>

// Vector constructors
__forceinline__ __host__ __device__
float3 make_float3(float2 v, float x)
{
	return make_float3(v.x, v.y, x);
}

// Vector addition
__forceinline__ __host__ __device__
float2 operator+(float2 A, float2 B)
{
	return make_float2(A.x + B.x, A.y + B.y);
}

__forceinline__ __host__ __device__
float2 operator+(float k, float2 V)
{
	return make_float2(V.x + k, V.y + k);
}

 __forceinline__ __host__ __device__
float2 operator+(float2 V, float k)
{
	return make_float2(V.x + k, V.y + k);
}

__forceinline__ __host__ __device__
float3 operator+(float3 A, float3 B)
{
	return make_float3(A.x + B.x, A.y + B.y, A.z + B.z);
}

__forceinline__ __host__ __device__
float3 operator+(float k, float3 V)
{
	return make_float3(V.x + k, V.y + k, V.z + k);
}

__forceinline__ __host__ __device__
float3 operator+(float3 V, float k)
{
	return make_float3(V.x + k, V.y + k, V.z + k);
}

// Vector subtraction
__forceinline__ __host__ __device__
float2 operator-(float2 A, float2 B)
{
	return make_float2(A.x - B.x, A.y - B.y);
}

__forceinline__ __host__ __device__
float3 operator-(float3 A, float3 B)
{
	return make_float3(A.x - B.x, A.y - B.y, A.z - B.z);
}

// Vector multiplication
__forceinline__ __host__ __device__
float2 operator*(float2 V, float k)
{
	return make_float2(V.x * k, V.y * k);
}

__forceinline__ __host__ __device__
float2 operator*(float k, float2 V)
{
	return make_float2(V.x * k, V.y * k);
}

__forceinline__ __host__ __device__
float3 operator*(float3 V, float k)
{
	return make_float3(V.x * k, V.y * k, V.z * k);
}

__forceinline__ __host__ __device__
float3 operator*(float k, float3 V)
{
	return make_float3(V.x * k, V.y * k, V.z * k);
}

// Vector division
__forceinline__ __host__ __device__
float2 operator/(float2 A, float2 B)
{
	return make_float2(A.x / B.x, A.y / B.y);
}

__forceinline__ __host__ __device__
float3 operator/(float3 A, float3 B)
{
	return make_float3(A.x / B.x, A.y / B.y, A.z / B.z);
}

// Vector normalization
__forceinline__ __host__ __device__
float2 normalize(float2 v)
{
	float s = sqrt(v.x * v.x + v.y * v.y);
	return make_float2(v.x/s, v.y/s);
}

__forceinline__ __host__ __device__
float3 normalize(float3 v)
{
	float s = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return make_float3(v.x/s, v.y/s, v.z/s);
}

// Vector lengths
__forceinline__ __host__ __device__
float length(float2 v)
{
	return sqrt(v.x * v.x + v.y * v.y);
}

// Color utilities
__forceinline__ __host__ __device__
uchar4 rgb_to_uchar4(float3 c)
{
	uint32_t r = 255.0f * c.x;
	uint32_t g = 255.0f * c.y;
	uint32_t b = 255.0f * c.z;
	return make_uchar4(r, g, b, 0xff);
}

// Miscellaneous utiltiies
__forceinline__ __host__ __device__
float3 glm_to_float3(const glm::vec3 &v)
{
	return make_float3(v.x, v.y, v.z);
}
