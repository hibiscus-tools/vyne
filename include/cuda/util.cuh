#pragma once

#include <vector>

inline CUdeviceptr cuda_alloc_buffer(uint32_t size)
{
	CUdeviceptr ptr;
	cudaMalloc((void **) &ptr, size);
	return ptr;
}

template <typename T>
CUdeviceptr cuda_element_buffer(const T &value)
{
	uint32_t size = sizeof(T);
	CUdeviceptr ptr = cuda_alloc_buffer(size);
	cudaMemcpy((void *) ptr, &value, size, cudaMemcpyHostToDevice);
	return ptr;
}

template <typename T>
void cuda_element_copy(CUdeviceptr ptr, const T &value)
{
	cudaMemcpy((void *) ptr, &value, sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
CUdeviceptr cuda_vector_buffer(const std::vector <T> &buffer)
{
	uint32_t size = buffer.size() * sizeof(T);
	CUdeviceptr ptr = cuda_alloc_buffer(size);
	cudaMemcpy((void *) ptr, buffer.data(), size, cudaMemcpyHostToDevice);
	return ptr;
}
