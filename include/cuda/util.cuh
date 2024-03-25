#pragma once

#include <vector>

inline CUdeviceptr cuda_alloc(uint32_t size)
{
	CUdeviceptr ptr;
	cudaMalloc((void **) &ptr, size);
	return ptr;
}

template <typename T>
inline T *cuda_alloc_buffer(uint32_t elements)
{
	T *ptr;
	cudaMalloc((void **) &ptr, sizeof(T) * elements);
	return ptr;
}

template <typename T>
CUdeviceptr cuda_element_buffer(const T &value)
{
	uint32_t size = sizeof(T);
	CUdeviceptr ptr = cuda_alloc(size);
	cudaMemcpy((void *) ptr, &value, size, cudaMemcpyHostToDevice);
	return ptr;
}

template <typename T>
CUdeviceptr cuda_vector_buffer(const std::vector <T> &buffer)
{
	uint32_t size = buffer.size() * sizeof(T);
	CUdeviceptr ptr = cuda_alloc(size);
	cudaMemcpy((void *) ptr, buffer.data(), size, cudaMemcpyHostToDevice);
	return ptr;
}

template <typename T>
void cuda_element_copy(CUdeviceptr ptr, const T &value)
{
	// TODO: error handling
	cudaMemcpy((void *) ptr, &value, sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void cuda_download(CUdeviceptr ptr, std::vector <T> &buffer)
{
	cudaMemcpy((void *) buffer.data(), (void *) ptr, sizeof(T) * buffer.size(), cudaMemcpyDeviceToHost);
}