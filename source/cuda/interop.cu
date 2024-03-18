#include "cuda/interop.cuh"

cudaTextureObject_t cuda_import_vulkan_texture(const vk::Device &device, const littlevk::Image &image)
{
	int fd = littlevk::find_memory_fd(device, image.memory);

	cudaExternalMemoryHandleDesc memory_handle {};
	memory_handle.type = cudaExternalMemoryHandleTypeOpaqueFd;
	memory_handle.handle.fd = fd;
	memory_handle.size = image.device_size();

	cudaExternalMemory_t external_memory {};
	cudaImportExternalMemory(&external_memory, &memory_handle);

	cudaExternalMemoryMipmappedArrayDesc mipmap_description {};
	mipmap_description.offset = 0;
	mipmap_description.numLevels = 1;
	mipmap_description.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	mipmap_description.extent = make_cudaExtent(image.extent.width, image.extent.height, 1);

	cudaMipmappedArray_t mipmap_array = nullptr;
	cudaExternalMemoryGetMappedMipmappedArray(&mipmap_array, external_memory, &mipmap_description);

	cudaResourceDesc resource_description {};
	resource_description.resType = cudaResourceTypeMipmappedArray;
	resource_description.res.mipmap.mipmap = mipmap_array;

	cudaTextureDesc texture_description {};
	texture_description.normalizedCoords = true;
	texture_description.readMode = cudaReadModeNormalizedFloat;
	texture_description.filterMode = cudaFilterModeLinear;

	cudaTextureObject_t texture {};
	cudaCreateTextureObject(&texture, &resource_description, &texture_description, nullptr);

	return texture;
}

cudaSurfaceObject_t cuda_import_vulkan_surface(const vk::Device &device, const littlevk::Image &image)
{
	cudaError error;

	int fd = littlevk::find_memory_fd(device, image.memory);

	cudaExternalMemoryHandleDesc memory_handle {};
	memory_handle.type = cudaExternalMemoryHandleTypeOpaqueFd;
	memory_handle.handle.fd = fd;
	memory_handle.size = image.device_size();

	cudaExternalMemory_t external_memory {};
	error = cudaImportExternalMemory(&external_memory, &memory_handle);
	cuda_check(error);

	cudaExternalMemoryMipmappedArrayDesc mipmap_description {};
	mipmap_description.offset = 0;
	mipmap_description.numLevels = 1;
	mipmap_description.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	mipmap_description.extent = make_cudaExtent(image.extent.width, image.extent.height, 0);

	cudaMipmappedArray_t mipmap_array = nullptr;
	error = cudaExternalMemoryGetMappedMipmappedArray(&mipmap_array, external_memory, &mipmap_description);
	cuda_check(error);

	cudaArray_t level_array = nullptr;
	error = cudaGetMipmappedArrayLevel(&level_array, mipmap_array, 0);
	cuda_check(error);

	cudaResourceDesc resource_description {};
	resource_description.resType = cudaResourceTypeArray;
	resource_description.res.array.array = level_array;

	cudaSurfaceObject_t surface {};
	error = cudaCreateSurfaceObject(&surface, &resource_description);
	cuda_check(error);

	return surface;
}
