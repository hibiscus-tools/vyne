#pragma once

#include <imgui/backends/imgui_impl_vulkan.h>

#include <littlevk/littlevk.hpp>

#include <oak/contexts.hpp>

#define cuda_check(error) \
	if (error != cudaSuccess) { \
		fprintf(stderr, "cuda error (%s | %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
		abort(); \
	}

template <typename T>
T *cuda_import_vulkan_buffer(const vk::Device &device, const littlevk::Buffer &buffer)
{
	int fd = littlevk::find_memory_fd(device, buffer.memory);

	cudaExternalMemoryHandleDesc memory_handle {};
	memory_handle.type = cudaExternalMemoryHandleTypeOpaqueFd;
	memory_handle.handle.fd = fd;
	memory_handle.size = buffer.device_size();

	cudaExternalMemory_t external_memory {};
	cudaImportExternalMemory(&external_memory, &memory_handle);

	cudaExternalMemoryBufferDesc buffer_description {};
	buffer_description.offset = 0;
	buffer_description.size = buffer.device_size();

	T *cuda_buffer = nullptr;
	cudaExternalMemoryGetMappedBuffer((void **) &cuda_buffer, external_memory, &buffer_description);

	return cuda_buffer;
}

cudaTextureObject_t cuda_import_vulkan_texture(const vk::Device &, const littlevk::Image &);
cudaSurfaceObject_t cuda_import_vulkan_surface(const vk::Device &, const littlevk::Image &);

// TODO: source file
struct LinkedFramebuffer {
	littlevk::Image image;
	cudaSurfaceObject_t surface;
	vk::DescriptorSet imgui;

	static LinkedFramebuffer from(const DeviceResourceContext &drc, const littlevk::ImageCreateInfo &info, const vk::Sampler &sampler) {
		littlevk::Image image = littlevk::image(drc.device, info, drc.memory_properties).unwrap(drc.dal);
		cudaSurfaceObject_t surface = cuda_import_vulkan_surface(drc.device, image);

		littlevk::submit_now
		(
			drc.device,
			drc.command_pool,
			drc.graphics_queue,
			[&](const vk::CommandBuffer &cmd) {
				littlevk::transition(cmd, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
			}
		);

		vk::DescriptorSet imgui = ImGui_ImplVulkan_AddTexture
		(
			static_cast <VkSampler> (sampler),
			static_cast <VkImageView> (image.view),
			static_cast <VkImageLayout> (vk::ImageLayout::eGeneral)
		);

		return { image, surface, imgui };
	}
};
