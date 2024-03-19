#include <thread>
#include <mutex>

#include <microlog/microlog.h>

#include <glm/gtc/quaternion.hpp>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_vulkan.h>

#include <surface_indirect_functions.h>

#include <optix/optix.h>
#include <optix/optix_stubs.h>
#include <optix/optix_function_table_definition.h>

#include <oak/camera.hpp>
#include <oak/contexts.hpp>
#include <oak/mesh.hpp>

#include "cuda/optix/util.cuh"
#include "cuda/util.cuh"
#include "cuda/vector_math.cuh"
#include "io.hpp"
#include "shaders/optix/ssdfg.cuh"
#include "ssdfg/contexts.cuh"
#include "ssdfg/kernels.cuh"
#include "util.hpp"

struct Bounds {
	float min_x;
	float min_y;
	float max_x;
	float max_y;

	bool within(float x, float y) const {
		return (x >= min_x) && (x <= max_x)
			&& (y >= min_y) && (y <= max_y);
	}
};

struct MouseInfo {
	bool drag = false;
	bool voided = true;
	float last_x = 0.0f;
	float last_y = 0.0f;
} static mouse;

static std::vector <Bounds> image_bounds;

static bool within_any(float x, float y)
{
	for (const Bounds &bounds : image_bounds) {
		if (bounds.within(x, y))
			return true;
	}

	return false;
}

void button_callback(GLFWwindow *window, int button, int action, int mods)
{
	// Ignore if on ImGui window
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		double xpos;
		double ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		if (action == GLFW_PRESS)
			mouse.drag = within_any(xpos, ypos);
		else
			mouse.drag = false;

		if (action == GLFW_RELEASE)
			mouse.voided = true;
	}
}

void cursor_callback(GLFWwindow *window, double xpos, double ypos)
{
	Transform *camera_transform = (Transform *) glfwGetWindowUserPointer(window);

	if (mouse.voided) {
		mouse.last_x = xpos;
		mouse.last_y = ypos;
		mouse.voided = false;
	}

	float xoffset = xpos - mouse.last_x;
	float yoffset = ypos - mouse.last_y;

	mouse.last_x = xpos;
	mouse.last_y = ypos;

	constexpr float sensitivity = 0.001f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	if (mouse.drag) {
		camera_transform->rotation.x += yoffset;
		camera_transform->rotation.y -= xoffset;

		if (camera_transform->rotation.x > 89.0f)
			camera_transform->rotation.x = 89.0f;
		if (camera_transform->rotation.x < -89.0f)
			camera_transform->rotation.x = -89.0f;
	}
}

void handle_key_input(GLFWwindow *const win, Transform &camera_transform)
{
	static float last_time = 0.0f;

	constexpr float speed = 25.0f;

	float delta = speed * float(glfwGetTime() - last_time);
	last_time = glfwGetTime();

	// TODO: littlevk io system
	glm::vec3 velocity(0.0f);
	if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS)
		velocity.z -= delta;
	else if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS)
		velocity.z += delta;

	if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS)
		velocity.x -= delta;
	else if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS)
		velocity.x += delta;

	if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS)
		velocity.y += delta;
	else if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS)
		velocity.y -= delta;

	glm::quat q = glm::quat(camera_transform.rotation);
	velocity = q * glm::vec4(velocity, 0.0f);
	camera_transform.position += velocity;
}

int main()
{
	static const std::vector <const char *> extensions {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
		VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME
	};

	// Load physical device
	auto predicate = [](vk::PhysicalDevice phdev) {
		return littlevk::physical_device_able(phdev, extensions);
	};

	vk::PhysicalDevice phdev = littlevk::pick_physical_device(predicate);
	
	// Enable features
	vk::PhysicalDeviceFeatures2KHR features {};
	
	// Configure a device resource context
	DeviceResourceContext drc = DeviceResourceContext::from(phdev, { 1920, 1080 }, extensions, features);

	glfwSetMouseButtonCallback(drc.window->handle, button_callback);
	glfwSetCursorPosCallback(drc.window->handle, cursor_callback);

	// Configure rendering context and imgui
	RenderContext rc = RenderContext::from(drc);

	imgui_context_from(drc, rc);

	ImGuiIO &io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	io.ConfigWindowsMoveFromTitleBarOnly = true;

	// Initialize OptiX
	OptixDeviceContext optix_context = make_context();

	// Load the programs
	OptixModule module = optix_module_from_source(optix_context, VYNE_ROOT "/bin/ssdfg.cu.o");

	// Load program groups
	auto program_groups = optix_program_groups
	(
		optix_context, module,
		std::array <OptixProgramType, 3> {
			OptixProgramType::ray_generation("__raygen__"),
			OptixProgramType::closest_hit("__closesthit__"),
			OptixProgramType::miss("__miss__"),
		}
	);

	// Link the pipeline
	OptixPipeline pipeline = nullptr;

	optixPipelineCreate
	(
		optix_context,
		&pipeline_compile_options(),
		&pipeline_link_options(),
		program_groups.data(), program_groups.size(),
		nullptr, 0,
		&pipeline
	);

	const vk::Extent2D extent { 256, 256 };

	// Load the meshes
	Mesh target = load_geometry(VYNE_ROOT "/data/spot.obj").front();
	Mesh source = load_geometry(VYNE_ROOT "/data/sphere.obj").front();
	auto target_src = SilhouetteRenderContext::from(drc, optix_context, program_groups, target, extent);
	auto source_src = SilhouetteRenderContext::from(drc, optix_context, program_groups, source, extent);

	// Camera parameters
	Camera camera;
	Transform camera_transform;
	glfwSetWindowUserPointer(drc.window->handle, &camera_transform);

	// Differentiable rendering process
	const littlevk::ImageCreateInfo image_info {
		.width = extent.width,
		.height = extent.height,
		.format = drc.swapchain.format,
		.usage = vk::ImageUsageFlagBits::eSampled
			| vk::ImageUsageFlagBits::eTransferDst,
		.aspect = vk::ImageAspectFlagBits::eColor,
		.external = true
	};

	auto sampler = littlevk::SamplerCompiler(drc.device, drc.dal);
	auto motion_gradients_lfb = LinkedFramebuffer::from(drc, image_info, sampler);

	float2 *motion_gradients = (float2 *) cuda_alloc_buffer(sizeof(float2) * extent.width * extent.height);

	auto render = [&]() {
		camera.aspect = float(extent.width)/float(extent.height);
		target_src.render(pipeline, camera, camera_transform);
		source_src.render(pipeline, camera, camera_transform);

		image_space_motion_gradients <<< 64, 128 >>>
		(
			target_src.sdf,
			target_src.gradients,
			source_src.sdf,
			source_src.gradients,
			motion_gradients,
			extent.width,
			extent.height
		);

		cudaDeviceSynchronize();

		render_image_space_vectors <<< 64, 128 >>>
		(
			motion_gradients,
			motion_gradients_lfb.surface,
			extent.width,
			extent.height
		);

		cudaDeviceSynchronize();
	};

	std::vector <vk::DescriptorSet> views;
	views.push_back(target_src.visibility_lfb.imgui);
	views.push_back(target_src.gradients_lfb.imgui);
	views.push_back(source_src.visibility_lfb.imgui);
	views.push_back(source_src.gradients_lfb.imgui);

	// Render loop
	size_t frame = 0;
	while (drc.valid_window()) {
		// Get events
		glfwPollEvents();

		handle_key_input(drc.window->handle, camera_transform);

		auto [cmd, op] = *drc.new_frame(frame);

		render();

		LiveRenderContext(drc, rc).begin_render_pass(cmd, op);

		imgui_begin();

		ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

		if (ImGui::BeginMainMenuBar()) {
			if (ImGui::BeginMenu("View")) {
				if (ImGui::BeginMenu("Target")) {
					if (ImGui::MenuItem("Silhouette"))
						views.push_back(target_src.visibility_lfb.imgui);
					if (ImGui::MenuItem("Silhouette boundary"))
						views.push_back(target_src.boundary_lfb.imgui);
					if (ImGui::MenuItem("Depth map"))
						views.push_back(target_src.depth_lfb.imgui);
					if (ImGui::MenuItem("Signed distance field"))
						views.push_back(target_src.sdf_lfb.imgui);
					if (ImGui::MenuItem("SDF gradients"))
						views.push_back(target_src.gradients_lfb.imgui);

					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("Source")) {
					if (ImGui::MenuItem("Silhouette"))
						views.push_back(source_src.visibility_lfb.imgui);
					if (ImGui::MenuItem("Silhouette boundary"))
						views.push_back(source_src.boundary_lfb.imgui);
					if (ImGui::MenuItem("Depth map"))
						views.push_back(source_src.depth_lfb.imgui);
					if (ImGui::MenuItem("Signed distance field"))
						views.push_back(source_src.sdf_lfb.imgui);
					if (ImGui::MenuItem("SDF gradients"))
						views.push_back(source_src.gradients_lfb.imgui);

					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("Additional")) {
					if (ImGui::MenuItem("Motion gradients"))
						views.push_back(motion_gradients_lfb.imgui);

					ImGui::EndMenu();
				}

				ImGui::EndMenu();
			}

			ImGui::EndMainMenuBar();
		}

		image_bounds.clear();

		std::vector <uint32_t> erasal;
		for (uint32_t i = 0; i < views.size(); i++) {
			std::string id = "Viewport##" + std::to_string(i);

			bool open = true;
			if (ImGui::Begin(id.c_str(), &open)) {
				if (!open)
					erasal.insert(erasal.begin(), i);

				ImGui::Image(views[i], ImVec2(512, 512));

				ImVec2 min = ImGui::GetItemRectMin();
				ImVec2 max = ImGui::GetItemRectMax();

				Bounds bounds;
				bounds.min_x = min.x;
				bounds.min_y = min.y;
				bounds.max_x = max.x;
				bounds.max_y = max.y;

				image_bounds.push_back(bounds);

				ImGui::End();
			}
		}

		for (uint32_t i : erasal)
			views.erase(views.begin() + i);

		imgui_end(cmd);

		cmd.endRenderPass();

		// End of frame
		drc.end_frame(cmd, frame);
		drc.present_frame(op, frame);

		frame = 1 - frame;
	}
}