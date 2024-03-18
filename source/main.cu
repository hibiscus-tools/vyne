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

inline float3 glm_to_float3(const glm::vec3 &v)
{
	return make_float3(v.x, v.y, v.z);
}

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
		float d = FLT_MAX;
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

__forceinline__ __host__ __device__
uchar4 rgb_to_uchar4(float3 c)
{
	uint32_t r = 255.0f * c.x;
	uint32_t g = 255.0f * c.y;
	uint32_t b = 255.0f * c.z;
	return make_uchar4(r, g, b, 0xff);
}

__global__
void display_silhouette
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
void display_sdf
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
void display_sdf_gradients
(
	const float2 *__restrict__ gradients,
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
		float3 color = make_float3(0.5 + 0.5 * gradients[i], 0);
		surf2Dwrite(rgb_to_uchar4(color), fb, x * sizeof(uchar4), y);
	}
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

	auto src = SilhouetteRenderContext::from(drc, optix_context, program_groups, target, extent);

	// Allocate the parameters
	Packet packet;
	Camera camera;
	Transform camera_transform;

	glfwSetWindowUserPointer(drc.window->handle, &camera_transform);

	CUdeviceptr device_packet = cuda_element_buffer(packet);

	// Differentiable rendering process
	auto rtx_functor = [&]() {
		camera.aspect = float(extent.width)/float(extent.height);

		RayFrame rayframe = camera.rayframe(camera_transform);
		packet.origin = glm_to_float3(rayframe.origin);
		packet.lower_left = glm_to_float3(rayframe.lower_left);
		packet.horizontal = glm_to_float3(rayframe.horizontal);
		packet.vertical = glm_to_float3(rayframe.vertical);

		packet.visibility = src.visibility;
		packet.resolution = make_uint2(extent.width, extent.height);

		packet.gas = src.gas;

		cuda_element_copy(device_packet, packet);

		optixLaunch
		(
			pipeline, 0,
			device_packet, sizeof(Packet), &src.sbt,
			extent.width, extent.height, 1
		);

		cudaDeviceSynchronize();

		// TODO: cuStreams
		silhouette_edges <<<64, 128>>> (src.visibility, src.boundary, extent.width, extent.height);
		cudaDeviceSynchronize();

		signed_distance_field <<<64, 128>>> (src.visibility, src.boundary, src.sdf, extent.width, extent.height);
		cudaDeviceSynchronize();

		sdf_spatial_gradient <<<64, 128>>> (src.sdf, src.gradients, extent.width, extent.height);
		cudaDeviceSynchronize();

		display_silhouette <<<64, 128>>> (src.visibility, src.visibility_lfb.surface, extent.width, extent.height);
		cudaDeviceSynchronize();

		display_silhouette <<<64, 128>>> (src.boundary, src.boundary_lfb.surface, extent.width, extent.height);
		cudaDeviceSynchronize();

		display_sdf <<<64, 128>>> (src.sdf, src.sdf_lfb.surface, extent.width, extent.height);
		cudaDeviceSynchronize();

		display_sdf_gradients <<<64, 128>>> (src.gradients, src.gradients_lfb.surface, extent.width, extent.height);
		cudaDeviceSynchronize();
	};

	std::vector <vk::DescriptorSet> views;
	views.push_back(src.visibility_lfb.imgui);
	views.push_back(src.sdf_lfb.imgui);
	views.push_back(src.gradients_lfb.imgui);

	// Render loop
	size_t frame = 0;
	while (drc.valid_window()) {
		// Get events
		glfwPollEvents();

		handle_key_input(drc.window->handle, camera_transform);

		auto [cmd, op] = *drc.new_frame(frame);

		rtx_functor();

		LiveRenderContext(drc, rc).begin_render_pass(cmd, op);

		imgui_begin();

		ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

		if (ImGui::BeginMainMenuBar()) {
			if (ImGui::BeginMenu("View")) {
				if (ImGui::MenuItem("Silhouette"))
					views.push_back(src.visibility_lfb.imgui);
				if (ImGui::MenuItem("Silhouette Boundary"))
					views.push_back(src.boundary_lfb.imgui);
				if (ImGui::MenuItem("Signed distance field"))
					views.push_back(src.sdf_lfb.imgui);
				if (ImGui::MenuItem("SDF gradients"))
					views.push_back(src.gradients_lfb.imgui);

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