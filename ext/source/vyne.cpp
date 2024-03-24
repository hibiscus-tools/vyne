#include <torch/extension.h>

#include <oak/mesh.hpp>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	py::class_ <Mesh> (m, "Mesh")
	        .def(py::init <> ());
}