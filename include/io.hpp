#pragma once

#include <vector>
#include <filesystem>

#include <oak/mesh.hpp>

std::vector <Mesh> load_geometry(const std::filesystem::path &);