#pragma once

#include <fstream>
#include <sstream>
#include <string>

#include <microlog/microlog.h>

inline std::string read_file(const std::string &path)
{
	std::ifstream file(path);
	ulog_assert(file.is_open(), "readfile", "could not open file: %s\n", path.c_str());
	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}
