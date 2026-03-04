#pragma once
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>

struct Individual
{
    // Note: std::vector<char> is used instead of std::vector<bool>
    // to bypass the C++ bit-packing specialization and allow MPI access.
    std::vector<char> genes;
    double cost = 0.0;
    double sse = 0.0;
    int rank = 0;
    double crowding = 0.0;

    inline bool get_gene(size_t index) const
    {
        return genes[index] != 0;
    }
    inline void set_gene(size_t index, bool val)
    {
        genes[index] = val ? 1 : 0;
    }
};

template <typename T>
std::vector<T> read_binary(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open " + filename);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<T> buffer(size / sizeof(T));
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
        throw std::runtime_error("Error reading " + filename);
    return buffer;
}