#pragma once
#include <vector>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

struct Individual
{
    std::vector<uint8_t> bits;
    double cost = 0.0;
    double sse = 0.0;
    int rank = 0;
    double crowding = 0.0;

    inline bool get_bit(size_t index) const
    {
        return (bits[index / 8] >> (index % 8)) & 1;
    }
    inline void set_bit(size_t index, bool val)
    {
        if (val)
            bits[index / 8] |= (1 << (index % 8));
        else
            bits[index / 8] &= ~(1 << (index % 8));
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