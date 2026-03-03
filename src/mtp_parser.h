#pragma once
#include <string>
#include <vector>

struct MTPData
{
    int species_count = 0;
    int radial_basis_size = 0;
    int alpha_moments_count = 0;
    int alpha_scalar_moments = 0;
    std::vector<int> alpha_index_basic;
    std::vector<int> alpha_index_times;
    std::vector<int> alpha_moment_mapping;
};

MTPData parse_mtp(const std::string &filepath);