#include "mtp_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

// Helper to extract a flattened list of integers from a string like "{{0, 1}, {2, 3}}"
static std::vector<int> extract_ints(const std::string &s)
{
    std::vector<int> result;
    std::string temp = s;
    // Replace all braces and commas with spaces
    for (char &c : temp)
    {
        if (c == '{' || c == '}' || c == ',')
            c = ' ';
    }
    std::stringstream ss(temp);
    int val;
    while (ss >> val)
    {
        result.push_back(val);
    }
    return result;
}

MTPData parse_mtp(const std::string &filepath)
{
    MTPData data;
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open MTP file: " + filepath);
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Trim leading and trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty() || line == "MTP")
            continue;

        // Stop immediately if we hit the active learning binary segment
        if (line.find("#MVS_v1.1") == 0)
            break;

        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos)
        {
            std::string key = line.substr(0, eq_pos);
            std::string val = line.substr(eq_pos + 1);

            // Trim key and value
            key.erase(key.find_last_not_of(" \t") + 1);
            val.erase(0, val.find_first_not_of(" \t"));

            if (key == "species_count")
                data.species_count = std::stoi(val);
            else if (key == "radial_basis_size")
                data.radial_basis_size = std::stoi(val);
            else if (key == "alpha_moments_count")
                data.alpha_moments_count = std::stoi(val);
            else if (key == "alpha_scalar_moments")
                data.alpha_scalar_moments = std::stoi(val);
            else if (key == "alpha_index_basic")
                data.alpha_index_basic = extract_ints(val);
            else if (key == "alpha_index_times")
                data.alpha_index_times = extract_ints(val);
            else if (key == "alpha_moment_mapping")
                data.alpha_moment_mapping = extract_ints(val);
        }
    }
    return data;
}