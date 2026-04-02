#pragma once
#include <random>
#include <string>
#include <vector>

class NSGA2
{
public:
    int pop_size, n_var;
    std::mt19937 gen;

    // Direct pre-allocated buffers
    std::vector<char> genes;      // length: 2 * pop_size * n_var
    std::vector<double> cost_sse; // length: 2 * pop_size * 2
    std::vector<int> rank;        // length: 2 * pop_size
    std::vector<double> crowding; // length: 2 * pop_size

    NSGA2(int pop_size_, int n_var_, int seed);
    void initialize_population(const std::string &pop_file = "");
    void generate_offspring();
    void survival(int num_inds);
    void save_pareto(const std::string &prefix);
};