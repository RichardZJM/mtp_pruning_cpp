#pragma once
#include "core.h"
#include <random>
#include <string>
#include <vector>

class NSGA2
{
    int pop_size, n_var;
    std::mt19937 gen;

public:
    NSGA2(int pop_size_, int n_var_, int seed);
    void initialize_population(std::vector<Individual> &pop);
    void generate_offspring(const std::vector<Individual> &parents, std::vector<Individual> &offspring);
    void survival(std::vector<Individual> &pop, std::vector<Individual> &offspring);
    void save_pareto(const std::vector<Individual> &pop, const std::string &prefix);
};