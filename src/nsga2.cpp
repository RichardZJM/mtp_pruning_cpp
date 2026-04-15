#include "nsga2.h"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

NSGA2::NSGA2(int pop_size_, int n_var_, int seed) : pop_size(pop_size_), n_var(n_var_), gen(seed)
{
    // Pre-allocate buffers for up to 2 * pop_size to support offspring alongside parents seamlessly
    genes.resize(2 * pop_size * n_var, 0);
    cost_sse.resize(2 * pop_size * 2, 0.0);
    rank.resize(2 * pop_size, 0);
    crowding.resize(2 * pop_size, 0.0);
}

void NSGA2::initialize_population(const std::string &pop_file)
{
    assert(pop_size >= 2 && "Population size must be at least 2.");

    // Clear the first pop_size section
    std::fill(genes.begin(), genes.begin() + pop_size * n_var, 0);

    int loaded_count = 0;
    if (!pop_file.empty())
    {
        std::ifstream f(pop_file);
        if (f.is_open())
        {
            std::string line;
            while (std::getline(f, line) && loaded_count < pop_size)
            {
                std::stringstream ss(line);
                std::string token;
                int j = 0;
                while (std::getline(ss, token, ',') && j < n_var)
                {
                    if (token == "1")
                        genes[loaded_count * n_var + j] = 1;
                    else
                        genes[loaded_count * n_var + j] = 0;
                    ++j;
                }
                loaded_count++;
            }
        }
    }

    if (loaded_count < pop_size)
    {
        int start_idx = loaded_count;
        if (start_idx == 0)
        {
            // Initial individual [0] is all 0s already.
            std::fill(genes.begin() + n_var, genes.begin() + 2 * n_var, 1);
            start_idx = 2;
        }
        else if (start_idx == 1)
        {
            std::fill(genes.begin() + n_var, genes.begin() + 2 * n_var, 1);
            start_idx = 2;
        }

        int remaining = pop_size - start_idx;
        for (int i = start_idx; i < pop_size; ++i)
        {
            // Evenly distribute probabilities for the newly generated batch between 0 and 1
            double prob = (remaining > 1) ? (double)(i - start_idx) / (remaining - 1) : 0.5;

            std::bernoulli_distribution dist(prob);
            for (int j = 0; j < n_var; ++j)
            {
                if (dist(gen))
                    genes[i * n_var + j] = 1;
            }
        }
    }

    // Scramble population using Fisher-Yates across chunk sizes of n_var
    for (int i = pop_size - 1; i > 0; --i)
    {
        std::uniform_int_distribution<int> swap_dist(0, i);
        int j = swap_dist(gen);
        if (i != j)
        {
            std::swap_ranges(genes.begin() + i * n_var,
                             genes.begin() + (i + 1) * n_var,
                             genes.begin() + j * n_var);
        }
    }
}

void NSGA2::generate_offspring()
{
    std::uniform_int_distribution<int> t_dist(0, pop_size - 1);
    std::bernoulli_distribution cross_dist(0.5);
    std::uniform_int_distribution<int> bit_dist(0, n_var - 1);

    int off_offset = pop_size;

    for (int i = 0; i < pop_size; i += 2)
    {
        int p1 = t_dist(gen), p2 = t_dist(gen);
        int p1_idx = (rank[p1] < rank[p2] || (rank[p1] == rank[p2] && crowding[p1] > crowding[p2])) ? p1 : p2;

        int p3 = t_dist(gen), p4 = t_dist(gen);
        int p2_idx = (rank[p3] < rank[p4] || (rank[p3] == rank[p4] && crowding[p3] > crowding[p4])) ? p3 : p4;

        int o1 = off_offset + i;
        int o2 = off_offset + i + 1;

        // Uniform Crossover (gene-by-gene)
        for (int b = 0; b < n_var; ++b)
        {
            bool swap = cross_dist(gen);
            genes[o1 * n_var + b] = swap ? genes[p1_idx * n_var + b] : genes[p2_idx * n_var + b];
            if (o2 < 2 * pop_size)
            {
                genes[o2 * n_var + b] = swap ? genes[p2_idx * n_var + b] : genes[p1_idx * n_var + b];
            }
        }

        // Single Gene Mutation
        int bit1 = bit_dist(gen);
        genes[o1 * n_var + bit1] ^= 1;

        if (o2 < 2 * pop_size)
        {
            int bit2 = bit_dist(gen);
            genes[o2 * n_var + bit2] ^= 1;
        }
    }
}

void NSGA2::survival(int num_inds)
{
    std::vector<std::vector<int>> fronts(1);
    std::vector<int> domination_count(num_inds, 0);
    std::vector<std::vector<int>> dominates(num_inds);

    for (int i = 0; i < num_inds; ++i)
    {
        for (int j = i + 1; j < num_inds; ++j)
        {
            double cost_i = cost_sse[i * 2], sse_i = cost_sse[i * 2 + 1];
            double cost_j = cost_sse[j * 2], sse_j = cost_sse[j * 2 + 1];

            bool i_dom_j = (cost_i <= cost_j && sse_i <= sse_j) &&
                           (cost_i < cost_j || sse_i < sse_j);

            bool j_dom_i = (cost_j <= cost_i && sse_j <= sse_i) &&
                           (cost_j < cost_i || sse_j < sse_i);

            if (i_dom_j)
            {
                dominates[i].push_back(j);
                domination_count[j]++;
            }
            else if (j_dom_i)
            {
                dominates[j].push_back(i);
                domination_count[i]++;
            }
        }
        if (domination_count[i] == 0)
        {
            rank[i] = 0;
            fronts[0].push_back(i);
        }
    }

    int i = 0;
    while (i < (int)fronts.size())
    {
        std::vector<int> next_front;
        for (int p : fronts[i])
        {
            for (int q : dominates[p])
            {
                domination_count[q]--;
                if (domination_count[q] == 0)
                {
                    rank[q] = i + 1;
                    next_front.push_back(q);
                }
            }
        }

        if (next_front.empty())
            break;

        fronts.push_back(next_front);
        i++;
    }

    std::vector<int> next_gen_indices;
    next_gen_indices.reserve(pop_size);

    for (auto &front : fronts)
    {
        if (front.empty())
            continue;

        for (int idx : front)
            crowding[idx] = 0.0;

        for (int m = 0; m < 2; ++m)
        {
            std::sort(front.begin(), front.end(), [&](int a, int b)
                      { return (m == 0) ? cost_sse[a * 2] < cost_sse[b * 2] : cost_sse[a * 2 + 1] < cost_sse[b * 2 + 1]; });

            crowding[front.front()] = INFINITY;
            crowding[front.back()] = INFINITY;

            double min_val = (m == 0) ? cost_sse[front.front() * 2] : cost_sse[front.front() * 2 + 1];
            double max_val = (m == 0) ? cost_sse[front.back() * 2] : cost_sse[front.back() * 2 + 1];

            if (max_val - min_val <= 1e-9)
                continue;

            for (size_t j = 1; j < front.size() - 1; ++j)
            {
                double diff = (m == 0) ? cost_sse[front[j + 1] * 2] - cost_sse[front[j - 1] * 2]
                                       : cost_sse[front[j + 1] * 2 + 1] - cost_sse[front[j - 1] * 2 + 1];
                crowding[front[j]] += diff / (max_val - min_val);
            }
        }

        if (next_gen_indices.size() + front.size() <= (size_t)pop_size)
        {
            for (int idx : front)
                next_gen_indices.push_back(idx);
        }
        else
        {
            std::sort(front.begin(), front.end(), [&](int a, int b)
                      { return crowding[a] > crowding[b]; });
            int needed = pop_size - next_gen_indices.size();
            for (int j = 0; j < needed; ++j)
                next_gen_indices.push_back(front[j]);
            break;
        }
    }

    // Directly reorder into a consistent layout for the next generation
    std::vector<char> next_genes(pop_size * n_var);
    std::vector<double> next_cost_sse(pop_size * 2);
    std::vector<int> next_rank(pop_size);
    std::vector<double> next_crowding(pop_size);

    for (int k = 0; k < pop_size; ++k)
    {
        int old_idx = next_gen_indices[k];
        std::copy(genes.begin() + old_idx * n_var, genes.begin() + (old_idx + 1) * n_var, next_genes.begin() + k * n_var);
        next_cost_sse[k * 2] = cost_sse[old_idx * 2];
        next_cost_sse[k * 2 + 1] = cost_sse[old_idx * 2 + 1];
        next_rank[k] = rank[old_idx];
        next_crowding[k] = crowding[old_idx];
    }

    // Deposit updated info cleanly
    std::copy(next_genes.begin(), next_genes.end(), genes.begin());
    std::copy(next_cost_sse.begin(), next_cost_sse.end(), cost_sse.begin());
    std::copy(next_rank.begin(), next_rank.end(), rank.begin());
    std::copy(next_crowding.begin(), next_crowding.end(), crowding.begin());
}

void NSGA2::save_pareto(const std::string &prefix)
{
    std::vector<int> pareto_front;
    // We only need to check the first pop_size since these are our survivors.
    for (int i = 0; i < pop_size; ++i)
    {
        if (rank[i] == 0)
        {
            pareto_front.push_back(i);
        }
    }

    std::sort(pareto_front.begin(), pareto_front.end(), [&](int a, int b)
              {
        if (std::abs(cost_sse[a * 2] - cost_sse[b * 2]) > 1e-9) 
            return cost_sse[a * 2] < cost_sse[b * 2];
        return cost_sse[a * 2 + 1] < cost_sse[b * 2 + 1]; });

    std::ofstream f_obj(prefix + "_objectives.csv");
    std::ofstream f_pop(prefix + "_population.csv");

    if (!f_obj.is_open() || !f_pop.is_open())
        return;

    for (int idx : pareto_front)
    {
        f_obj << cost_sse[idx * 2] << "," << cost_sse[idx * 2 + 1] << "\n";

        for (int i = 0; i < n_var; ++i)
        {
            f_pop << (genes[idx * n_var + i] ? "1" : "0") << (i == n_var - 1 ? "" : ",");
        }
        f_pop << "\n";
    }
}