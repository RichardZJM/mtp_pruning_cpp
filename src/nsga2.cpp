#include "nsga2.h"
#include <algorithm>
#include <iostream>

NSGA2::NSGA2(int pop_size_, int n_var_, int seed) : pop_size(pop_size_), n_var(n_var_), gen(seed) {}

void NSGA2::initialize_population(std::vector<Individual> &pop)
{
    pop.resize(pop_size);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    int bytes = (n_var + 7) / 8;

    for (int i = 0; i < pop_size; ++i)
    {
        pop[i].bits.resize(bytes);
        if (i == 0)
            std::fill(pop[i].bits.begin(), pop[i].bits.end(), 0);
        else if (i == 1)
            std::fill(pop[i].bits.begin(), pop[i].bits.end(), 0xFF);
        else
        {
            for (int b = 0; b < bytes; ++b)
                pop[i].bits[b] = dist(gen);
        }
    }
}

void NSGA2::generate_offspring(const std::vector<Individual> &parents, std::vector<Individual> &offspring)
{
    offspring.resize(pop_size);
    std::uniform_int_distribution<int> t_dist(0, pop_size - 1);
    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    std::uniform_int_distribution<int> bit_dist(0, n_var - 1); // Exact single bitflip

    for (int i = 0; i < pop_size; i += 2)
    {
        int p1 = t_dist(gen), p2 = t_dist(gen);
        int p1_idx = (parents[p1].rank < parents[p2].rank || (parents[p1].rank == parents[p2].rank && parents[p1].crowding > parents[p2].crowding)) ? p1 : p2;

        int p3 = t_dist(gen), p4 = t_dist(gen);
        int p2_idx = (parents[p3].rank < parents[p4].rank || (parents[p3].rank == parents[p4].rank && parents[p3].crowding > parents[p4].crowding)) ? p3 : p4;

        offspring[i].bits.resize(parents[0].bits.size());
        if (i + 1 < pop_size)
            offspring[i + 1].bits.resize(parents[0].bits.size());

        // Uniform Crossover
        for (size_t b = 0; b < parents[0].bits.size(); ++b)
        {
            uint8_t mask = byte_dist(gen);
            offspring[i].bits[b] = (parents[p1_idx].bits[b] & mask) | (parents[p2_idx].bits[b] & ~mask);
            if (i + 1 < pop_size)
            {
                offspring[i + 1].bits[b] = (parents[p2_idx].bits[b] & mask) | (parents[p1_idx].bits[b] & ~mask);
            }
        }

        // Single Bit Mutation
        int bit1 = bit_dist(gen);
        offspring[i].bits[bit1 / 8] ^= (1 << (bit1 % 8));

        if (i + 1 < pop_size)
        {
            int bit2 = bit_dist(gen);
            offspring[i + 1].bits[bit2 / 8] ^= (1 << (bit2 % 8));
        }
    }
}

void NSGA2::survival(std::vector<Individual> &pop, std::vector<Individual> &offspring)
{
    std::vector<Individual> combined = pop;
    combined.insert(combined.end(), offspring.begin(), offspring.end());

    std::vector<std::vector<int>> fronts(1);
    std::vector<int> domination_count(combined.size(), 0);
    std::vector<std::vector<int>> dominates(combined.size());

    // Non-dominated sort
    for (size_t i = 0; i < combined.size(); ++i)
    {
        for (size_t j = i + 1; j < combined.size(); ++j)
        {
            // FIX 1: Fixed copy-paste error in j_dom_i (sse <= sse)
            bool i_dom_j = (combined[i].cost <= combined[j].cost && combined[i].sse <= combined[j].sse) &&
                           (combined[i].cost < combined[j].cost || combined[i].sse < combined[j].sse);

            bool j_dom_i = (combined[j].cost <= combined[i].cost && combined[j].sse <= combined[i].sse) &&
                           (combined[j].cost < combined[i].cost || combined[j].sse < combined[i].sse);

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
            combined[i].rank = 0;
            fronts[0].push_back(i);
        }
    }

    // FIX 2: Fixed crash by changing loop condition and break logic
    int i = 0;
    while (i < fronts.size())
    {
        std::vector<int> next_front;
        for (int p : fronts[i])
        {
            for (int q : dominates[p])
            {
                domination_count[q]--;
                if (domination_count[q] == 0)
                {
                    combined[q].rank = i + 1;
                    next_front.push_back(q);
                }
            }
        }

        if (next_front.empty())
            break; // Stop before incrementing i

        fronts.push_back(next_front);
        i++;
    }

    pop.clear();
    for (auto &front : fronts)
    {
        if (front.empty())
            continue;

        // Crowding Distance
        for (int idx : front)
            combined[idx].crowding = 0.0;

        for (int m = 0; m < 2; ++m)
        {
            std::sort(front.begin(), front.end(), [&](int a, int b)
                      { return (m == 0) ? combined[a].cost < combined[b].cost : combined[a].sse < combined[b].sse; });

            combined[front.front()].crowding = INFINITY;
            combined[front.back()].crowding = INFINITY;

            double min_val = (m == 0) ? combined[front.front()].cost : combined[front.front()].sse;
            double max_val = (m == 0) ? combined[front.back()].cost : combined[front.back()].sse;

            if (max_val - min_val <= 1e-9) // Added small epsilon safety
                continue;

            for (size_t j = 1; j < front.size() - 1; ++j)
            {
                double diff = (m == 0) ? combined[front[j + 1]].cost - combined[front[j - 1]].cost
                                       : combined[front[j + 1]].sse - combined[front[j - 1]].sse;
                combined[front[j]].crowding += diff / (max_val - min_val);
            }
        }

        if (pop.size() + front.size() <= (size_t)pop_size)
        {
            for (int idx : front)
                pop.push_back(combined[idx]);
        }
        else
        {
            std::sort(front.begin(), front.end(), [&](int a, int b)
                      { return combined[a].crowding > combined[b].crowding; });
            int needed = pop_size - pop.size();
            for (int j = 0; j < needed; ++j)
                pop.push_back(combined[front[j]]);
            break;
        }
    }
}

void NSGA2::save_pareto(const std::vector<Individual> &pop, const std::string &prefix)
{
    std::vector<Individual> pareto_front;
    for (const auto &ind : pop)
    {
        if (ind.rank == 0)
        {
            pareto_front.push_back(ind);
        }
    }

    std::sort(pareto_front.begin(), pareto_front.end(), [](const Individual &a, const Individual &b)
              {
        if (std::abs(a.cost - b.cost) > 1e-9) return a.cost < b.cost;
        return a.sse < b.sse; });

    std::ofstream f_obj(prefix + "_objectives.csv");
    std::ofstream f_pop(prefix + "_population.csv");

    if (!f_obj.is_open() || !f_pop.is_open())
        return;

    for (const auto &ind : pareto_front)
    {
        f_obj << ind.cost << "," << ind.sse << "\n";

        for (int i = 0; i < n_var; ++i)
        {
            f_pop << (ind.get_bit(i) ? "1" : "0") << (i == n_var - 1 ? "" : ",");
        }
        f_pop << "\n";
    }
}