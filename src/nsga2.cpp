#include "nsga2.h"
#include <algorithm>
#include <iostream>
#include <cassert>

NSGA2::NSGA2(int pop_size_, int n_var_, int seed) : pop_size(pop_size_), n_var(n_var_), gen(seed) {}

void NSGA2::initialize_population(std::vector<Individual> &pop)
{
    assert(pop_size >= 2 && "Population size must be at least 2.");
    pop.resize(pop_size);

    for (int i = 0; i < pop_size; ++i)
    {
        pop[i].genes.assign(n_var, 0);
        if (i == 1)
        {
            std::fill(pop[i].genes.begin(), pop[i].genes.end(), 1);
        }
        else if (i > 1)
        {
            std::bernoulli_distribution dist((double)(i - 1) / (pop_size - 1));
            for (int j = 0; j < n_var; ++j)
            {
                if (dist(gen))
                    pop[i].genes[j] = 1;
            }
        }
    }
}

void NSGA2::generate_offspring(const std::vector<Individual> &parents, std::vector<Individual> &offspring)
{
    offspring.resize(pop_size);
    std::uniform_int_distribution<int> t_dist(0, pop_size - 1);
    std::bernoulli_distribution cross_dist(0.5);
    std::uniform_int_distribution<int> bit_dist(0, n_var - 1);

    for (int i = 0; i < pop_size; i += 2)
    {
        int p1 = t_dist(gen), p2 = t_dist(gen);
        int p1_idx = (parents[p1].rank < parents[p2].rank || (parents[p1].rank == parents[p2].rank && parents[p1].crowding > parents[p2].crowding)) ? p1 : p2;

        int p3 = t_dist(gen), p4 = t_dist(gen);
        int p2_idx = (parents[p3].rank < parents[p4].rank || (parents[p3].rank == parents[p4].rank && parents[p3].crowding > parents[p4].crowding)) ? p3 : p4;

        offspring[i].genes.resize(n_var);
        if (i + 1 < pop_size)
            offspring[i + 1].genes.resize(n_var);

        // Uniform Crossover (gene-by-gene)
        for (int b = 0; b < n_var; ++b)
        {
            bool swap = cross_dist(gen);
            offspring[i].genes[b] = swap ? parents[p1_idx].genes[b] : parents[p2_idx].genes[b];
            if (i + 1 < pop_size)
            {
                offspring[i + 1].genes[b] = swap ? parents[p2_idx].genes[b] : parents[p1_idx].genes[b];
            }
        }

        // Single Gene Mutation
        int bit1 = bit_dist(gen);
        offspring[i].genes[bit1] ^= 1;

        if (i + 1 < pop_size)
        {
            int bit2 = bit_dist(gen);
            offspring[i + 1].genes[bit2] ^= 1;
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

    for (size_t i = 0; i < combined.size(); ++i)
    {
        for (size_t j = i + 1; j < combined.size(); ++j)
        {
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
            break;

        fronts.push_back(next_front);
        i++;
    }

    pop.clear();
    for (auto &front : fronts)
    {
        if (front.empty())
            continue;

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

            if (max_val - min_val <= 1e-9)
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
            f_pop << (ind.get_gene(i) ? "1" : "0") << (i == n_var - 1 ? "" : ",");
        }
        f_pop << "\n";
    }
}