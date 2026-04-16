#include "evaluator.h"
#include <iostream>
#include <set>
#include <cmath>
#include <algorithm>

// --- SSE Calculator ---
SSECalculator::SSECalculator(const std::vector<double> &xtwx_, const std::vector<double> &xtwy_,
                             double ytwy_, double reg, int n_species_, int n_var, int rank)
    : xtwx(xtwx_), xtwy(xtwy_), ytwy(ytwy_), n_species(n_species_)
{
    n_features = n_species + n_var;

    for (int i = 0; i < n_features; ++i)
    {
        xtwx[i * n_features + i] += reg;
    }

    active_buf.reserve(n_features);
    A_buf.resize(n_features * n_features);
    B_buf.resize(n_features);

    std::vector<char> all_ones(n_var, 1);
    base_sse = 1.0;
    base_sse = calculate(all_ones.data());

    if (rank == 0)
    {
        std::cout << "Base SSE: " << base_sse << "\n";
    }
}

double SSECalculator::calculate(const char *genes) const
{
    active_buf.clear();

    for (int i = 0; i < n_species; ++i)
        active_buf.push_back(i);

    int n_var = n_features - n_species;
    for (int i = 0; i < n_var; ++i)
    {
        if (genes[i])
            active_buf.push_back(i + n_species);
    }

    int n = active_buf.size();
    if (n == 0)
        return INFINITY;

    for (int i = 0; i < n; ++i)
    {
        B_buf[i] = xtwy[active_buf[i]];
        for (int j = 0; j < n; j++)
        {
            A_buf[i * n + j] = xtwx[active_buf[i] * n_features + active_buf[j]];
        }
    }

    char uplo = 'U';
    int nrhs = 1, info = 0;
    dposv_(&uplo, &n, &nrhs, A_buf.data(), &n, B_buf.data(), &n, &info);

    if (info > 0)
        return INFINITY;

    double theta_dot_xtwy = 0;
    for (int i = 0; i < n; ++i)
    {
        theta_dot_xtwy += B_buf[i] * xtwy[active_buf[i]];
    }

    return (ytwy - theta_dot_xtwy) / base_sse;
}

// --- Cost Calculator ---
CostCalculator::CostCalculator(int num_moments_, const std::vector<int> &basic_,
                               const std::vector<int> &times_, const std::vector<int> &scalar_,
                               double neigh, int radial, int rank)
    : num_moments(num_moments_), basic_indices(basic_), scalar_indices(scalar_),
      neigh_count(neigh), radial_basis_size(radial)
{
    std::set<int> mus_set, rank_set;
    for (size_t i = 0; i < basic_.size() / 4; ++i)
    {
        mus_set.insert(basic_[i * 4]);
        int r = std::max({basic_[i * 4 + 1], basic_[i * 4 + 2], basic_[i * 4 + 3]});
        rank_set.insert(r);
    }
    n_mus = mus_set.size();
    n_ranks = rank_set.size();

    std::vector<std::vector<int>> py_parents(num_moments);
    for (size_t i = 0; i < times_.size() / 4; ++i)
    {
        int p1 = times_[i * 4], p2 = times_[i * 4 + 1], child = times_[i * 4 + 3];
        py_parents[child].push_back(p1);
        py_parents[child].push_back(p2);
    }

    parents_idx.push_back(0);
    for (int i = 0; i < num_moments; ++i)
    {
        for (int p : py_parents[i])
            parents_data.push_back(p);
        parents_idx.push_back(parents_data.size());
    }

    mus_flags_buf.resize(n_mus);
    rank_flags_buf.resize(n_ranks);
    to_preserve_buf.resize(num_moments);

    std::vector<char> all_ones(scalar_indices.size(), 1);
    base_cost = 1.0;
    base_cost = calculate(all_ones.data(), scalar_indices.size());

    if (rank == 0)
    {
        std::cout << "Base Cost: " << base_cost << "\n";
    }
}

double CostCalculator::calculate(const char *genes, int n_var) const
{
    std::fill(mus_flags_buf.begin(), mus_flags_buf.end(), 0);
    std::fill(rank_flags_buf.begin(), rank_flags_buf.end(), 0);
    std::fill(to_preserve_buf.begin(), to_preserve_buf.end(), 0);

    while (!q_buf.empty())
        q_buf.pop();

    for (int i = 0; i < n_var; ++i)
    {
        if (genes[i])
        {
            int m = scalar_indices[i];
            if (!to_preserve_buf[m])
            {
                to_preserve_buf[m] = 1;
                q_buf.push(m);
            }
        }
    }

    while (!q_buf.empty())
    {
        int child = q_buf.front();
        q_buf.pop();
        for (int j = parents_idx[child]; j < parents_idx[child + 1]; j += 2)
        {
            int p1 = parents_data[j], p2 = parents_data[j + 1];
            if (!to_preserve_buf[p1])
            {
                to_preserve_buf[p1] = 1;
                q_buf.push(p1);
            }
            if (!to_preserve_buf[p2])
            {
                to_preserve_buf[p2] = 1;
                q_buf.push(p2);
            }
        }
    }

    int ntimes = 0, nbasic = 0;
    for (int i = 0; i < num_moments; ++i)
    {
        if (to_preserve_buf[i])
        {
            int edges = (parents_idx[i + 1] - parents_idx[i]) / 2;
            ntimes += edges;
            if (edges == 0)
            {
                nbasic++;
                int mu = basic_indices[i * 4];
                int r = std::max({basic_indices[i * 4 + 1], basic_indices[i * 4 + 2], basic_indices[i * 4 + 3]});
                if (mu < n_mus)
                    mus_flags_buf[mu] = 1;
                if (r < n_ranks)
                    rank_flags_buf[r] = 1;
            }
        }
    }

    int max_rank = 0, mus_count = 0;
    for (char b : rank_flags_buf)
        if (b)
            max_rank++;
    for (char b : mus_flags_buf)
        if (b)
            mus_count++;

    double cost = neigh_count * (24 + 4 * max_rank + 8 * radial_basis_size + 14 +
                                 4 * mus_count * radial_basis_size + 39 * nbasic) +
                  9 * ntimes;

    return cost / base_cost;
}

void CostCalculator::canonicalize(char *genes, int n_var) const
{
    // 1. FREE-RIDE RULE
    // Calculate raw tree exactly ONCE. This populates `to_preserve_buf`, `mus_flags_buf`, and `rank_flags_buf`
    double raw_cost = calculate(genes, n_var);

    for (int i = 0; i < n_var; ++i)
    {
        if (to_preserve_buf[scalar_indices[i]])
        {
            genes[i] = 1;
        }
    }

    // 2. FAST FILL RULE
    // Calculate the absolute allowable threshold relative to the raw tree.
    double max_incremental_cost = 0.10 * (raw_cost * base_cost);

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int i = 0; i < n_var; ++i)
        {
            if (!genes[i])
            {
                int m = scalar_indices[i];

                // Check if all immediate dependencies are strictly met in O(1) checks
                bool deps_met = true;
                for (int j = parents_idx[m]; j < parents_idx[m + 1]; ++j)
                {
                    if (!to_preserve_buf[parents_data[j]])
                    {
                        deps_met = false;
                        break;
                    }
                }

                if (deps_met)
                {
                    // Calculate exact incremental cost without traversing the tree
                    double incremental_cost = 0;
                    int edges = (parents_idx[m + 1] - parents_idx[m]) / 2;

                    if (edges > 0)
                    {
                        incremental_cost = 9 * edges;
                    }
                    else
                    {
                        incremental_cost = neigh_count * 39;
                        int mu = basic_indices[m * 4];
                        int r = std::max({basic_indices[m * 4 + 1], basic_indices[m * 4 + 2], basic_indices[m * 4 + 3]});

                        if (mu < n_mus && !mus_flags_buf[mu])
                            incremental_cost += neigh_count * 4 * radial_basis_size;
                        if (r < n_ranks && !rank_flags_buf[r])
                            incremental_cost += neigh_count * 4;
                    }

                    if (incremental_cost <= max_incremental_cost)
                    {
                        // Unionize: apply the gene and update the state arrays immediately
                        genes[i] = 1;
                        to_preserve_buf[m] = 1;

                        if (edges == 0)
                        {
                            int mu = basic_indices[m * 4];
                            int r = std::max({basic_indices[m * 4 + 1], basic_indices[m * 4 + 2], basic_indices[m * 4 + 3]});
                            if (mu < n_mus)
                                mus_flags_buf[mu] = 1;
                            if (r < n_ranks)
                                rank_flags_buf[r] = 1;
                        }

                        // We loop again in case newly accepted nodes trigger other nodes' dependencies
                        changed = true;
                    }
                }
            }
        }
    }
}