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

    Individual all_ones;
    all_ones.genes.resize(n_var, 1);
    base_sse = 1.0;
    base_sse = calculate(all_ones);

    if (rank == 0)
    {
        std::cout << "Base SSE: " << base_sse << "\n";
    }
}

double SSECalculator::calculate(const Individual &ind) const
{
    active_buf.clear();

    for (int i = 0; i < n_species; ++i)
        active_buf.push_back(i);

    int n_var = n_features - n_species;
    for (int i = 0; i < n_var; ++i)
    {
        if (ind.get_gene(i))
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
                               int neigh, int radial, int rank)
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

    Individual all_ones;
    all_ones.genes.resize(scalar_indices.size(), 1);
    base_cost = 1.0;
    base_cost = calculate(all_ones, scalar_indices.size());

    if (rank == 0)
    {
        std::cout << "Base Cost: " << base_cost << "\n";
    }
}

double CostCalculator::calculate(const Individual &ind, int n_var) const
{
    std::fill(mus_flags_buf.begin(), mus_flags_buf.end(), 0);
    std::fill(rank_flags_buf.begin(), rank_flags_buf.end(), 0);
    std::fill(to_preserve_buf.begin(), to_preserve_buf.end(), 0);

    while (!q_buf.empty())
        q_buf.pop();

    for (int i = 0; i < n_var; ++i)
    {
        if (ind.get_gene(i))
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