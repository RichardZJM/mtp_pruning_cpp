#pragma once
#include <vector>
#include <queue>

extern "C" void dposv_(const char *uplo, const int *n, const int *nrhs,
                       double *a, const int *lda, double *b, const int *ldb, int *info);

class SSECalculator
{
    std::vector<double> xtwx, xtwy;
    double ytwy, base_sse;
    int n_features, n_species;

    // Pre-allocated buffers to prevent re-allocation
    mutable std::vector<int> active_buf;
    mutable std::vector<double> A_buf;
    mutable std::vector<double> B_buf;

public:
    SSECalculator(const std::vector<double> &xtwx_, const std::vector<double> &xtwy_,
                  double ytwy_, double reg, int n_species_, int n_var, int rank);
    double calculate(const char *genes) const;
};

class CostCalculator
{
    int num_moments, n_ranks, n_mus, radial_basis_size;
    std::vector<int> scalar_indices, basic_indices, parents_data, parents_idx;
    double base_cost, neigh_count;

    // Pre-allocated buffers (using char to avoid vector<bool> overhead)
    mutable std::vector<char> mus_flags_buf;
    mutable std::vector<char> rank_flags_buf;
    mutable std::vector<char> to_preserve_buf;
    mutable std::queue<int> q_buf;

public:
    CostCalculator(int num_moments_, const std::vector<int> &basic_, const std::vector<int> &times_,
                   const std::vector<int> &scalar_, double neigh_count, int radial, int rank);
    double calculate(const char *genes, int n_var) const;
    void canonicalize(char *genes, int n_var) const;
};