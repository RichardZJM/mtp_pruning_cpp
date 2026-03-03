#pragma once
#include "core.h"
#include <vector>

extern "C" void dposv_(const char *uplo, const int *n, const int *nrhs,
                       double *a, const int *lda, double *b, const int *ldb, int *info);

class SSECalculator
{
    std::vector<double> xtwx, xtwy;
    double ytwy, base_sse;
    int n_features, n_species;

public:
    SSECalculator(const std::vector<double> &xtwx_, const std::vector<double> &xtwy_,
                  double ytwy_, double reg, int n_species_, int n_var, int rank);
    double calculate(const Individual &ind) const;
};

class CostCalculator
{
    int num_moments, n_ranks, n_mus, neigh_count, radial_basis_size;
    std::vector<int> scalar_indices, basic_indices, parents_data, parents_idx;
    double base_cost;

public:
    CostCalculator(int num_moments_, const std::vector<int> &basic_, const std::vector<int> &times_,
                   const std::vector<int> &scalar_, int neigh, int radial, int rank);
    double calculate(const Individual &ind, int n_var) const;
};