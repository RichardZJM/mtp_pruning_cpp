#include "json.hpp"
#include "evaluator.h"
#include "nsga2.h"
#include "mtp_parser.h"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include <algorithm>

#ifdef USE_MPI
#include <mpi.h>
#endif

using json = nlohmann::json;

static void work_range(int count, int mpi_rank, int mpi_size,
                       int &start, int &end)
{
    int base = count / mpi_size;
    int extra = count % mpi_size;
    if (mpi_rank < extra)
    {
        start = mpi_rank * (base + 1);
        end = start + base + 1;
    }
    else
    {
        start = extra * (base + 1) + (mpi_rank - extra) * base;
        end = start + base;
    }
}

void evaluate_population(std::vector<Individual> &group,
                         int n_var,
                         int mpi_rank,
                         int mpi_size,
                         CostCalculator &cost_calc,
                         SSECalculator &sse_calc)
{
    int count = 0;
    if (mpi_rank == 0)
    {
        count = static_cast<int>(group.size());
    }

#ifdef USE_MPI
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    if (count == 0)
        return;

    int genes_per_ind = n_var;

    // --- 1. Broadcast all genes to every rank ---
    int total_genes = count * genes_per_ind;
    std::vector<char> all_genes(total_genes, 0);

    if (mpi_rank == 0)
    {
        for (int i = 0; i < count; ++i)
        {
            std::copy(group[i].genes.begin(),
                      group[i].genes.end(),
                      all_genes.begin() + i * genes_per_ind);
        }
    }

#ifdef USE_MPI
    MPI_Bcast(all_genes.data(), total_genes, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

    // --- 2. Each rank evaluates its own slice ---
    int my_start, my_end;
    work_range(count, mpi_rank, mpi_size, my_start, my_end);
    int my_count = my_end - my_start;

    std::vector<double> local_results(my_count * 2);

    for (int i = 0; i < my_count; ++i)
    {
        int global_idx = my_start + i;

        Individual ind;
        auto src = all_genes.begin() + global_idx * genes_per_ind;
        ind.genes.assign(src, src + genes_per_ind);

        double c = cost_calc.calculate(ind, n_var);
        double s = sse_calc.calculate(ind);

        local_results[i * 2] = c;
        local_results[i * 2 + 1] = s;
    }

    // --- 3. Gatherv results back to root ---
#ifdef USE_MPI
    std::vector<int> recv_counts(mpi_size);
    std::vector<int> displs(mpi_size);

    for (int r = 0; r < mpi_size; ++r)
    {
        int rs, re;
        work_range(count, r, mpi_size, rs, re);
        recv_counts[r] = (re - rs) * 2;
    }
    displs[0] = 0;
    for (int r = 1; r < mpi_size; ++r)
    {
        displs[r] = displs[r - 1] + recv_counts[r - 1];
    }

    std::vector<double> all_results;
    if (mpi_rank == 0)
    {
        all_results.resize(count * 2);
    }

    MPI_Gatherv(local_results.data(), my_count * 2, MPI_DOUBLE,
                mpi_rank == 0 ? all_results.data() : nullptr,
                recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
#else
    std::vector<double> &all_results = local_results;
#endif

    // --- 4. Root writes results back into the Individual objects ---
    if (mpi_rank == 0)
    {
        for (int i = 0; i < count; ++i)
        {
            group[i].cost = all_results[i * 2];
            group[i].sse = all_results[i * 2 + 1];
        }
    }
}

int main(int argc, char **argv)
{
    int rank = 0, size = 1;
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    if (argc < 2)
    {
        if (rank == 0)
            std::cerr << "Usage: ./bin/prune <config.json>\n";
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    // --- 1. Config Parsing ---
    json config;
    {
        int open_ok = 0;
        if (rank == 0)
        {
            std::ifstream f(argv[1]);
            open_ok = f.is_open() ? 1 : 0;
#ifdef USE_MPI
            MPI_Bcast(&open_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
            if (!open_ok)
            {
                std::cerr << "Could not open " << argv[1] << "\n";
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return 1;
            }
            config = json::parse(f);
        }
        else
        {
#ifdef USE_MPI
            MPI_Bcast(&open_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!open_ok)
            {
                MPI_Finalize();
                return 1;
            }
#endif
        }
    }

#ifdef USE_MPI
    std::string config_str;
    if (rank == 0)
        config_str = config.dump();
    int len = static_cast<int>(config_str.size());
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    config_str.resize(len);
    MPI_Bcast(&config_str[0], len, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0)
        config = json::parse(config_str);
#endif

    int pop_size = config["pop_size"];
    int n_gen = config["n_gen"];
    double time_limit = config.value("time", -1.0);
    int save_interval = config.value("save_interval", -1);

    // --- 2. Load Data (AVOIDING I/O STORMS) ---
    MTPData mtp;
    std::vector<double> xtwx, xtwy;

    if (rank == 0)
    {
        std::cout << "Loading MTP and binary data...\n";
        mtp = parse_mtp(config["mtp_file"]);
        xtwx = read_binary<double>(config["xtwx_file"]);
        xtwy = read_binary<double>(config["xtwy_file"]);
    }

#ifdef USE_MPI
    MPI_Bcast(&mtp.species_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mtp.radial_basis_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mtp.alpha_moments_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mtp.alpha_scalar_moments, 1, MPI_INT, 0, MPI_COMM_WORLD);

    auto bcast_int_vec = [](std::vector<int> &vec, int r)
    {
        int sz = vec.size();
        MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (r != 0)
            vec.resize(sz);
        if (sz > 0)
            MPI_Bcast(vec.data(), sz, MPI_INT, 0, MPI_COMM_WORLD);
    };
    auto bcast_double_vec = [](std::vector<double> &vec, int r)
    {
        int sz = vec.size();
        MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (r != 0)
            vec.resize(sz);
        if (sz > 0)
            MPI_Bcast(vec.data(), sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    };

    bcast_int_vec(mtp.alpha_index_basic, rank);
    bcast_int_vec(mtp.alpha_index_times, rank);
    bcast_int_vec(mtp.alpha_moment_mapping, rank);
    bcast_double_vec(xtwx, rank);
    bcast_double_vec(xtwy, rank);
#endif

    int n_var = mtp.alpha_scalar_moments;

    // --- 3. Initialize Evaluators ---
    SSECalculator sse_calc(xtwx, xtwy, config["ytwy"],
                           config.value("regularization", 0.0),
                           mtp.species_count, n_var, rank);

    CostCalculator cost_calc(mtp.alpha_moments_count, mtp.alpha_index_basic,
                             mtp.alpha_index_times, mtp.alpha_moment_mapping,
                             config["neigh_count"], mtp.radial_basis_size, rank);

    NSGA2 ga(pop_size, n_var, 42 + rank);
    std::vector<Individual> pop;

    std::filesystem::path output_dir =
        std::filesystem::path(config["out_dir"].get<std::string>());
    if (rank == 0)
    {
        try
        {
            std::filesystem::create_directories(output_dir);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error creating directory: " << e.what() << "\n";
        }
    }

    // --- 4. Initialization & Gen 0 Evaluation ---
    if (rank == 0)
    {
        ga.initialize_population(pop);
        std::cout << "Evaluating initial population..." << std::endl;
    }

    evaluate_population(pop, n_var, rank, size, cost_calc, sse_calc);

    if (rank == 0)
    {
        std::vector<Individual> empty_offspring;
        ga.survival(pop, empty_offspring);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    auto elapsed_s = [&]()
    {
        return std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - start_time)
            .count();
    };

    // --- 5. Main Optimization Loop ---
    bool time_limit_reached = false;

    for (int gen = 0; gen < n_gen; ++gen)
    {
        std::vector<Individual> offspring;

        if (rank == 0)
        {
            ga.generate_offspring(pop, offspring);
        }

        evaluate_population(offspring, n_var, rank, size, cost_calc, sse_calc);

        if (rank == 0)
        {
            ga.survival(pop, offspring);

            int completed_gen = gen + 1;
            if (completed_gen % 10 == 0 || gen == 0)
            {
                std::cout << "Generation " << completed_gen << "/" << n_gen
                          << " | Elapsed: " << elapsed_s() << "s" << std::endl;
            }

            if (save_interval > 0 && completed_gen % save_interval == 0)
            {
                std::string prefix =
                    (output_dir / ("pareto_" + std::to_string(completed_gen)))
                        .string();
                std::cout << "Saving intermediate results at generation "
                          << completed_gen << "..." << std::endl;
                ga.save_pareto(pop, prefix);
            }
        }

        int stop = 0;
        if (rank == 0 && time_limit > 0.0 && elapsed_s() >= time_limit)
        {
            std::cout << "Time limit of " << time_limit
                      << "s reached after generation " << gen + 1
                      << ". Stopping early.\n";
            stop = 1;
            time_limit_reached = true;
        }

#ifdef USE_MPI
        MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

        if (stop)
            break;
    }

    // --- 6. Save Final Results ---
    if (rank == 0)
    {
        std::filesystem::path out_path = output_dir / "pareto_final";
        ga.save_pareto(pop, out_path.string());

        std::cout << "Optimization finished in " << elapsed_s() << "s";
        if (time_limit_reached)
            std::cout << " (stopped by time limit)";
        std::cout << ".\n";
        std::cout << "Results saved to " << out_path.string() << "_*.csv\n";
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}