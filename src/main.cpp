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

/**
 * Helper function to evaluate a specific group of individuals using MPI.
 * Used for both the initial population and subsequent offspring batches.
 */
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
        count = group.size();
    }

#ifdef USE_MPI
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    if (count == 0)
        return;

    int bytes_per_ind = (n_var + 7) / 8;
    // Calculate chunk size ensuring coverage of all individuals
    int chunk_size = (count + mpi_size - 1) / mpi_size;

    std::vector<uint8_t> send_buf;
    std::vector<double> recv_results(chunk_size * 2);
    std::vector<uint8_t> recv_bits(chunk_size * bytes_per_ind);

    // Root prepares the flattened bit buffer
    if (mpi_rank == 0)
    {
        send_buf.resize(chunk_size * mpi_size * bytes_per_ind, 0); // Padding with 0s
        for (int i = 0; i < count; ++i)
        {
            std::copy(group[i].bits.begin(), group[i].bits.end(), send_buf.begin() + i * bytes_per_ind);
        }
    }

#ifdef USE_MPI
    MPI_Scatter(send_buf.data(), chunk_size * bytes_per_ind, MPI_BYTE,
                recv_bits.data(), chunk_size * bytes_per_ind, MPI_BYTE,
                0, MPI_COMM_WORLD);
#else
    recv_bits = send_buf;
#endif

    // Workers evaluate their assigned chunk
    for (int i = 0; i < chunk_size; ++i)
    {
        int global_idx = mpi_rank * chunk_size + i;
        if (global_idx >= count)
            break; // Skip padding

        Individual ind;
        auto start = recv_bits.begin() + i * bytes_per_ind;
        ind.bits.assign(start, start + bytes_per_ind);

        // Perform Calculation
        double c = cost_calc.calculate(ind, n_var);
        double s = sse_calc.calculate(ind);

        recv_results[i * 2] = c;
        recv_results[i * 2 + 1] = s;
    }

    // Gather results back to root
    std::vector<double> all_results;
    if (mpi_rank == 0)
    {
        all_results.resize(chunk_size * mpi_size * 2);
    }

#ifdef USE_MPI
    MPI_Gather(recv_results.data(), chunk_size * 2, MPI_DOUBLE,
               all_results.data(), chunk_size * 2, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
#else
    all_results = recv_results;
#endif

    // Root updates the individual objects
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
    if (rank == 0)
    {
        std::ifstream f(argv[1]);
        if (!f.is_open())
        {
            std::cerr << "Could not open " << argv[1] << "\n";
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }
        config = json::parse(f);
    }

#ifdef USE_MPI
    std::string config_str;
    if (rank == 0)
        config_str = config.dump();
    int len = config_str.size();
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    config_str.resize(len);
    MPI_Bcast(&config_str[0], len, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0)
        config = json::parse(config_str);
#endif

    int pop_size = config["pop_size"];
    int n_gen = config["n_gen"];

    if (rank == 0)
        std::cout << "Loading MTP and binary data...\n";

    // --- 2. Load Data ---
    MTPData mtp = parse_mtp(config["mtp_file"]);
    auto xtwx = read_binary<double>(config["xtwx_file"]);
    auto xtwy = read_binary<double>(config["xtwy_file"]);

    int n_var = mtp.alpha_scalar_moments;

    // --- 3. Initialize Evaluators ---
    SSECalculator sse_calc(xtwx, xtwy, config["ytwy"], config.value("regularization", 0.0),
                           mtp.species_count, n_var, rank);

    CostCalculator cost_calc(mtp.alpha_moments_count, mtp.alpha_index_basic, mtp.alpha_index_times,
                             mtp.alpha_moment_mapping, config["neigh_count"], mtp.radial_basis_size, rank);

    NSGA2 ga(pop_size, n_var, 42 + rank);
    std::vector<Individual> pop;

    auto start_time = std::chrono::high_resolution_clock::now();

    // --- 4. Initialization & Gen 0 Evaluation ---
    if (rank == 0)
    {
        ga.initialize_population(pop);
        std::cout << "Evaluating initial population..." << std::endl;
    }

    // Evaluate the initial population (Parents for Gen 0)
    evaluate_population(pop, n_var, rank, size, cost_calc, sse_calc);

    // Perform initial ranking/sorting so parents have valid rank/crowding for selection
    if (rank == 0)
    {
        std::vector<Individual> empty_offspring;
        ga.survival(pop, empty_offspring);
    }

    // --- 5. Main Optimization Loop ---
    for (int gen = 0; gen < n_gen; ++gen)
    {
        std::vector<Individual> offspring;

        // A. Generate Offspring (Rank 0 only)
        if (rank == 0)
        {
            ga.generate_offspring(pop, offspring);
        }

        // B. Evaluate ONLY the new Offspring
        // Note: Parents in 'pop' retain their costs from the previous generation
        evaluate_population(offspring, n_var, rank, size, cost_calc, sse_calc);

        // C. Survival / Elitism (Rank 0 only)
        if (rank == 0)
        {
            // Merges Parents (pop) + Offspring, sorts, and selects best N into 'pop'
            ga.survival(pop, offspring);

            if ((gen + 1) % 10 == 0 || gen == 0)
            {
                std::cout << "Generation " << gen + 1 << "/" << n_gen << " complete." << std::endl;
            }
        }
    }

    // --- 6. Save Results ---
    if (rank == 0)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::filesystem::path output_dir = std::filesystem::path(config["out_dir"]);
        try
        {
            std::filesystem::create_directories(output_dir);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error creating directory: " << e.what() << "\n";
        }

        std::filesystem::path out_path = output_dir / "pareto_final";

        // Save Pareto Front (filters for rank == 0)
        ga.save_pareto(pop, out_path.string());

        std::cout << "Optimization finished in " << elapsed.count() << " seconds.\n";
        std::cout << "Results saved to " << out_path.string() << "_*.csv\n";
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}