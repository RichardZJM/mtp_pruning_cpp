#include "json.hpp"
#include "evaluator.h"
#include "nsga2.h"
#include "mtp_parser.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <dlfcn.h>

using json = nlohmann::json;

std::string find_latest_population(const std::filesystem::path &dir)
{
    if (!std::filesystem::exists(dir))
        return "";

    std::string final_pop = (dir / "pareto_final_population.csv").string();
    if (std::filesystem::exists(final_pop))
    {
        return final_pop;
    }

    int max_gen = -1;
    std::string latest_pop = "";
    for (const auto &entry : std::filesystem::directory_iterator(dir))
    {
        if (entry.is_regular_file())
        {
            std::string filename = entry.path().filename().string();
            if (filename.rfind("pareto_", 0) == 0 && filename.find("_population.csv") != std::string::npos)
            {
                size_t p1 = 7;
                size_t p2 = filename.find("_population.csv");
                if (p2 != std::string::npos && p2 > p1)
                {
                    try
                    {
                        int gen = std::stoi(filename.substr(p1, p2 - p1));
                        if (gen > max_gen)
                        {
                            max_gen = gen;
                            latest_pop = entry.path().string();
                        }
                    }
                    catch (...)
                    {
                    }
                }
            }
        }
    }
    return latest_pop;
}

void evaluate_population(int offset,
                         int count,
                         int n_var,
                         int mpi_rank,
                         int mpi_size,
                         CostCalculator &cost_calc,
                         SSECalculator &sse_calc,
                         std::vector<char> &genes,
                         std::vector<double> &cost_sse,
                         std::vector<char> &local_genes,
                         std::vector<double> &local_results,
                         double &eval_time,
                         double &mpi_time,
                         double &gather_time)
{
    if (count == 0)
        return;

    int local_count = count / mpi_size;
    int genes_per_ind = n_var;

#ifdef USE_MPI
    double t0 = 0;
    if (mpi_rank == 0)
        t0 = MPI_Wtime();

    MPI_Scatter(mpi_rank == 0 ? genes.data() + offset * genes_per_ind : nullptr,
                local_count * genes_per_ind, MPI_CHAR,
                local_genes.data(), local_count * genes_per_ind, MPI_CHAR,
                0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
        mpi_time += (MPI_Wtime() - t0);
#else
    if (mpi_rank == 0)
    {
        auto start_eval = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < count; ++i)
        {
            cost_calc.canonicalize(genes.data() + (offset + i) * genes_per_ind, n_var);
            cost_sse[(offset + i) * 2] = cost_calc.calculate(genes.data() + (offset + i) * genes_per_ind, n_var);
            cost_sse[(offset + i) * 2 + 1] = sse_calc.calculate(genes.data() + (offset + i) * genes_per_ind);
        }
        eval_time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_eval).count();
        return;
    }
#endif

#ifdef USE_MPI
    double start_eval = MPI_Wtime();
    for (int i = 0; i < local_count; ++i)
    {
        cost_calc.canonicalize(local_genes.data() + i * genes_per_ind, n_var);
        double c = cost_calc.calculate(local_genes.data() + i * genes_per_ind, n_var);
        double s = sse_calc.calculate(local_genes.data() + i * genes_per_ind);

        local_results[i * 2] = c;
        local_results[i * 2 + 1] = s;
    }
    eval_time += MPI_Wtime() - start_eval;

    double t2 = 0;
    if (mpi_rank == 0)
        t2 = MPI_Wtime();

    MPI_Gather(local_genes.data(), local_count * genes_per_ind, MPI_CHAR,
               mpi_rank == 0 ? genes.data() + offset * genes_per_ind : nullptr,
               local_count * genes_per_ind, MPI_CHAR,
               0, MPI_COMM_WORLD);

    MPI_Gather(local_results.data(), local_count * 2, MPI_DOUBLE,
               mpi_rank == 0 ? cost_sse.data() + offset * 2 : nullptr,
               local_count * 2, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
        gather_time += (MPI_Wtime() - t2);
#endif
}

int main(int argc, char **argv)
{
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);

    auto set_threads = [](const char *name, int val)
    {
        auto fn = (void (*)(int))dlsym(RTLD_DEFAULT, name);
        if (fn)
            fn(val);
    };
    auto set_threads_f = [](const char *name, int val)
    {
        auto fn = (void (*)(int *))dlsym(RTLD_DEFAULT, name);
        if (fn)
            fn(&val);
    };
    set_threads("openblas_set_num_threads", 1);
    set_threads("MKL_Set_Num_Threads", 1);
    set_threads_f("blas_set_num_threads_", 1);

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

#ifdef USE_MPI
    if (pop_size % size != 0)
    {
        pop_size = ((pop_size + size - 1) / size) * size;
        if (rank == 0)
        {
            std::cout << "Adjusted pop_size to " << pop_size
                      << " to be a perfect multiple of MPI size (" << size << ")\n";
        }
    }
#endif

    int n_gen = config["n_gen"];
    double time_limit = config.value("time", -1.0);
    int save_interval = config.value("save_interval", -1);

    // --- 2. Load Data ---
    MTPData mtp;

    if (rank == 0)
    {
        std::cout << "Loading MTP data...\n";
        mtp = parse_mtp(config["mtp_file"]);
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

    bcast_int_vec(mtp.alpha_index_basic, rank);
    bcast_int_vec(mtp.alpha_index_times, rank);
    bcast_int_vec(mtp.alpha_moment_mapping, rank);
#endif

    int n_var = mtp.alpha_scalar_moments;

    // --- 3. Initialize Evaluators and Buffers ---
    SSECalculator sse_calc(mtp.species_count, n_var, rank);

    CostCalculator cost_calc(mtp.alpha_moments_count, mtp.alpha_index_basic,
                             mtp.alpha_index_times, mtp.alpha_moment_mapping,
                             config["neigh_count"], mtp.radial_basis_size, rank);

    NSGA2 ga(pop_size, n_var, 42 + rank);

    std::vector<char> local_genes;
    std::vector<double> local_results;
    if (size > 0)
    {
        int local_count = pop_size / size;
        local_genes.resize(local_count * n_var);
        local_results.resize(local_count * 2);
    }

    std::string latest_pop_file = "";
    std::filesystem::path output_dir = config["out_dir"].get<std::string>();

    if (rank == 0)
    {
        std::string base_out_dir_str = output_dir.string();
        if (!base_out_dir_str.empty() && base_out_dir_str.back() == '/')
        {
            base_out_dir_str.pop_back();
        }

        int restart_count = 1;
        while (std::filesystem::exists(output_dir))
        {
            restart_count++;
            output_dir = std::filesystem::path(base_out_dir_str + "_" + std::to_string(restart_count));
        }

        for (int r = restart_count - 1; r >= 1; --r)
        {
            std::filesystem::path check_dir = (r == 1) ? std::filesystem::path(base_out_dir_str) : std::filesystem::path(base_out_dir_str + "_" + std::to_string(r));
            latest_pop_file = find_latest_population(check_dir);
            if (!latest_pop_file.empty())
            {
                break;
            }
        }

        try
        {
            std::filesystem::create_directories(output_dir);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error creating directory: " << e.what() << "\n";
        }

        if (!latest_pop_file.empty())
        {
            std::cout << "Restarting from previous results: " << latest_pop_file << "\n";
            std::cout << "Insufficient populations will be randomly generated. The population will be scrambled.\n";
        }
        std::cout << "Saving new results to: " << output_dir.string() << "\n";
    }

    double eval_time = 0.0;
    double mpi_time = 0.0;
    double gather_time = 0.0;

    auto start_time = std::chrono::high_resolution_clock::now();

    auto elapsed_s = [&]()
    {
        return std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - start_time)
            .count();
    };

    // --- 4. Initialization & Gen 0 Evaluation ---
    if (rank == 0)
    {
        ga.initialize_population(latest_pop_file);
        std::cout << "Evaluating initial population..." << std::endl;
    }

    evaluate_population(0, pop_size, n_var, rank, size, cost_calc, sse_calc,
                        ga.genes, ga.cost_sse, local_genes, local_results,
                        eval_time, mpi_time, gather_time);

    if (rank == 0)
    {
        ga.survival(pop_size);
    }

    // --- 5. Main Optimization Loop ---
    bool time_limit_reached = false;

    for (int gen = 0; gen < n_gen; ++gen)
    {
        if (rank == 0)
        {
            ga.generate_offspring();
        }

        evaluate_population(pop_size, pop_size, n_var, rank, size, cost_calc, sse_calc,
                            ga.genes, ga.cost_sse, local_genes, local_results,
                            eval_time, mpi_time, gather_time);

        if (rank == 0)
        {
            ga.survival(2 * pop_size);

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
                ga.save_pareto(prefix);
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

    std::vector<double> evals(size, 0.0);
#ifdef USE_MPI
    MPI_Gather(&eval_time, 1, MPI_DOUBLE, evals.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    evals[0] = eval_time;
#endif

    // --- 6. Save Final Results ---
    if (rank == 0)
    {
        double exec_time = elapsed_s();
        std::filesystem::path out_path = output_dir / "pareto_final";
        ga.save_pareto(out_path.string());

        std::cout << "Optimization finished in " << exec_time << "s";
        if (time_limit_reached)
            std::cout << " (stopped by time limit)";
        std::cout << ".\n";

#ifdef USE_MPI
        double max_eval = evals[0];
        double min_eval = evals[0];
        double sum_eval = 0.0;
        for (double e : evals)
        {
            if (e > max_eval)
                max_eval = e;
            if (e < min_eval)
                min_eval = e;
            sum_eval += e;
        }
        double mean_eval = sum_eval / size;

        double communication_time = mpi_time + gather_time - (max_eval - evals[0]);
        if (communication_time < 0.0)
            communication_time = 0.0;

        double serial_time = exec_time - max_eval - communication_time;
        if (serial_time < 0.0)
            serial_time = 0.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Evaluation times per process:[";
        for (int i = 0; i < size; ++i)
        {
            std::cout << evals[i] << " s" << (i == size - 1 ? "" : ", ");
        }
        std::cout << "]\n";
        std::cout << "Average fitness evaluation time: " << (mean_eval / exec_time * 100.0) << "%.\n";
        std::cout << "Communication time (Estimated): " << (communication_time / exec_time * 100.0) << "%.\n";
        std::cout << "Serial time (Estimated): " << (serial_time / exec_time * 100.0) << "%.\n";
        std::cout << "Wasted time due to load imbalance (Estimated): " << ((max_eval - min_eval) / exec_time * 100.0) << "%.\n";
#endif

        std::cout << "Results saved to " << out_path.string() << "_*.csv\n";
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}