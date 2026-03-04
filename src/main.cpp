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

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <dlfcn.h>

using json = nlohmann::json;

template <typename T>
std::vector<T> read_binary(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open " + filename);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<T> buffer(size / sizeof(T));
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
        throw std::runtime_error("Error reading " + filename);
    return buffer;
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
                         std::vector<double> &local_results)
{
    if (count == 0)
        return;

    int local_count = count / mpi_size;
    int genes_per_ind = n_var;

#ifdef USE_MPI
    // Master directly provides pointers starting at `offset * genes_per_ind`
    MPI_Scatter(mpi_rank == 0 ? genes.data() + offset * genes_per_ind : nullptr,
                local_count * genes_per_ind, MPI_CHAR,
                local_genes.data(), local_count * genes_per_ind, MPI_CHAR,
                0, MPI_COMM_WORLD);
#else
    // Non-MPI fallback reads directly from the NSGA object and calculates on the main pointer.
    if (mpi_rank == 0)
    {
        for (int i = 0; i < count; ++i)
        {
            cost_sse[(offset + i) * 2] = cost_calc.calculate(genes.data() + (offset + i) * genes_per_ind, n_var);
            cost_sse[(offset + i) * 2 + 1] = sse_calc.calculate(genes.data() + (offset + i) * genes_per_ind);
        }
        return;
    }
#endif

#ifdef USE_MPI
    for (int i = 0; i < local_count; ++i)
    {
        double c = cost_calc.calculate(local_genes.data() + i * genes_per_ind, n_var);
        double s = sse_calc.calculate(local_genes.data() + i * genes_per_ind);

        local_results[i * 2] = c;
        local_results[i * 2 + 1] = s;
    }

    // Interlaved pairs format gathers directly onto contiguous master array memory
    MPI_Gather(local_results.data(), local_count * 2, MPI_DOUBLE,
               mpi_rank == 0 ? cost_sse.data() + offset * 2 : nullptr,
               local_count * 2, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
#endif
}

int main(int argc, char **argv)
{
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);

    // Force single-threaded BLAS at runtime — setenv alone may be too late
    // if the library initialized its thread pool before main().
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
    set_threads("openblas_set_num_threads", 1); // OpenBLAS
    set_threads("MKL_Set_Num_Threads", 1);      // Intel MKL
    set_threads_f("blas_set_num_threads_", 1);  // Netlib (Fortran)

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

#ifdef USE_MPI
    // Pad pop_size so it aligns perfectly with MPI scatter/gather chunks.
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

    // --- 3. Initialize Evaluators and Buffers ---
    SSECalculator sse_calc(xtwx, xtwy, config["ytwy"],
                           config.value("regularization", 0.0),
                           mtp.species_count, n_var, rank);

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
        ga.initialize_population();
        std::cout << "Evaluating initial population..." << std::endl;
    }

    // Evaluate gen 0 evaluating directly at index 0 up to pop_size bounds
    evaluate_population(0, pop_size, n_var, rank, size, cost_calc, sse_calc,
                        ga.genes, ga.cost_sse, local_genes, local_results);

    if (rank == 0)
    {
        // Survival run isolated to rank population count itself to initialize parameters natively
        ga.survival(pop_size);
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
        if (rank == 0)
        {
            // Populate next pop_size sequence right inline natively over bounds
            ga.generate_offspring();
        }

        // Evaluate offspring generated at index `pop_size` up to count `pop_size` lengths
        evaluate_population(pop_size, pop_size, n_var, rank, size, cost_calc, sse_calc,
                            ga.genes, ga.cost_sse, local_genes, local_results);

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

    // --- 6. Save Final Results ---
    if (rank == 0)
    {
        std::filesystem::path out_path = output_dir / "pareto_final";
        ga.save_pareto(out_path.string());

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