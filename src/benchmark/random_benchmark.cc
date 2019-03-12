#include "falconn/lsh_nn_table.h"

#include <Eigen/Dense>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <fstream>

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::fixed;
using std::mt19937_64;
using std::normal_distribution;
using std::scientific;
using std::sqrt;
using std::thread;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborQueryPool;
using falconn::LSHNearestNeighborTable;
using falconn::QueryStatistics;
using falconn::StorageHashTable;

typedef falconn::DenseVector<float> Vec;

class Timer {
 public:
  Timer() { start_time = high_resolution_clock::now(); }

  double elapsed_seconds() {
    auto end_time = high_resolution_clock::now();
    auto elapsed = duration_cast<duration<double>>(end_time - start_time);
    return elapsed.count();
  }

 private:
  high_resolution_clock::time_point start_time;
};

template <typename PointType>
void thread_function(LSHNearestNeighborQueryPool<PointType>* query_pool,
                     const vector<PointType>& queries,
                     const vector<int>& true_nns, int query_index_start,
                     int query_index_end, int* num_correct_in_thread,
                     double* total_query_time_outside_in_thread) {
  for (int ii = query_index_start; ii < query_index_end; ++ii) {
    Timer query_time;

    int32_t res = query_pool->find_nearest_neighbor(queries[ii]);

    *total_query_time_outside_in_thread += query_time.elapsed_seconds();
    if (res == true_nns[ii]) {
      *num_correct_in_thread += 1;
    }
  }
}

template <typename PointType>
void run_experiment(LSHNearestNeighborTable<PointType>* table,
                    const vector<PointType>& queries,
                    const vector<int>& true_nns, int num_probes,
                    double* avg_query_time,
                    double* success_probability) {
  unique_ptr<LSHNearestNeighborQueryPool<PointType>> query_pool(
      table->construct_query_pool(num_probes));
  vector<int> num_correct_per_thread(1, 0);
  vector<double> total_query_time_outside_per_thread(1, 0.0);
  vector<int> index_start(1, 0);
  vector<int> index_end(1, 0);

  int queries = queries.size();
  int last_end = 0;
  index_start[0] = last_end;
  index_end[0] = last_end + queries;

  vector<thread> threads;
  Timer total_time;

  threads.push_back(thread(thread_function<PointType>, query_pool.get(),
          cref(queries), cref(true_nns), index_start[ii],
          index_end[ii], &(num_correct_per_thread[ii]),
          &(total_query_time_outside_per_thread[ii])));
  threads[0].join();


  double total_computation_time = total_time.elapsed_seconds();
  *success_probability = num_correct_per_thread[0] / queries.size();
  *avg_query_time = total_query_time_outside_per_thread[0] / queries.size();

  cout << "Total experiment wall clock time: " << scientific
       << total_computation_time << " seconds" << endl;
  cout << "Empirical success probability: " << fixed << *success_probability
       << endl
       << endl;
  cout << "Query statistics:" << endl;
  QueryStatistics stats = query_pool->get_query_statistics();
  cout << "Average total query time: " << scientific
       << stats.average_total_query_time << " seconds" << endl;
  cout << "Average LSH time:         " << stats.average_lsh_time << " seconds"
       << endl;
  cout << "Average hash table time:  " << stats.average_hash_table_time
       << " seconds" << endl;
  cout << "Average distance time:    " << stats.average_distance_time
       << " seconds" << endl;
  cout << "Average number of candidates:        " << fixed
       << stats.average_num_candidates << endl;
  cout << "Average number of unique candidates: "
       << stats.average_num_unique_candidates << endl
       << endl;
}

int main() {
  try {
    const char* sepline =
        "----------------------------------------------------------------------"
        "-";

    // Data set parameters
    size_t n = 100000;             // number of data points
    size_t d = 128;                 // dimension
    size_t num_queries = 10000;      // number of query points
    double r = sqrt(2.0) / 2.0;  // distance to planted query
    uint64_t seed = 119417657;

    // Common LSH parameters
    int num_tables = 10;
    int num_setup_threads = 0;
    StorageHashTable storage_hash_table = StorageHashTable::FlatHashTable;
    DistanceFunction distance_function = DistanceFunction::EuclideanSquared;

    cout << sepline << endl;
    cout << "FALCONN C++ random data benchmark" << endl << endl;
    cout << "Data set parameters: " << endl;
    cout << "n = " << n << endl;
    cout << "d = " << d << endl;
    cout << "num_queries = " << num_queries << endl;
    cout << "r = " << r << endl;
    cout << "seed = " << seed << endl << sepline << endl;

    // Load  data
    cout << "Load data set ..." << endl;
    vector<Vec> data;
    {
      float vec[d];
      uint32_t dim = 0;
      std::ifstream base_input("../rl_hnsw/notebooks/data/SIFT100K/sift_base.fvecs", std::ios::binary);

      for (size_t i = 0; i < n; i++) {
        Vec v(d);
        base_input.read((char *) &dim, sizeof(uint32_t));
        if (dim != d) {
          std::cout << "file error\n";
          exit(1);
        }
        base_input.read((char *) vec, dim * sizeof(float));
        for (size_t j = 0; j < d; ++j) v[j] = vec[j];
        data.push_back(v);
      }
    }

    // Load queries
    cout << "Load queries ..." << endl;
    vector<Vec> queries;
    {
      std::ifstream query_input("../rl_hnsw/notebooks/data/SIFT100K/sift_query.fvecs", std::ios::binary);
      float vec[d];
      uint32_t dim = 0;
      for (size_t i = 0; i < num_queries; i++){
        Vec q(d);

        query_input.read((char *) &dim, sizeof(uint32_t));
        if (dim != d) {
          std::cout << "file error\n";
          exit(1);
        }
        query_input.read((char *) vec, dim * sizeof(float));
        for (size_t j = 0; j < d; ++j) q[j] = vec[j];
        queries.push_back(q);
      }
    }
    vector<int> true_nn(num_queries);
    {
      std::cout << " Load groundtruths...\n";
      std::ifstream gt_input("../rl_hnsw/notebooks/data/SIFT100K/test_gt.ivecs", std::ios::binary);
      uint32_t dim = 0;
      for (size_t i = 0; i < num_queries; i++){
        gt_input.read((char *) &dim, sizeof(uint32_t));
        if (dim != 1) {
          std::cout << "file error\n";
          exit(1);
        }
        gt_input.read((char *) (true_nn.data() + dim*i), dim * sizeof(int));
      }
    }

    // Cross polytope hashing
    LSHConstructionParameters params_cp;
    params_cp.dimension = d;
    params_cp.lsh_family = LSHFamily::CrossPolytope;
    params_cp.distance_function = distance_function;
    params_cp.storage_hash_table = storage_hash_table;
    params_cp.k = 3;
    params_cp.l = num_tables;
    params_cp.last_cp_dimension = 16;
    params_cp.num_rotations = 3;
    params_cp.num_setup_threads = num_setup_threads;
    params_cp.seed = seed ^ 833840234;
    int num_probes_cp = 896;

    cout << "Cross polytope hash" << endl << endl;

    Timer cp_construction;

    unique_ptr<LSHNearestNeighborTable<Vec>> cptable(
        move(construct_table<Vec>(data, params_cp)));

    double cp_construction_time = cp_construction.elapsed_seconds();

    cout << "k = " << params_cp.k << endl;
    cout << "last_cp_dim = " << params_cp.last_cp_dimension << endl;
    cout << "num_rotations = " << params_cp.num_rotations << endl;
    cout << "l = " << params_cp.l << endl;
    cout << "Number of probes = " << num_probes_cp << endl;
    cout << "Construction time: " << cp_construction_time << " seconds" << endl
         << endl;

    double cp_avg_time;
    double cp_success_prob;
    run_experiment(cptable.get(), queries, true_nn, num_probes_cp, &cp_avg_time, &cp_success_prob);

    cout << sepline << endl << "Summary:" << endl;
    cout << "Success probabilities:" << endl;
    cout << "  CP: " << cp_success_prob << endl;
    cout << "Average query times (seconds):" << endl;
    cout << "  CP time: " << cp_avg_time << endl;

  } catch (exception& e) {
    cerr << "exception: " << e.what() << endl;
    return 1;
  } catch (...) {
    cerr << "Unknown error" << endl;
    return 1;
  }
  return 0;
}
