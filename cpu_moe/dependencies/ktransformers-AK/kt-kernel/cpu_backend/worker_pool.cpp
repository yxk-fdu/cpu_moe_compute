/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:34
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "worker_pool.h"
#include "cpu_compute_trace.h"

#include <hwloc/bitmap.h>
#include <numa.h>
#include <numaif.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <map>

#include "hwloc.h"

thread_local int WorkerPool::thread_local_id = -1;

namespace {
int moe_core_index(int relative_index) {
  static const int offset = [] {
    const char* value = std::getenv("KT_MOE_CORE_OFFSET");
    if (!value || !*value) return 0;
    char* end = nullptr;
    const long n = std::strtol(value, &end, 10);
    if (end == value || *end || n < 0 || n > 4096) {
      throw std::invalid_argument("KT_MOE_CORE_OFFSET must be an integer in [0, 4096]");
    }
    fprintf(stderr, "[KT_MOE_AFFINITY] core_offset=%ld\n", n);
    return int(n);
  }();
  return offset + relative_index;
}
void configure_moe_realtime(pthread_t thread, const char* role) {
  const char* value = std::getenv("KT_MOE_RT_PRIORITY");
  if (!value || !*value) return;
  char* end = nullptr;
  const long priority = std::strtol(value, &end, 10);
  if (*end || priority < sched_get_priority_min(SCHED_FIFO) || priority > sched_get_priority_max(SCHED_FIFO)) {
    throw std::invalid_argument("Invalid KT_MOE_RT_PRIORITY: " + std::string(value));
  }
  sched_param param{}; param.sched_priority = int(priority);
  int rc = pthread_setschedparam(thread, SCHED_FIFO, &param);
  int policy = 0; sched_param actual{};
  if (!rc) rc = pthread_getschedparam(thread, &policy, &actual);
  if (rc || policy != SCHED_FIFO || actual.sched_priority != priority) {
    throw std::runtime_error("Cannot set requested SCHED_FIFO for " + std::string(role) +
                             ": " + (rc ? strerror(rc) : "verification mismatch"));
  }
  fprintf(stderr, "[KT_MOE_RT] role=%s policy=SCHED_FIFO priority=%ld verified=true\n", role, priority);
}
}  // namespace


InNumaPool::InNumaPool(int max_thread_num) {
  printf("In Numa Worker Pool at NUMA %d, %d threads\n", numa_node_of_cpu(sched_getcpu()), max_thread_num);
  total_worker_count = max_thread_num;
  set_restricted_worker_count(total_worker_count);
  thread_state_ = std::unique_ptr<ThreadState[]>(new ThreadState[max_thread_num]);
  for (int i = 0; i < total_worker_count; i++) {
    thread_state_[i].status.store(ThreadStatus::WAITING, std::memory_order_release);
  }
  workers_.resize(total_worker_count);
  for (int i = 1; i < total_worker_count; i++) {
    workers_[i] = std::thread(&InNumaPool::worker_thread, this, i, -1);
  }
}

namespace {
void bind_compute_thread(pthread_t thread, int cpu, const std::string& name) {
  pthread_setname_np(thread, name.c_str());
  const int mask_cpus = std::max(cpu + 1, CPU_SETSIZE);
  const size_t mask_size = CPU_ALLOC_SIZE(mask_cpus);
  auto* mask = CPU_ALLOC(mask_cpus);
  if (!mask) throw std::bad_alloc();
  CPU_ZERO_S(mask_size, mask);
  CPU_SET_S(cpu, mask_size, mask);
  const int rc = pthread_setaffinity_np(thread, mask_size, mask);
  CPU_FREE(mask);
  if (rc != 0) {
    throw std::runtime_error("Failed to bind " + name + " to CPU " + std::to_string(cpu) + ": " + strerror(rc));
  }
  configure_moe_realtime(thread, name.c_str());
}
}  // namespace

namespace {
std::vector<std::vector<int>> resolve_pool_cpus(const std::vector<int>& numa_nodes,
                                               const std::vector<int>& thread_counts, bool inventory) {
  if (numa_nodes.empty() || (!inventory && numa_nodes.size() != thread_counts.size())) {
    throw std::invalid_argument("CPUInfer requires a NUMA node and thread count for every pool");
  }
  hwloc_topology_t topology;
  if (hwloc_topology_init(&topology) != 0) throw std::runtime_error("Cannot initialize CPU topology");
  auto topology_owner = std::unique_ptr<std::remove_pointer_t<hwloc_topology_t>, decltype(&hwloc_topology_destroy)>(
      topology, hwloc_topology_destroy);
  if (hwloc_topology_load(topology) != 0) throw std::runtime_error("Cannot load CPU topology");
  auto allowed = std::unique_ptr<std::remove_pointer_t<hwloc_bitmap_t>, decltype(&hwloc_bitmap_free)>(
      hwloc_bitmap_alloc(), hwloc_bitmap_free);
  auto candidates = std::unique_ptr<std::remove_pointer_t<hwloc_bitmap_t>, decltype(&hwloc_bitmap_free)>(
      hwloc_bitmap_alloc(), hwloc_bitmap_free);
  if (!allowed || !candidates || hwloc_get_cpubind(topology, allowed.get(), HWLOC_CPUBIND_THREAD) != 0) {
    throw std::runtime_error("Cannot read CPUInfer effective CPU affinity");
  }

  std::map<int, int> next_core;
  std::vector<std::vector<int>> pools;
  for (size_t i = 0; i < numa_nodes.size(); ++i) {
    const int node_id = numa_nodes[i];
    const int count = inventory ? 0 : thread_counts[i];
    const auto node = get_numa_node(topology, node_id);
    if (inventory && !node) {
      pools.emplace_back();
      continue;
    }
    if (!node || (!inventory && count <= 0)) {
      throw std::invalid_argument("Invalid CPUInfer pool: NUMA " + std::to_string(node_id) +
                                  ", threads " + std::to_string(count));
    }
    auto& index = next_core.try_emplace(node_id, moe_core_index(0)).first->second;
    auto& cpus = pools.emplace_back();
    while (inventory || cpus.size() < static_cast<size_t>(count)) {
      const auto core = hwloc_get_obj_inside_cpuset_by_type(topology, node->cpuset, HWLOC_OBJ_CORE, index++);
      if (!core) {
        if (inventory) break;
        throw std::invalid_argument("Insufficient eligible physical cores for CPUInfer on NUMA " +
                                    std::to_string(node_id) + " (pool requested " +
                                    std::to_string(count) + " threads; found " + std::to_string(cpus.size()) + ")");
      }
      hwloc_bitmap_and(candidates.get(), core->cpuset, allowed.get());
      hwloc_bitmap_and(candidates.get(), candidates.get(), node->cpuset);
      const int cpu = hwloc_bitmap_first(candidates.get());
      if (cpu >= 0) cpus.push_back(cpu);
    }
  }
  return pools;
}
}  // namespace

std::vector<std::vector<int>> resolve_worker_pool_cpus(const std::vector<int>& numa_nodes,
                                                       const std::vector<int>& thread_counts) {
  return resolve_pool_cpus(numa_nodes, thread_counts, false);
}

std::vector<std::vector<int>> available_worker_pool_cpus(const std::vector<int>& numa_nodes) {
  return resolve_pool_cpus(numa_nodes, {}, true);
}

InNumaPool::InNumaPool(const std::vector<int>& cpus, int numa_id, int threads_id_start) {
  total_worker_count = static_cast<int>(cpus.size());
  set_restricted_worker_count(total_worker_count);
  thread_state_ = std::make_unique<ThreadState[]>(total_worker_count);
  workers_.resize(total_worker_count);
  for (int i = 0; i < total_worker_count; i++) {
    thread_state_[i].status.store(ThreadStatus::WAITING, std::memory_order_release);
  }
  try {
    for (int i = 1; i < total_worker_count; i++) {
      workers_[i] = std::thread(&InNumaPool::worker_thread, this, i, numa_id);
      const std::string name = "numa_" + std::to_string(numa_id) + "_t_" + std::to_string(i + threads_id_start);
      bind_compute_thread(workers_[i].native_handle(), cpus[i], name);
    }
  } catch (...) {
    stop_workers();
    throw;
  }
}

InNumaPool::~InNumaPool() {
  stop_workers();
}

void InNumaPool::stop_workers() {
  for (int i = 0; i < total_worker_count; i++) {
    {
      std::lock_guard<std::mutex> lock(thread_state_[i].mutex);
      thread_state_[i].status.store(ThreadStatus::EXIT, std::memory_order_release);
    }
    thread_state_[i].cv.notify_one();
  }
  for (int i = 0; i < total_worker_count; i++) {
    if (workers_[i].joinable()) {
      workers_[i].join();
    }
  }
}

int InNumaPool::get_thread_num() {
  throw std::runtime_error("Deprecated");
  return total_worker_count;
}

void InNumaPool::set_restricted_worker_count(int count) { restricted_worker_count = count; }

#ifdef PROFILE_BALANCE
// Only the first PROFILE_MAX_JOBS jobs print.  A serving run drives one job per MoE
// layer per decode step, so an unconditional printf would flood stdout from the hot
// path and perturb the very thing it measures.  Override with -DPROFILE_MAX_JOBS=N.
#ifndef PROFILE_MAX_JOBS
#define PROFILE_MAX_JOBS 200
#endif
static std::atomic<uint64_t> g_prof_job_seq{0};
#endif

void InNumaPool::wait() {
  kt::cpu_trace::mark("pool_wait_enter");
  for (int i = 0; i < worker_count; i++) {
    while (thread_state_[i].status.load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }

  kt::cpu_trace::mark("pool_wait_return");
#ifdef PROFILE_BALANCE
  // worker_count is min(restricted_worker_count, task_num), so a 0-task job leaves it
  // at 0 -- guard the whole report or we would divide by zero and print NaN.
  if (worker_count > 0) {
    size_t max_time = 0;
    size_t min_time = thread_state_[0].finish_ns;
    size_t sum = 0;
    for (int i = 0; i < worker_count; i++) {
      sum += thread_state_[i].finish_ns;
      max_time = std::max(max_time, thread_state_[i].finish_ns);
      min_time = std::min(min_time, thread_state_[i].finish_ns);
    }
    // Fraction of pool capacity that did useful work: 1.0 = every worker finished
    // together.  Jobs short enough that every finish_ns rounds to 0 have no meaningful
    // ratio, so report 0.0 instead of NaN.
    double balance = (max_time > 0) ? (1.0 * sum / (double)(max_time * (size_t)worker_count)) : 0.0;

    uint64_t total_init = 0, total_compute = 0, total_finalize = 0, total_schedule = 0;
    uint64_t total_tasks = 0, max_task = 0;
    for (int i = 0; i < worker_count; i++) {
      total_init += thread_state_[i].init_ns;
      total_compute += thread_state_[i].compute_ns;
      total_finalize += thread_state_[i].finalize_ns;
      total_schedule += thread_state_[i].schedule_ns;
      total_tasks += thread_state_[i].task_count;
      max_task = std::max(max_task, thread_state_[i].max_task_ns);
    }

    uint64_t seq = g_prof_job_seq.fetch_add(1, std::memory_order_relaxed);
    if (seq < (uint64_t)PROFILE_MAX_JOBS) {
      uint64_t t_mono_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
      uint64_t t_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
      // Dispatch -> here, i.e. including thread wake-up latency.  This is the
      // CPU-side interval to align against an NPU/torch timeline; max_worker_ms is
      // only the slowest thread's own execution time and is always the smaller of
      // the two.
      double job_wall_ms =
          (job_start_ns_ > 0) ? (double)(t_mono_ns - job_start_ns_) / 1e6 : 0.0;

      printf("[PROFILE_BALANCE] seq=%llu t_mono_ns=%llu t_wall_ns=%llu job_wall_ms=%.3f "
             "workers=%d tasks=%llu total_ms=%.3f init_ms=%.3f compute_ms=%.3f "
             "finalize_ms=%.3f schedule_ms=%.3f max_worker_ms=%.3f min_worker_ms=%.3f "
             "balance=%.4f max_task_us=%.3f\n",
             (unsigned long long)seq, (unsigned long long)t_mono_ns,
             (unsigned long long)t_wall_ns, job_wall_ms, worker_count,
             (unsigned long long)total_tasks, sum / 1e6, total_init / 1e6,
             total_compute / 1e6, total_finalize / 1e6, total_schedule / 1e6,
             max_time / 1e6, min_time / 1e6, balance, max_task / 1e3);
      // Unbuffered: a killed or aborted run must not lose the lines already computed.
      fflush(stdout);
    }
  }
#endif
}

void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> compute_func) {
  do_work_stealing_job(task_num, nullptr, compute_func, nullptr);
}

void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {
  do_work_stealing_job_async(task_num, init_func, compute_func, finalize_func);
  wait();
}

void InNumaPool::do_work_stealing_job_async(int task_num, std::function<void(int)> init_func,
                                            std::function<void(int)> compute_func,
                                            std::function<void(int)> finalize_func) {
  lifecycle_task_id_=kt::cpu_trace::context.id; lifecycle_layer_=kt::cpu_trace::context.layer;
  kt::cpu_trace::mark("pool_dispatch");
  init_func_ = init_func;
  compute_func_ = compute_func;
  finalize_func_ = finalize_func;
  worker_count = std::min(restricted_worker_count, task_num);
#ifdef PROFILE_BALANCE
  // Stamped before any worker is notified, so the job_wall_ms reported in wait()
  // covers thread wake-up latency as well -- that is the CPU-side interval that has
  // to line up against the NPU timeline.
  job_start_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
#endif
  curr_.store(0, std::memory_order_release);
  end_ = task_num;
  for (int i = 0; i < worker_count; i++) {
    {
      std::lock_guard<std::mutex> lock(thread_state_[i].mutex);
      thread_state_[i].status.store(ThreadStatus::WORKING, std::memory_order_release);
    }
    thread_state_[i].cv.notify_one();
  }
  WorkerPool::thread_local_id = 0;
  process_tasks(0);
}

void InNumaPool::process_tasks(int thread_id) {
  kt::cpu_trace::ContextScope scope({lifecycle_task_id_,lifecycle_layer_});
  kt::cpu_trace::mark("worker_begin",thread_id);
#ifdef PROFILE_BALANCE
  // steady_clock, not high_resolution_clock: on libstdc++ the latter *is* system_clock,
  // so an NTP step mid-job would corrupt the total and make it incomparable with the
  // per-stage steady_clock values accumulated below.
  auto start = std::chrono::steady_clock::now();
#endif
  auto& s = thread_state_[thread_id];
#ifdef PROFILE_BALANCE
  // Per-job, NOT cumulative.  These are accumulated with += / ++ / max, so without this
  // reset they keep growing over every job the pool has ever run -- while finish_ns
  // (assigned, not accumulated) stays per-job.  The report mixes both, so compute_ms
  // ended up exceeding total_ms, which is impossible within a single job.
  s.init_ns = 0;
  s.compute_ns = 0;
  s.finalize_ns = 0;
  s.schedule_ns = 0;
  s.task_count = 0;
  s.max_task_ns = 0;
#endif
  if (init_func_ != nullptr) {
    auto t0 = std::chrono::steady_clock::now();
    init_func_(thread_id);
    s.init_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  }

  // omp-guided-style work scheduling
  while (true) {
    int old = curr_.load(std::memory_order_relaxed);
    int rem = end_ - old;
    if (rem <= 0) {
      break;
    }

    int block = (rem + worker_count - 1) / worker_count;
    block = 1;
    auto sched_start = std::chrono::steady_clock::now();
    int task_id = curr_.fetch_add(block, std::memory_order_acq_rel);
    s.schedule_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sched_start).count();
    if (task_id >= end_) {
      break;
    }

    for (int i = 0; i < block; i++) {
      if (task_id + i >= end_) {
        break;
      }
      auto task_start = std::chrono::steady_clock::now();
      compute_func_(task_id + i);
      auto task_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - task_start).count();
      s.compute_ns += task_ns;
      s.max_task_ns = std::max<uint64_t>(s.max_task_ns, task_ns);
      ++s.task_count;
    }
  }

  if (finalize_func_ != nullptr) {
    auto t0 = std::chrono::steady_clock::now();
    finalize_func_(thread_id);
    s.finalize_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  }

#ifdef PROFILE_BALANCE
  // MUST be written before the release store below.  wait() spins on `status` with
  // acquire and reads finish_ns immediately after observing WAITING, so publishing
  // WAITING first let it read a stale/zero finish_ns -- which showed up as
  // compute_ms > total_ms, i.e. more time inside the tasks than the worker ran.
  s.finish_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
#endif
  kt::cpu_trace::mark("worker_end",thread_id);
  s.status.store(ThreadStatus::WAITING, std::memory_order_release);
}

void InNumaPool::worker_thread(int thread_id, int numa_id) {
  if (numa_id >= 0) {
    set_memory_to_numa(numa_id);
  }
  auto start = std::chrono::high_resolution_clock::now();
  WorkerPool::thread_local_id = thread_id;  // 设置线程本地变量
  while (true) {
    ThreadStatus status = thread_state_[thread_id].status.load(std::memory_order_acquire);
    if (status == ThreadStatus::WORKING) {
      process_tasks(thread_id);
      start = std::chrono::high_resolution_clock::now();
    } else if (status == ThreadStatus::WAITING) {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      if (duration > 50) {
        std::unique_lock<std::mutex> lock(thread_state_[thread_id].mutex);
        thread_state_[thread_id].cv.wait(lock, [&] {
          return thread_state_[thread_id].status.load(std::memory_order_acquire) != ThreadStatus::WAITING;
        });
      }
    } else if (status == ThreadStatus::EXIT) {
      return;
    }
  }
}

NumaJobDistributor::NumaJobDistributor(int numa_count) {
  std::vector<int> numa_ids;
  for (int i = 0; i < numa_count; i++) {
    numa_ids.push_back(i);
  }
  init(numa_ids);
}

NumaJobDistributor::NumaJobDistributor(std::vector<int> numa_ids) { init(numa_ids); }
NumaJobDistributor::NumaJobDistributor(std::vector<int> numa_ids, const std::vector<std::vector<int>>& pool_cpus) {
  init(numa_ids, pool_cpus);
}

void NumaJobDistributor::init(std::vector<int> numa_ids) {
  this->numa_count = numa_ids.size();
  this->ready_bar = std::unique_ptr<std::barrier<>>(new std::barrier<>(numa_count + 1));
  this->numa_ids = numa_ids;
  for (size_t i = 0; i < numa_count; i++) {
    status.push_back(nullptr);
    mutexes.push_back(std::make_unique<std::mutex>());
    cvs.push_back(std::make_unique<std::condition_variable>());
  }

  workers.resize(numa_count);
  for (int i = 0; i < numa_count; i++) {
    std::thread([this, i]() { workers[i] = std::thread(&NumaJobDistributor::worker_thread, this, i); }).join();
  }
  ready_bar->arrive_and_wait();
}

void NumaJobDistributor::init(std::vector<int> numa_ids, const std::vector<std::vector<int>>& pool_cpus) {
  this->numa_count = numa_ids.size();
  this->ready_bar = std::unique_ptr<std::barrier<>>(new std::barrier<>(numa_count + 1));
  this->numa_ids = numa_ids;
  for (size_t i = 0; i < numa_count; i++) {
    status.push_back(nullptr);
    mutexes.push_back(std::make_unique<std::mutex>());
    cvs.push_back(std::make_unique<std::condition_variable>());
  }

  workers.resize(numa_count);
  std::map<int, int> numa_threads_count;
  int started = 0;
  try {
    while (started < numa_count) {
      const int i = started;
      const int node = numa_ids[i];
      const int start_id = numa_threads_count[node];
      workers[i] = std::thread(&NumaJobDistributor::worker_thread, this, i);
      ++started;
      const std::string name = "numa_" + std::to_string(node) + "_m_" + std::to_string(start_id);
      bind_compute_thread(workers[i].native_handle(), pool_cpus[i].front(), name);
      numa_threads_count[node] += static_cast<int>(pool_cpus[i].size());
    }
  } catch (...) {
    for (int i = started; i < numa_count; ++i) ready_bar->arrive_and_drop();
    ready_bar->arrive_and_wait();
    stop_workers();
    throw;
  }
  ready_bar->arrive_and_wait();
}

NumaJobDistributor::~NumaJobDistributor() {
  stop_workers();
}

void NumaJobDistributor::stop_workers() {
  for (int i = 0; i < numa_count; i++) {
    if (!workers[i].joinable()) continue;
    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::EXIT, std::memory_order_release);
    }
    cvs[i]->notify_one();
  }
  for (int i = 0; i < numa_count; i++) {
    if (workers[i].joinable()) {
      workers[i].join();
    }
  }
}

#ifdef USE_NUMA_JOB_DIRECT_WORK

void NumaJobDistributor::do_numa_job(std::function<void(int)> compute_func) {
  auto ctx=kt::cpu_trace::context;
  this->compute_func = [compute_func,ctx](int i){kt::cpu_trace::ContextScope scope(ctx);kt::cpu_trace::mark("numa_take",i);compute_func(i);kt::cpu_trace::mark("numa_return",i);};
  auto me_numa = numa_node_of_cpu(sched_getcpu());
  for (int i = 0; i < numa_count; i++) {
    if (i == me_numa) continue;

    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::WORKING, std::memory_order_release);
    }
    cvs[i]->notify_one();
  }
  compute_func(me_numa);
  for (int i = 0; i < numa_count; i++) {
    if (i == me_numa) continue;

    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }
}
#else
void NumaJobDistributor::do_numa_job(std::function<void(int)> compute_func) {
  auto ctx=kt::cpu_trace::context;
  this->compute_func = [compute_func,ctx](int i){kt::cpu_trace::ContextScope scope(ctx);kt::cpu_trace::mark("numa_take",i);compute_func(i);kt::cpu_trace::mark("numa_return",i);};
  for (int i = 0; i < numa_count; i++) {
    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::WORKING, std::memory_order_release);
    }
    cvs[i]->notify_one();
  }
  for (int i = 0; i < numa_count; i++) {
    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }
}
#endif

void NumaJobDistributor::worker_thread(int numa_id) {
  auto start = std::chrono::high_resolution_clock::now();
  set_memory_to_numa(numa_ids[numa_id]);
  status[numa_id] =
      std::move(std::unique_ptr<std::atomic<ThreadStatus>>(new std::atomic<ThreadStatus>(ThreadStatus::WAITING)));
  ready_bar->arrive_and_wait();
  while (true) {
    auto stat = status[numa_id]->load(std::memory_order_acquire);
    if (stat == ThreadStatus::WORKING) {
      auto me_numa = numa_node_of_cpu(sched_getcpu());
      // printf("numa work on %d, me %d\n", numa_id, me_numa);
      compute_func(numa_id);
      status[numa_id]->store(ThreadStatus::WAITING, std::memory_order_release);
      start = std::chrono::high_resolution_clock::now();
    } else if (stat == ThreadStatus::WAITING) {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      if (duration > 50) {
        std::unique_lock<std::mutex> lock(*mutexes[numa_id]);
        cvs[numa_id]->wait(lock, [&] {
          return status[numa_id]->load(std::memory_order_acquire) != ThreadStatus::WAITING;
        });
      }
    } else if (stat == ThreadStatus::EXIT) {
      return;
    }
  }
}

void WorkerPool::init(WorkerPoolConfig config) {
  if (config.subpool_count <= 0 || static_cast<size_t>(config.subpool_count) != config.subpool_numa_map.size()) {
    throw std::invalid_argument("CPUInfer subpool_count must match the NUMA configuration");
  }
  auto pool_cpus = config.subpool_cpus;
  if (pool_cpus.empty()) {
    pool_cpus = resolve_worker_pool_cpus(config.subpool_numa_map, config.subpool_thread_count);
  } else {
    if (pool_cpus.size() != config.subpool_numa_map.size() || pool_cpus.size() != config.subpool_thread_count.size()) {
      throw std::invalid_argument("Explicit CPUInfer placement must specify CPUs for every pool");
    }
    std::vector<int> selected;
    for (size_t i = 0; i < pool_cpus.size(); ++i) {
      const auto available = available_worker_pool_cpus({config.subpool_numa_map[i]});
      if (pool_cpus[i].empty() || pool_cpus[i].size() != static_cast<size_t>(config.subpool_thread_count[i])) {
        throw std::invalid_argument("Explicit CPUInfer placement does not match the thread count");
      }
      for (int cpu : pool_cpus[i]) {
        if (std::find(available[0].begin(), available[0].end(), cpu) == available[0].end() ||
            std::find(selected.begin(), selected.end(), cpu) != selected.end()) {
          throw std::invalid_argument("CPUInfer placement is no longer eligible: CPU " + std::to_string(cpu));
        }
        selected.push_back(cpu);
      }
    }
  }
  numa_worker_pools.reserve(config.subpool_count);
  std::map<int, int> numa_threads_count;
  for (int i = 0; i < config.subpool_count; i++) {
    const int node = config.subpool_numa_map[i];
    const int start_id = numa_threads_count[node];
    // Keep NUMA allocation policy changes on a temporary thread.
    auto pool = std::async(std::launch::async, [&, node, start_id, i] {
      set_to_numa(node);
      return std::make_unique<InNumaPool>(pool_cpus[i], node, start_id);
    });
    numa_worker_pools.push_back(pool.get());
    numa_threads_count[node] += static_cast<int>(pool_cpus[i].size());
  }
  distributor = std::make_unique<NumaJobDistributor>(config.subpool_numa_map, pool_cpus);
}

WorkerPool::WorkerPool(WorkerPoolConfig config) : config(config) { init(config); }

WorkerPool::WorkerPool(int total_threads) {
  config.subpool_count = numa_num_configured_nodes();
  config.subpool_numa_map.resize(config.subpool_count);
  config.subpool_thread_count.resize(config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    config.subpool_numa_map[i] = i;
    config.subpool_thread_count[i] = total_threads / config.subpool_count;
  }
  init(config);
}

WorkerPool::WorkerPool(int total_threads, int single_numa_id) {
  set_to_numa(single_numa_id);
  config.subpool_count = numa_num_configured_nodes();
  config.subpool_numa_map.resize(config.subpool_count);
  config.subpool_thread_count.resize(config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    config.subpool_numa_map[i] = single_numa_id;
    config.subpool_thread_count[i] = total_threads / config.subpool_count;
  }
  init(config);
}

WorkerPool::~WorkerPool() {}

int WorkerPool::get_thread_num() { return total_thread_count; }

void WorkerPool::set_restricted_worker_count(int count) {
  for (int i = 0; i < numa_count; i++) {
    numa_worker_pools[i]->set_restricted_worker_count(threads_per_numa);
  }
}

InNumaPool* WorkerPool::get_subpool(int numa_id) { return numa_worker_pools[numa_id].get(); }

NumaJobDistributor* WorkerPool::dispense_backend() { return distributor.get(); }

void WorkerPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {
  numa_worker_pools[0]->do_work_stealing_job(task_num, init_func, compute_func, finalize_func);
}

void WorkerPool::do_work_stealing_job(int task_num, std::function<void(int)> compute_func) {
  do_work_stealing_job(task_num, nullptr, compute_func, nullptr);
}
