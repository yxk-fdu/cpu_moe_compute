// Shared immutable MXFP4 NZ sources; existing CPUInfer/KT threads own execution.
#pragma once
#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include "../../cpu_backend/cpuinfer.h"
#include "../../cpu_backend/cpu_compute_trace.h"
#include "mxfp8_match.h"
#include "nz_moe_kernels.hpp"
#if defined(__AVX512BF16__) && defined(__AVX512F__)
#include "nz_moe_x86.hpp"
#endif
#if defined(__aarch64__) && defined(__linux__) && (defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || \
    (defined(__GNUC__) && !defined(__clang__)))
#define KT_NZ_ARM_BF16
#include "nz_moe_arm.hpp"
#endif

namespace kt_nz {
class MoE {
  WorkerPool* pool_;
  int layer_, hidden_, intermediate_, topk_, max_tokens_;
  std::vector<std::vector<uintptr_t>> sources_;
  std::vector<float> input_;
  std::vector<uint16_t> packed_input_, gates_, down_;
  struct Work { int expert, first, count; };
  std::vector<Work> work_;
  std::vector<int> expert_counts_, expert_end_, expert_routes_, route_slot_;
  struct Task {
    CPUInfer* cpu = nullptr;
    MoE* owner;
    const int* count;
    int k;
    const int64_t* ids;
    const float* routes;
    const uint16_t* input;
    uint16_t* output;
  };
  std::vector<std::unique_ptr<Task>> tasks_;
  static size_t dimension(int value) {
    if(value<1)throw std::invalid_argument("NZ dimensions must be positive");
    return size_t(value);
  }
  static float from_bf16(uint16_t v) { uint32_t u=uint32_t(v)<<16;float f;std::memcpy(&f,&u,4);return f; }
  static uint16_t to_bf16(float v) {v=kt_match_npu::bf16(v);uint32_t u;std::memcpy(&u,&v,4);return u>>16;}
  static void entry(void* opaque) {
    auto* t=static_cast<Task*>(opaque);
    t->cpu->enqueue(&MoE::forward,t->owner,t->count,t->k,t->ids,t->routes,t->input,t->output);
  }
  static void prepare(float* input, uint16_t* packed, int size) {
#if defined(__AVX512BF16__) && defined(__AVX512F__)
    x86::prepare(input, packed, size);
#elif defined(KT_NZ_ARM_BF16)
    arm::prepare(input, packed, size);
#endif
  }

  static void project(const uint8_t* w, const uint8_t* scales, int n, int groups, int first, int rows,
                      const InputTile& x, const OutputTile& y, int tokens) {
#if defined(__AVX512BF16__) && defined(__AVX512F__)
    x86::project(w, scales, n, groups, first, rows, x, y, tokens);
#elif defined(KT_NZ_ARM_BF16)
    arm::project(w, scales, n, groups, first, rows, x, y, tokens);
#endif
  }

  int group_routes(const int64_t* ids, int size) {
    std::fill(expert_counts_.begin(), expert_counts_.end(), 0);
    for (int route=0; route<size; ++route)
      if (ids[route]>=0) ++expert_counts_[ids[route]];

    int active=0;
    for (size_t expert=0; expert<expert_counts_.size(); ++expert) {
      expert_end_[expert]=active;
      active+=expert_counts_[expert];
    }
    for (int route=0; route<size; ++route) {
      const auto expert=ids[route];
      if (expert<0) continue;
      const int slot=expert_end_[expert]++;
      expert_routes_[slot]=route;
      route_slot_[route]=slot;
    }

    work_.clear();
    // Schedule repeated experts first, with one decoded block per token tile.
    for (const bool repeated : {true, false}) {
      for (int expert=0; expert<int(expert_counts_.size()); ++expert) {
        const int count=expert_counts_[expert];
        if (count==0 || (count>1)!=repeated) continue;
        const int end=expert_end_[expert];
        for (int slot=end-count; slot<end; slot+=kTokenTile)
          work_.push_back({expert,slot,std::min(kTokenTile,end-slot)});
      }
    }
    return active;
  }

  void run_projection(InNumaPool& pool, int k, bool gate_up) {
    const int n=gate_up?2*intermediate_:hidden_;
    const int dim=gate_up?hidden_:intermediate_;
    const int group_count=int(work_.size());
    const int parallel_parts=(pool_->config.subpool_thread_count[0]+group_count-1)/group_count;
    const int row_parts=std::min(n/4,std::max(parallel_parts,(n+kRowsPerJob-1)/kRowsPerJob));
    auto& destination=gate_up?gates_:down_;
    pool.do_work_stealing_job(group_count*row_parts, [&](int task) {
      const auto& group=work_[task/row_parts];
      const int part=task%row_parts;
      const int first=int(int64_t(n/4)*part/row_parts)*4;
      const int rows=int(int64_t(n/4)*(part+1)/row_parts)*4-first;
      std::array<std::array<float,kRowsPerJob>,kTokenTile> accum;
      InputTile x{};
      OutputTile y{};
      for (int i=0; i<group.count; ++i) {
        const int slot=group.first+i;
        x[i]=gate_up ? packed_input_.data()+size_t(expert_routes_[slot]/k)*hidden_
                     : gates_.data()+size_t(slot)*2*intermediate_;
        y[i]=accum[i].data();
      }
      const int field=gate_up?0:1;
      project(reinterpret_cast<const uint8_t*>(sources_[field][group.expert]),
              reinterpret_cast<const uint8_t*>(sources_[field+2][group.expert]),
              n,dim/32,first,rows,x,y,group.count);
      for (int i=0; i<group.count; ++i) {
        auto* out=destination.data()+size_t(group.first+i)*n+first;
        for (int row=0; row<rows; ++row) out[row]=to_bf16(accum[i][row]);
      }
    });
  }

 public:
  MoE(WorkerPool& pool,int layer,int hidden,int intermediate,int topk,int max_tokens,
      std::vector<std::vector<uintptr_t>> sources)
      :pool_(&pool),layer_(layer),hidden_(hidden),intermediate_(intermediate),
       topk_(topk),max_tokens_(max_tokens),sources_(std::move(sources)),input_(dimension(hidden)),
       packed_input_(dimension(max_tokens)*dimension(hidden)) {
#if !(defined(__AVX512BF16__) && defined(__AVX512F__)) && \
    !(defined(KT_NZ_ARM_BF16))
    throw std::invalid_argument("NZ MXFP4 requires x86 AVX512BF16 or ARM BF16 SIMD");
#endif
#if defined(KT_NZ_ARM_BF16)
    if (!arm::supported()) throw std::invalid_argument("NZ MXFP4 requires ARM BF16 SIMD");
#endif
    if(!pool_||layer<0||hidden<=0||hidden%32||intermediate<=0||intermediate%32||topk<1||max_tokens<1)
      throw std::invalid_argument("Invalid NZ MoE dimensions");
    if(pool_->config.subpool_count!=1||sources_.size()!=4||sources_[0].empty())
      throw std::invalid_argument("NZ MoE requires one NUMA pool and four borrowed source fields");
    for(const auto& field:sources_) {
      if(field.size()!=sources_[0].size())throw std::invalid_argument("NZ source count mismatch");
      for(auto p:field)if(!p)throw std::invalid_argument("Null NZ source pointer");
    }
    if(!kt_match_npu::matching_requested())throw std::invalid_argument("NZ MoE requires KT_MOE_MATCH_NPU_MXFP8=1");
    const size_t capacity=dimension(max_tokens_)*dimension(topk_);
    if (capacity>size_t(std::numeric_limits<int>::max()) ||
        sources_[0].size()>size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("NZ routing capacity exceeds supported range");
    expert_counts_.resize(sources_[0].size());
    expert_end_.resize(sources_[0].size());
    expert_routes_.resize(capacity);
    route_slot_.resize(capacity);
    work_.reserve(capacity);
    gates_.reserve(capacity*2*intermediate_);
    down_.reserve(capacity*hidden_);

  }
  auto weight_pointers() const {return sources_;}
  std::pair<intptr_t,intptr_t> forward_task(intptr_t count,int k,intptr_t ids,intptr_t routes,
                                          intptr_t input,intptr_t output,bool incremental=false) {
    if(incremental||k<1||k>topk_||!count||!ids||!routes||!input||!output)
      throw std::invalid_argument("Invalid NZ forward task");
    for(const auto& task:tasks_) {
      if(task->count==reinterpret_cast<const int*>(count)&&task->k==k&&
         task->ids==reinterpret_cast<const int64_t*>(ids)&&task->routes==reinterpret_cast<const float*>(routes)&&
         task->input==reinterpret_cast<const uint16_t*>(input)&&task->output==reinterpret_cast<uint16_t*>(output))
        return {reinterpret_cast<intptr_t>(&entry),reinterpret_cast<intptr_t>(task.get())};
    }
    auto task=std::make_unique<Task>();
    *task={nullptr,this,reinterpret_cast<const int*>(count),k,reinterpret_cast<const int64_t*>(ids),
           reinterpret_cast<const float*>(routes),reinterpret_cast<const uint16_t*>(input),reinterpret_cast<uint16_t*>(output)};
    auto address=reinterpret_cast<intptr_t>(task.get());tasks_.push_back(std::move(task));
    return {reinterpret_cast<intptr_t>(&entry),address};
  }
  void forward(const int* count,int k,const int64_t* ids,const float* routes,
               const uint16_t* input,uint16_t* output) {
    const int tokens=*count;
    if (tokens<0 || tokens>max_tokens_ || k<1 || k>topk_)
      throw std::invalid_argument("NZ route shape exceeds captured capacity");
    kt::cpu_trace::Span trace(layer_,0,tokens,k,ids,[](int64_t id){return id<0;});
    for (int r=0; r<tokens*k; ++r)
      if (ids[r]>=int64_t(sources_[0].size()))
        throw std::invalid_argument("NZ expert ID out of range");
    const int active=group_routes(ids,tokens*k);
    if (active==0) {
      std::fill_n(output,size_t(tokens)*hidden_,uint16_t{0});
      return;
    }
    const size_t gate_size=size_t(active)*2*intermediate_;
    const size_t down_size=size_t(active)*hidden_;
    if (gates_.size()<gate_size) gates_.resize(gate_size);
    if (down_.size()<down_size) down_.resize(down_size);

    pool_->dispense_backend()->do_numa_job([&](int part) {
      auto* sub=pool_->get_subpool(part);
      for (int t=0; t<tokens; ++t) {
        const auto* first=ids+t*k;
        if (std::all_of(first,first+k,[](int64_t id){return id<0;})) continue;
        for (int i=0; i<hidden_; ++i) input_[i]=from_bf16(input[size_t(t)*hidden_+i]);
        prepare(input_.data(),packed_input_.data()+size_t(t)*hidden_,hidden_);
      }
      run_projection(*sub,k,true);
      const int blocks=intermediate_/32;
      sub->do_work_stealing_job(active*blocks,[&](int task) {
        auto* g=gates_.data()+size_t(task/blocks)*2*intermediate_+(task%blocks)*32;
        const auto* u=g+intermediate_;
        std::array<float,32> activation;
        for (int j=0; j<32; ++j) {
          const float gv=std::min(from_bf16(g[j]),10.f);
          const float uv=std::max(-10.f,std::min(from_bf16(u[j]),10.f));
          activation[j]=kt_match_npu::bf16((gv/(1.f+expf(-gv)))*uv);
        }
        prepare(activation.data(),g,32);
      });
      run_projection(*sub,k,false);

      // Groups can run in any order. Reduce in the original per-token route order.
      const int chunks=(hidden_+255)/256;
      sub->do_work_stealing_job(tokens*chunks,[&](int task) {
        const int t=task/chunks;
        const int begin=(task%chunks)*256;
        const int len=std::min(256,hidden_-begin);
        std::array<float,256> sum{};
        for (int r=0; r<k; ++r) {
          const int route=t*k+r;
          if (ids[route]<0) continue;
          const auto* d=down_.data()+size_t(route_slot_[route])*hidden_+begin;
          const float score=kt_match_npu::bf16(routes[route]);
          for (int j=0; j<len; ++j) sum[j]+=from_bf16(d[j])*score;
        }
        auto* out=output+size_t(t)*hidden_+begin;
        for (int j=0; j<len; ++j) out[j]=to_bf16(sum[j]);
      });
    });

  }
};
} // namespace kt_nz

#undef KT_NZ_ARM_BF16
