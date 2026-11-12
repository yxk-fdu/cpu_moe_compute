#pragma once

#include <arm_neon.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

#include "mxfp8_match.h"
#include "nz_moe_kernels.hpp"

namespace kt_nz::arm {
inline bool supported() {
  return (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
}

#if defined(KT_NZ_SVE256)
inline bool sve256_supported() {
#if defined(HWCAP_SVE) && defined(HWCAP2_SVE2) && defined(HWCAP2_SVEBF16) && defined(PR_SVE_GET_VL)
  static const bool capable = (getauxval(AT_HWCAP) & HWCAP_SVE) &&
      (getauxval(AT_HWCAP2) & (HWCAP2_SVE2 | HWCAP2_SVEBF16)) == (HWCAP2_SVE2 | HWCAP2_SVEBF16);
  static thread_local const bool width_ok = capable && [] {
    const int vl = prctl(PR_SVE_GET_VL);
    return vl >= 0 && (vl & PR_SVE_VL_LEN_MASK) == 32;
  }();
  return width_ok;
#else
  return false;
#endif
}
#endif
} // namespace kt_nz::arm


#if defined(__GNUC__) && !defined(__clang__)
// The KT extension can keep its ARMv8.2 baseline target.
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+fp16+dotprod+bf16")
#endif

namespace kt_nz::arm {

struct alignas(16) WeightTable {
  uint16_t scale[256][16];

  WeightTable() {
    constexpr int fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (int exponent = 0; exponent < 256; ++exponent) {
      for (int nibble = 0; nibble < 16; ++nibble) {
        const float value = std::ldexp(float(fp4[nibble]), exponent - 128);
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        // AVX512 BF16 conversion flushes FP32 subnormals to signed zero.
        if ((bits & 0x7fffffff) < 0x00800000) bits &= 0x80000000;
        else bits += 0x7fff + ((bits >> 16) & 1u);
        scale[exponent][nibble] = uint16_t(bits >> 16);
      }
    }
  }
};

using Accumulators = std::array<float32x4_t, 4>;
using Activation = std::array<bfloat16x8_t, 4>;


inline void prepare(float* input, uint16_t* packed, int size) {
#if defined(KT_NZ_SVE256)
  if (sve256_supported()) return sve256::prepare(input, packed, size);
#endif
  kt_match_npu::quant_dequant_mxfp8(input, size);
  for (int i = 0; i < size; i += 8) {
    auto a = vcombine_bf16(vcvt_bf16_f32(vld1q_f32(input + i)), vcvt_bf16_f32(vld1q_f32(input + i + 4)));
    vst1q_u16(packed + i, vreinterpretq_u16_bf16(a));
  }
}

inline Activation load(const uint16_t* input) {
  Activation blocks;
  for (int i = 0; i < 4; ++i) blocks[i] = vreinterpretq_bf16_u16(vld1q_u16(input + i * 8));
  return blocks;
}

using Decoded = std::array<bfloat16x8_t, 4>;
inline Decoded decode(const uint8_t* weight, const uint16_t* scale) {
    const uint8x16x2_t table = {{vreinterpretq_u8_u16(vld1q_u16(scale)),
                               vreinterpretq_u8_u16(vld1q_u16(scale + 8))}};
    const auto bits = vld1q_u8(weight);
    const auto low = vshlq_n_u8(vandq_u8(bits, vdupq_n_u8(15)), 1);
    const auto high = vshlq_n_u8(vshrq_n_u8(bits, 4), 1);
    const auto left = vzip1q_u8(low, high), right = vzip2q_u8(low, high);
    const auto left_hi = vaddq_u8(left, vdupq_n_u8(1));
    const auto right_hi = vaddq_u8(right, vdupq_n_u8(1));
    return {vreinterpretq_bf16_u8(vqtbl2q_u8(table, vzip1q_u8(left, left_hi))),
            vreinterpretq_bf16_u8(vqtbl2q_u8(table, vzip2q_u8(left, left_hi))),
            vreinterpretq_bf16_u8(vqtbl2q_u8(table, vzip1q_u8(right, right_hi))),
            vreinterpretq_bf16_u8(vqtbl2q_u8(table, vzip2q_u8(right, right_hi)))};
}

inline void accumulate(Accumulators& sum, const Decoded& weight, const Activation& input) {
    for (int i = 0; i < 4; ++i) {
        auto upper = vbfmlaltq_f32(sum[i], weight[i], input[i]);
        sum[i] = vbfmlalbq_f32(upper, weight[i], input[i]);
    }
}

inline float reduce(const Accumulators& sum) {
  // Match the x86 reduction tree even with -ffast-math.
  auto low = vaddq_f32(sum[2], sum[0]);
  auto high = vaddq_f32(sum[3], sum[1]);
  asm volatile("" : "+w"(low), "+w"(high));
  auto four = vaddq_f32(high, low);
  asm volatile("" : "+w"(four));
  auto two = vadd_f32(vget_high_f32(four), vget_low_f32(four));
  asm volatile("" : "+w"(two));
  return vget_lane_f32(vpadd_f32(two, two), 0);
}


template<int Tokens>
void project_neon(const uint8_t* weights, const uint8_t* scales, int n, int groups,
                  int first, int rows, const InputTile& input, const OutputTile& output) {
  static const WeightTable table;
  std::array<std::array<Accumulators, Tokens>, kRowsPerJob> sums{};
  for (int g = 0; g < groups; ++g) {
    std::array<Activation, Tokens> x;
    for (int t = 0; t < Tokens; ++t) x[t] = load(input[t] + g * 32);
    const auto* w = weights + (size_t(g) * n + first) * 16;
    for (int r = 0; r < rows; ++r) {
      const auto exponent = scales[size_t(first + r) * groups + g];
      const auto block = decode(w + r * 16, table.scale[exponent]);
      for (int t = 0; t < Tokens; ++t) accumulate(sums[r][t], block, x[t]);
    }
  }
  for (int r = 0; r < rows; ++r)
    for (int t = 0; t < Tokens; ++t) output[t][r] = reduce(sums[r][t]);
}

inline void project(const uint8_t* w, const uint8_t* s, int n, int groups, int first, int rows,
                    const InputTile& x, const OutputTile& y, int tokens) {
#if defined(KT_NZ_SVE256)
  if (sve256_supported()) return sve256::project(w,s,n,groups,first,rows,x,y,tokens);
#endif
  switch (tokens) {
    case 1: return project_neon<1>(w,s,n,groups,first,rows,x,y);
    case 2: return project_neon<2>(w,s,n,groups,first,rows,x,y);
    case 3: return project_neon<3>(w,s,n,groups,first,rows,x,y);
    case 4: return project_neon<4>(w,s,n,groups,first,rows,x,y);
    case 5: return project_neon<5>(w,s,n,groups,first,rows,x,y);
    case 6: return project_neon<6>(w,s,n,groups,first,rows,x,y);
  }
}

} // namespace kt_nz::arm

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
