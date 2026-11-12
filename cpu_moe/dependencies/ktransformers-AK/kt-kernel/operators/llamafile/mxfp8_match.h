#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
namespace kt_match_npu {
inline bool matching_requested() {
 static const bool value = std::getenv("KT_MOE_MATCH_NPU_MXFP8") != nullptr;
 return value;
}
inline bool enabled() {
#if defined(__aarch64__)
 // ARM uses the existing GGUF MXFP4 x Q8_0 llama.cpp path, regardless
 // of the launch environment. Keep the original x86 matching behavior.
 return false;
#else
 return matching_requested();
#endif
}
inline float bf16(float x) {
 uint32_t u; std::memcpy(&u,&x,4); u += 0x7fff + ((u >> 16) & 1); u &= 0xffff0000u;
 std::memcpy(&x,&u,4); return x;
}
inline void quant_dequant_mxfp8(float* values, int count) {
 for (int base=0; base<count; base+=32) {
  int end=std::min(base+32,count); float maximum=0;
  for(int i=base;i<end;++i) maximum=std::max(maximum,std::fabs(values[i]));
  if(maximum==0) continue;
  int e; std::frexp(maximum,&e); int scale_exp=std::max(-127,std::min(127,e-9));
  float scale=std::ldexp(1.0f,scale_exp);
  for(int i=base;i<end;++i) {
   float x=std::min(448.0f,std::fabs(values[i])/scale);
   int xe; std::frexp(x,&xe); float unit=std::ldexp(1.0f,std::max(-9,xe-4));
   float q=std::nearbyint(x/unit)*unit;
   values[i]=std::copysign(q*scale,values[i]);
  }
 }
}
inline void quant_dequant(float* values, int count) {
 if (enabled()) quant_dequant_mxfp8(values, count);
}
inline float dot_gguf_mxfp4_mxfp8(const uint8_t* weights, const float* input, int blocks) {
 static const int8_t table_values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
#if defined(__AVX512BF16__) && defined(__AVX512F__)
 const __m128i table=_mm_loadu_si128(reinterpret_cast<const __m128i*>(table_values));
 const __m128i mask=_mm_set1_epi8(15);
 __m512 sum=_mm512_setzero_ps();
 for(int b=0;b<blocks;++b) {
  const uint8_t* w=weights+b*17;
  __m128i packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+1));
  __m128i lo=_mm_shuffle_epi8(table,_mm_and_si128(packed,mask));
  __m128i hi=_mm_shuffle_epi8(table,_mm_and_si128(_mm_srli_epi16(packed,4),mask));
  __m512 scale=_mm512_set1_ps(std::ldexp(1.0f,int(w[0])-128));
  __m512 wl=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)),scale);
  __m512 wh=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)),scale);
  __m512bh wb=_mm512_cvtne2ps_pbh(wh,wl);
  __m512bh xb=_mm512_cvtne2ps_pbh(_mm512_loadu_ps(input+b*32+16),_mm512_loadu_ps(input+b*32));
  sum=_mm512_dpbf16_ps(sum,wb,xb);
 }
 return _mm512_reduce_add_ps(sum);
#else
 float sum=0;
 for(int b=0;b<blocks;++b) {
  const uint8_t* w=weights+b*17;float scale=std::ldexp(1.0f,int(w[0])-128);
  for(int i=0;i<16;++i) {
   sum += float(table_values[w[1+i]&15])*scale*input[b*32+i];
   sum += float(table_values[w[1+i]>>4])*scale*input[b*32+16+i];
  }
 }
 return sum;
#endif
}

// Reuse a GGUF weight block across independent activation rows. Keep each
// row's conversion, accumulation and reduction identical to the single-row dot.
template <int Rows>
inline void dot_gguf_mxfp4_mxfp8_many(const uint8_t* weights, const float* input,
                                    ptrdiff_t input_stride, int blocks,
                                    float* output, ptrdiff_t output_stride) {
 static_assert(Rows >= 2 && Rows <= 4, "MXFP4 token tile must contain 2..4 rows");
#if defined(__AVX512BF16__) && defined(__AVX512F__)
 static const int8_t values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
 const __m128i table=_mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
 const __m128i mask=_mm_set1_epi8(15);
 __m512 sums[Rows];
 for (int r=0;r<Rows;++r) sums[r]=_mm512_setzero_ps();
 for (int b=0;b<blocks;++b) {
  const uint8_t* w=weights+b*17;
  const __m128i packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+1));
  const __m128i lo=_mm_shuffle_epi8(table,_mm_and_si128(packed,mask));
  const __m128i hi=_mm_shuffle_epi8(table,_mm_and_si128(_mm_srli_epi16(packed,4),mask));
  const __m512 scale=_mm512_set1_ps(std::ldexp(1.0f,int(w[0])-128));
  const __m512 wl=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)),scale);
  const __m512 wh=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)),scale);
  const __m512bh wb=_mm512_cvtne2ps_pbh(wh,wl);
  for (int r=0;r<Rows;++r) {
   const float* x=input+r*input_stride+b*32;
   const __m512bh xb=_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x+16),_mm512_loadu_ps(x));
   sums[r]=_mm512_dpbf16_ps(sums[r],wb,xb);
  }
 }
 for (int r=0;r<Rows;++r) output[r*output_stride]=_mm512_reduce_add_ps(sums[r]);
#else
 // With -ffast-math a scalar multi-row loop may be reassociated differently
 // from the old dot. Retain the original kernel on CPUs without BF16 SIMD.
 for (int r=0;r<Rows;++r)
  output[r*output_stride]=dot_gguf_mxfp4_mxfp8(weights, input+r*input_stride, blocks);
#endif
}
}
