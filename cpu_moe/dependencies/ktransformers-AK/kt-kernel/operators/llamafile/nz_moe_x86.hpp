#pragma once

#include "mxfp8_match.h"
#include "nz_moe_kernels.hpp"

namespace kt_nz::x86 {

using ActivationBlock=__m512bh;
using Decoded=__m512bh;
struct Partial { __m512 value; };

inline Decoded decode(const uint8_t* w,uint8_t exp) {
    const auto lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const auto mask=_mm_set1_epi8(15);
    const auto bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w));
    const auto lo=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
    const auto hi=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
    const auto scale=_mm512_set1_ps(std::ldexp(1.f,int(exp)-128));
    const auto wl=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_unpacklo_epi8(lo,hi))),scale);
    const auto wh=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_unpackhi_epi8(lo,hi))),scale);
    return _mm512_cvtne2ps_pbh(wh,wl);
}
inline ActivationBlock load_activation(const uint16_t* x) {
    return (ActivationBlock)_mm512_loadu_si512(x);
}
inline void mac(Partial& sum,Decoded w,ActivationBlock x) {
    sum.value=_mm512_dpbf16_ps(sum.value,w,x);
}
inline float reduce(const Partial& s) { return _mm512_reduce_add_ps(s.value); }

inline void prepare(float* x,uint16_t* out,int size) {
    kt_match_npu::quant_dequant_mxfp8(x,size);
    for (int i=0; i<size; i+=32) {
        const auto a=_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x+i+16),_mm512_loadu_ps(x+i));
        _mm512_storeu_si512(out+i,(__m512i)a);
    }
}
template<int Tokens>
__attribute__((noinline))
void project(const uint8_t* weights,const uint8_t* scales,int n,int groups,
             int first,int rows,const InputTile& x,const OutputTile& y) {
    constexpr int Groups=4;
    std::array<std::array<Partial,Tokens>,kRowsPerJob> sums;
    std::fill_n(sums.begin(),rows,std::array<Partial,Tokens>{});
    for (int group=0; group<groups; group+=Groups) {
        const auto* w=weights+(size_t(group)*n+first)*16;
        const auto* s=scales+size_t(first)*groups+group;
        for (int r=0; r<rows; ++r) {
            auto accum=sums[r];
            for (int k=0; k<std::min(Groups,groups-group); ++k) {
                const auto* wk=w+(size_t(k)*n+r)*16;
                const auto exp=s[size_t(r)*groups+k];
                const auto wt=decode(wk,exp);
                for (int t=0; t<Tokens; ++t) {
                    const auto a=load_activation(x[t]+(group+k)*32);
                    mac(accum[t],wt,a);
                }
            }
            sums[r]=accum;
        }
    }
    for (int r=0; r<rows; ++r)
        for (int t=0; t<Tokens; ++t)
            y[t][r]=reduce(sums[r][t]);
}


inline void project(const uint8_t* w,const uint8_t* s,int n,int groups,int first,int rows,
                    const InputTile& x,const OutputTile& y,int tokens) {
    switch (tokens) {
    case 1: return project<1>(w,s,n,groups,first,rows,x,y);
    case 2: return project<2>(w,s,n,groups,first,rows,x,y);
    case 3: return project<3>(w,s,n,groups,first,rows,x,y);
    case 4: return project<4>(w,s,n,groups,first,rows,x,y);
    case 5: return project<5>(w,s,n,groups,first,rows,x,y);
    case 6: return project<6>(w,s,n,groups,first,rows,x,y);
    }
}

}  // namespace kt_nz::x86
