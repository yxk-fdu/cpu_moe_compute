#include "nz_moe_arm.hpp"

#include <arm_sve.h>

namespace kt_nz::sve256 {
namespace {

typedef svfloat32_t VecF32 __attribute__((arm_sve_vector_bits(256)));
typedef svbfloat16_t VecBF16 __attribute__((arm_sve_vector_bits(256)));

struct Activation256 {
    VecBF16 lo, hi;
};

inline Activation256 load_activation(const uint16_t* input) {
    const auto all = svptrue_b16();
    return {svreinterpret_bf16_u16(svld1_u16(all, input)),
            svreinterpret_bf16_u16(svld1_u16(all, input+16))};
}

inline Activation256 decode_sve(const uint8_t* src, const uint16_t* table) {
    const auto all = svptrue_b16();
    const auto weights = svld1ub_u16(all, src);
    const auto lut = svld1_u16(all, table);
    const auto lo = svand_n_u16_x(all, weights, 15);
    const auto hi = svlsr_n_u16_x(all, weights, 4);
    return {svreinterpret_bf16_u16(svtbl_u16(lut, svzip1_u16(lo, hi))),
            svreinterpret_bf16_u16(svtbl_u16(lut, svzip2_u16(lo, hi)))};
}

inline void pack_input(const float* x, uint16_t* out, int size) {
    for (int j = 0; j < size; j += 8) {
        const auto a = vcombine_bf16(vcvt_bf16_f32(vld1q_f32(x+j)),
                                    vcvt_bf16_f32(vld1q_f32(x+j+4)));
        vst1q_u16(out+j, vreinterpretq_u16_bf16(a));
    }
}


float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value,&bits,sizeof(value));
    return value;
}

uint32_t to_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits,&value,sizeof(bits));
    return bits;
}

void quant_sve(float* values, int count) {
    const auto pg = svptrue_b32();
    for (int base=0; base<count; base+=32) {
        auto* in=values+base;
        auto maximum=svdup_f32(0);
        for (int j=0; j<32; j+=8)
            maximum=svmax_f32_x(pg,maximum,svabs_f32_x(pg,svld1_f32(pg,in+j)));
        const int exponent=int((to_bits(svmaxv_f32(pg,maximum))>>23)&255);
        if (exponent==0) {
            kt_match_npu::quant_dequant_mxfp8(in,32);
            continue;
        }
        const int scale=exponent-135;
        if (scale < -126 || exponent==255) {
            kt_match_npu::quant_dequant_mxfp8(in,32);
            continue;
        }
        const float step=from_bits(uint32_t(127+scale)<<23);
        const float inv_step=from_bits(uint32_t(127-scale)<<23);
        for (int j=0; j<32; j+=8) {
            const auto x=svld1_f32(pg,in+j);
            const auto v=svmin_n_f32_x(pg,svmul_n_f32_x(pg,svabs_f32_x(pg,x),inv_step),448);
            const auto e=svlsr_n_u32_x(pg,svreinterpret_u32_f32(v),23);
            const auto unit_exp=svsub_n_u32_x(pg,svmax_n_u32_x(pg,e,121),3);
            const auto unit=svreinterpret_f32_u32(svlsl_n_u32_x(pg,unit_exp,23));
            const auto inverse=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svsub_u32_x(pg,svdup_u32(254),unit_exp),23));
            const auto rounded=svrintn_f32_x(pg,svmul_f32_x(pg,v,inverse));
            const auto result=svmul_n_f32_x(pg,svmul_f32_x(pg,rounded,unit),step);
            const auto sign=svand_n_u32_x(pg,svreinterpret_u32_f32(x),0x80000000);
            svst1_f32(pg,in+j,svreinterpret_f32_u32(svorr_u32_x(pg,svreinterpret_u32_f32(result),sign)));
        }
    }
}

inline void accumulate(VecF32& lo, VecF32& hi, const Activation256& w, const Activation256& x) {
    lo = svbfmlalb_f32(svbfmlalt_f32(lo, w.lo, x.lo), w.lo, x.lo);
    hi = svbfmlalb_f32(svbfmlalt_f32(hi, w.hi, x.hi), w.hi, x.hi);
}
template<int Tokens>
struct Sums256 {
    std::array<VecF32, Tokens> lo, hi;
};

inline float reduce256(VecF32 lo, VecF32 hi) {
    alignas(32) float lanes[16];
    svst1_f32(svptrue_b32(), lanes, lo);
    svst1_f32(svptrue_b32(), lanes+8, hi);
    return arm::reduce({vld1q_f32(lanes),vld1q_f32(lanes+4),vld1q_f32(lanes+8),vld1q_f32(lanes+12)});
}

inline float reduce256_single_token(VecF32 lo, VecF32 hi) {
    // Keep the same pairwise addition tree as arm::reduce.
    const auto all = svptrue_b32();
    auto eight = svadd_f32_x(all, hi, lo);
    asm volatile("" : "+w"(eight));
    auto four = svadd_f32_x(all, svext_f32(eight, eight, 4), eight);
    asm volatile("" : "+w"(four));
    auto two = svadd_f32_x(all, svext_f32(four, four, 2), four);
    asm volatile("" : "+w"(two));
    const auto sum = svadd_f32_x(all, two, svext_f32(two, two, 1));
    return svlastb_f32(svptrue_pat_b32(SV_VL1), sum);
}


template<int Tokens>
void project_rows(const uint8_t* weights,const uint8_t* scales,int n,int groups,int first,int rows,
                  const InputTile& x,const OutputTile& y) {
    static const arm::WeightTable table;
    std::array<Sums256<Tokens>,kRowsPerJob> sums;
    std::fill_n(sums.begin(),rows,Sums256<Tokens>{});
    constexpr int kGroups=4;
    const size_t stride=size_t(n)*16;
    for (int g=0; g<groups; g+=kGroups) {
        const auto* w=weights+(size_t(g)*n+first)*16;
        const auto* s=scales+size_t(first)*groups+g;
        const int chunk=std::min(kGroups,groups-g);
        // This distance and coverage were selected on Kunpeng 950 with cold NZ weights.
        if (g+kGroups<groups) {
            const int ahead=std::min(kGroups,groups-g-kGroups);
            for (int k=0;k<ahead;++k) {
                const auto* line=w+size_t(k+kGroups)*stride;
                #pragma GCC unroll 4
                for (int offset=0;offset<256;offset+=64) {
                    if (offset<rows*16)
                        asm volatile("prfm pldl2keep, [%0]" :: "r"(line+offset));
                }
            }
        }
        for (int row=0;row<rows;++row) {
            auto acc=sums[row];
            for (int k=0;k<chunk;++k) {
                const auto exponent=s[size_t(row)*groups+k];
                const auto wt=decode_sve(w+size_t(k)*stride+row*16,table.scale[exponent]);
                for (int t=0;t<Tokens;++t) {
                    const auto a=load_activation(x[t]+(g+k)*32);
                    accumulate(acc.lo[t],acc.hi[t],wt,a);
                }
            }
            sums[row]=acc;
        }
    }
    for (int r=0;r<rows;++r) {
        if constexpr (Tokens == 1) {
            y[0][r] = reduce256_single_token(sums[r].lo[0], sums[r].hi[0]);
        } else {
            for (int t=0;t<Tokens;++t) y[t][r]=reduce256(sums[r].lo[t],sums[r].hi[t]);
        }
    }
}
inline Activation256 load_activation_deinterleaved(const uint16_t* input) {
    const auto all = svptrue_b16();
    const auto pair = svld2_bf16(all, reinterpret_cast<const bfloat16_t*>(input));
    return {svget2_bf16(pair, 0), svget2_bf16(pair, 1)}; // lo=even, hi=odd
}

inline Activation256 decode_sve_nozip(const uint8_t* src, const uint16_t* table) {
    const auto all = svptrue_b16();
    const auto weights = svld1ub_u16(all, src);
    const auto lut = svld1_u16(all, table);
    const auto hi = svlsr_n_u16_x(all, weights, 4);
    const auto lo = svand_n_u16_x(all, weights, 15);
    return {svreinterpret_bf16_u16(svtbl_u16(lut, lo)),  // even weights
            svreinterpret_bf16_u16(svtbl_u16(lut, hi))}; // odd weights
}

inline void accumulate_exact_nozip(VecF32& acc_b, VecF32& acc_t,
                                   const Activation256& w, const Activation256& x) {
    acc_b = svbfmlalb_f32(svbfmlalb_f32(acc_b, w.hi, x.hi), w.lo, x.lo);
    acc_t = svbfmlalt_f32(svbfmlalt_f32(acc_t, w.hi, x.hi), w.lo, x.lo);
}

template<>
void project_rows<1>(const uint8_t* weights, const uint8_t* scales, int n, int groups,
                     int first, int rows, const InputTile& x, const OutputTile& y) {
    static const arm::WeightTable table;
    std::array<Sums256<1>, kRowsPerJob> sums;
    std::fill_n(sums.begin(), rows, Sums256<1>{});
    constexpr int kGroups = 4;
    const size_t stride = size_t(n) * 16;
    const uint16_t* x0 = x[0];
    if (groups >= kGroups) {
        const auto* initial = weights + size_t(first) * 16;
        const int bytes = std::min(512, rows * 16);
        for (int k = 0; k < kGroups; ++k) {
            const auto* group = initial + size_t(k) * stride;
            for (int offset = 0; offset < bytes; offset += 64)
                asm volatile("prfm pldl2keep, [%0]" :: "r"(group + offset));
        }
    }

    for (int g = 0; g < groups; g += kGroups) {
        const auto* w0 = weights + (size_t(g) * n + first) * 16;
        const auto* s = scales + size_t(first) * groups + g;
        const int chunk = std::min(kGroups, groups - g);

        if (chunk == 4) {
            const auto* w1 = w0 + stride;
            const auto* w2 = w0 + 2 * stride;
            const auto* w3 = w0 + 3 * stride;
            const auto a0 = load_activation_deinterleaved(x0 + (g + 0) * 32);
            const auto a1 = load_activation_deinterleaved(x0 + (g + 1) * 32);
            const auto a2 = load_activation_deinterleaved(x0 + (g + 2) * 32);
            const auto a3 = load_activation_deinterleaved(x0 + (g + 3) * 32);

            const bool do_pf = g + 2 * kGroups <= groups;

            const bool prefetch_scales = (g & 63) == 0;
            int row = 0;
            for (; row + 2 <= rows; row += 2) {
                if (do_pf && ((row & 3) == 0)) {
                    const auto* pf0 = w0 + 4 * stride;
                    const auto* pf1 = pf0 + stride;
                    const auto* pf2 = pf0 + 2 * stride;
                    const auto* pf3 = pf0 + 3 * stride;
                    asm volatile("prfm pldl2keep, [%0]" :: "r"(pf0 + row * 16));
                    asm volatile("prfm pldl2keep, [%0]" :: "r"(pf1 + row * 16));
                    asm volatile("prfm pldl2keep, [%0]" :: "r"(pf2 + row * 16));
                    asm volatile("prfm pldl2keep, [%0]" :: "r"(pf3 + row * 16));
                }
                if ((row & 3) == 0 && row + 16 < rows) {
                    const int offset = (row + 16) * 16;
                    asm volatile("prfm pldl1keep, [%0]" :: "r"(w0 + offset));
                    asm volatile("prfm pldl1keep, [%0]" :: "r"(w1 + offset));
                    asm volatile("prfm pldl1keep, [%0]" :: "r"(w2 + offset));
                    asm volatile("prfm pldl1keep, [%0]" :: "r"(w3 + offset));
                }
                if (prefetch_scales && row + 9 < rows) {
                    const auto* next_scale = s + size_t(row + 8) * groups;
                    asm volatile("prfm pldl2keep, [%0]" :: "r"(next_scale));
                    asm volatile("prfm pldl2keep, [%0]" :: "r"(next_scale + groups));
                }
                auto b0 = sums[row].lo[0],     t0 = sums[row].hi[0];
                auto b1 = sums[row + 1].lo[0], t1 = sums[row + 1].hi[0];
                const auto* sr0 = s + size_t(row) * groups;
                const auto* sr1 = sr0 + groups;
                accumulate_exact_nozip(b0, t0, decode_sve_nozip(w0 + row * 16,      table.scale[sr0[0]]), a0);
                accumulate_exact_nozip(b1, t1, decode_sve_nozip(w0 + row * 16 + 16, table.scale[sr1[0]]), a0);
                accumulate_exact_nozip(b0, t0, decode_sve_nozip(w1 + row * 16,      table.scale[sr0[1]]), a1);
                accumulate_exact_nozip(b1, t1, decode_sve_nozip(w1 + row * 16 + 16, table.scale[sr1[1]]), a1);
                accumulate_exact_nozip(b0, t0, decode_sve_nozip(w2 + row * 16,      table.scale[sr0[2]]), a2);
                accumulate_exact_nozip(b1, t1, decode_sve_nozip(w2 + row * 16 + 16, table.scale[sr1[2]]), a2);
                accumulate_exact_nozip(b0, t0, decode_sve_nozip(w3 + row * 16,      table.scale[sr0[3]]), a3);
                accumulate_exact_nozip(b1, t1, decode_sve_nozip(w3 + row * 16 + 16, table.scale[sr1[3]]), a3);
                sums[row].lo[0]     = b0; sums[row].hi[0]     = t0;
                sums[row + 1].lo[0] = b1; sums[row + 1].hi[0] = t1;
            }
            for (; row < rows; ++row) {
                auto b = sums[row].lo[0], t = sums[row].hi[0];
                const auto* sr = s + size_t(row) * groups;
                accumulate_exact_nozip(b, t, decode_sve_nozip(w0 + row * 16, table.scale[sr[0]]), a0);
                accumulate_exact_nozip(b, t, decode_sve_nozip(w1 + row * 16, table.scale[sr[1]]), a1);
                accumulate_exact_nozip(b, t, decode_sve_nozip(w2 + row * 16, table.scale[sr[2]]), a2);
                accumulate_exact_nozip(b, t, decode_sve_nozip(w3 + row * 16, table.scale[sr[3]]), a3);
                sums[row].lo[0] = b; sums[row].hi[0] = t;
            }
        } else {
            // Keep the same accumulator order for an incomplete K group.
            for (int k = 0; k < chunk; ++k) {
                const auto a = load_activation_deinterleaved(x0 + (g + k) * 32);
                const auto* wk = w0 + size_t(k) * stride;
                for (int row = 0; row < rows; ++row) {
                    const auto exp = s[size_t(row) * groups + k];
                    accumulate_exact_nozip(sums[row].lo[0], sums[row].hi[0],
                                           decode_sve_nozip(wk + row * 16, table.scale[exp]), a);
                }
            }
        }
    }

    for (int r = 0; r < rows; ++r) {
        const auto lo = svzip1_f32(sums[r].lo[0], sums[r].hi[0]);
        const auto hi = svzip2_f32(sums[r].lo[0], sums[r].hi[0]);
        y[0][r] = reduce256_single_token(lo, hi);
    }
}

}  // namespace

void prepare(float* input,uint16_t* packed,int size) {
    quant_sve(input,size);
    pack_input(input,packed,size);
}

void activate(const float* gate, const float* up, uint16_t* packed, int size) {
    std::array<float,32> values;
    for (int first=0; first<size; first+=32) {
        for (int j=0; j<32; ++j) {
            const float g=std::min(kt_match_npu::bf16(gate[first+j]),10.f);
            const float u=std::max(-10.f,std::min(kt_match_npu::bf16(up[first+j]),10.f));
            values[j]=kt_match_npu::bf16((g/(1.f+expf(-g)))*u);
        }
        prepare(values.data(),packed+first,32);
    }
}

void combine(const float* const* values, const float* scores, int routes,
             uint16_t* output, int size) {
    std::array<float,256> sum{};
    for (int route=0; route<routes; ++route) {
        const float score=kt_match_npu::bf16(scores[route]);
        for (int j=0; j<size; ++j) sum[j]+=kt_match_npu::bf16(values[route][j])*score;
    }
    for (int j=0; j<size; ++j) {
        const float value=kt_match_npu::bf16(sum[j]);
        uint32_t bits; std::memcpy(&bits,&value,sizeof(bits));
        output[j]=bits>>16;
    }
}

void project(const uint8_t* w,const uint8_t* s,int n,int groups,int first,int rows,
             const InputTile& x,const OutputTile& y,int tokens) {
    switch (tokens) {
    case 1: return project_rows<1>(w,s,n,groups,first,rows,x,y);
    case 2: return project_rows<2>(w,s,n,groups,first,rows,x,y);
    case 3: return project_rows<3>(w,s,n,groups,first,rows,x,y);
    case 4: return project_rows<4>(w,s,n,groups,first,rows,x,y);
    case 5: return project_rows<5>(w,s,n,groups,first,rows,x,y);
    case 6: return project_rows<6>(w,s,n,groups,first,rows,x,y);
    }
}

}  // namespace kt_nz::sve256
