#pragma once

#include <array>
#include <cstdint>

namespace kt_nz {

inline constexpr int kTokenTile = 6;
inline constexpr int kRowsPerJob = 160;
using InputTile = std::array<const uint16_t*, kTokenTile>;
using OutputTile = std::array<float*, kTokenTile>;

#if defined(KT_NZ_SVE256)
namespace sve256 {
void prepare(float* input, uint16_t* packed, int size);
void project(const uint8_t* weights, const uint8_t* scales, int rows, int groups,
             int first, int count, const InputTile& input, const OutputTile& output, int tokens);
}  // namespace sve256
#endif

}  // namespace kt_nz
