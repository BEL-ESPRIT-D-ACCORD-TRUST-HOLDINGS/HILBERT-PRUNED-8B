/* ops_common.h - scalar math shared by the CPU reference and the CUDA kernels,
 * so both backends evaluate the same formulas (SPEC.md section 4). */
#ifndef OPS_COMMON_H
#define OPS_COMMON_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define SI static __host__ __device__ __forceinline__
#else
#define SI static inline
#endif

SI float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* Round to nearest even; NaN stays NaN. */
SI uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

SI float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff, u;
    if (exp == 0) {
        if (man == 0) {
            u = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                exp--;
            }
            u = sign | (exp << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7f800000u | (man << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

SI float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
SI float siluf_(float x) { return x / (1.0f + expf(-x)); }
/* torch.nn.functional.softplus with beta=1, threshold=20 */
SI float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

#endif
