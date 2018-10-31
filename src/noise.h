#pragma once

#include "geom.h"

// We want the object on the GPU and on the CPU
#ifdef __CUDA_ARCH__
__constant__
#endif
    // A random permutation
    const int perm[256] = {
        151, 160, 137, 91,  90,  15,  131, 13,  201, 95,  96,  53,  194, 233,
        7,   225, 140, 36,  103, 30,  69,  142, 8,   99,  37,  240, 21,  10,
        23,  190, 6,   148, 247, 120, 234, 75,  0,   26,  197, 62,  94,  252,
        219, 203, 117, 35,  11,  32,  57,  177, 33,  88,  237, 149, 56,  87,
        174, 20,  125, 136, 171, 168, 68,  175, 74,  165, 71,  134, 139, 48,
        27,  166, 77,  146, 158, 231, 83,  111, 229, 122, 60,  211, 133, 230,
        220, 105, 92,  41,  55,  46,  245, 40,  244, 102, 143, 54,  65,  25,
        63,  161, 1,   216, 80,  73,  209, 76,  132, 187, 208, 89,  18,  169,
        200, 196, 135, 130, 116, 188, 159, 86,  164, 100, 109, 198, 173, 186,
        3,   64,  52,  217, 226, 250, 124, 123, 5,   202, 38,  147, 118, 126,
        255, 82,  85,  212, 207, 206, 59,  227, 47,  16,  58,  17,  182, 189,
        28,  42,  223, 183, 170, 213, 119, 248, 152, 2,   44,  154, 163, 70,
        221, 153, 101, 155, 167, 43,  172, 9,   129, 22,  39,  253, 19,  98,
        108, 110, 79,  113, 224, 232, 178, 185, 112, 104, 218, 246, 97,  228,
        251, 34,  242, 193, 238, 210, 144, 12,  191, 179, 162, 241, 81,  51,
        145, 235, 249, 14,  239, 107, 49,  192, 214, 31,  181, 199, 106, 157,
        184, 84,  204, 176, 115, 121, 50,  45,  127, 4,   150, 254, 138, 236,
        205, 93,  222, 114, 67,  29,  24,  72,  243, 141, 128, 195, 78,  66,
        215, 61,  156, 180};

// We want the object on the GPU and on the CPU
#ifdef __CUDA_ARCH__
__constant__
#endif
    // The gradients that will be used in the perlin algorithm
    const Vec3f grad3[256] = {{-0.0101255, -0.332831, -0.942932},
                             {0.322723, -0.869862, -0.373081},
                             {0.672098, 0.0938854, -0.734486},
                             {0.268604, -0.91964, -0.286554},
                             {0.252045, -0.694893, -0.673496},
                             {-0.145087, -0.925064, -0.351008},
                             {-0.557681, 0.655993, 0.508591},
                             {0.317359, -0.164076, -0.934003},
                             {0.99561, 0.0827227, -0.0437889},
                             {-0.868747, -0.358256, -0.341951},
                             {0.129141, 0.757184, -0.640309},
                             {-0.306622, 0.174409, -0.935716},
                             {-0.964277, 0.16877, 0.204174},
                             {-0.654047, -0.301881, 0.693607},
                             {0.935072, 0.15054, -0.320901},
                             {0.473575, -0.60944, 0.635854}};

// A fast hash function
inline HD Vec3f hash(int px, int py, int pz) {
  return grad3[perm[(px + perm[(py + perm[pz & 0xFF]) & 0xFF]) &
                    0xFF] &
               0x0F];
}

// The interpolation done in the perlin algorithm
inline HD real perlin(Vec3f v) {
  Vec3f v_floor = v.floor();
  int i0[3]{(int)v_floor.x, (int)v_floor.y, (int)v_floor.z};
  int i1[3]{i0[0]+1, i0[1]+1, i0[2]+1};
  Vec3f f = v.frac();

  real a000 = hash(i0[0],i0[1],i0[2]) | (f - Vec3f{0.f, 0.f, 0.f});
  real a001 = hash(i0[0],i0[1],i1[2]) | (f - Vec3f{0.f, 0.f, 1.f});
  real a010 = hash(i0[0],i1[1],i0[2]) | (f - Vec3f{0.f, 1.f, 0.f});
  real a011 = hash(i0[0],i1[1],i1[2]) | (f - Vec3f{0.f, 1.f, 1.f});
  real a100 = hash(i1[0],i0[1],i0[2]) | (f - Vec3f{1.f, 0.f, 0.f});
  real a101 = hash(i1[0],i0[1],i1[2]) | (f - Vec3f{1.f, 0.f, 1.f});
  real a110 = hash(i1[0],i1[1],i0[2]) | (f - Vec3f{1.f, 1.f, 0.f});
  real a111 = hash(i1[0],i1[1],i1[2]) | (f - Vec3f{1.f, 1.f, 1.f});

  real a00 = smoothstep(a000, a001, f.z);
  real a01 = smoothstep(a010, a011, f.z);
  real a10 = smoothstep(a100, a101, f.z);
  real a11 = smoothstep(a110, a111, f.z);

  real a0 = smoothstep(a00, a01, f.y);
  real a1 = smoothstep(a10, a11, f.y);

  return smoothstep(a0, a1, f.x);
}

// The application of multiple perlin noise.
inline HD real fractal_perlin(Vec3f pos, unsigned n_octaves, real omega,
                              real alpha) {
  real value = 0.0;
  real sum_alpha = 0.0;
  real alpha_i = 1.0;
  for (int i = 0; i < n_octaves; ++i) {
    alpha_i *= alpha;
    sum_alpha += alpha_i;
    value += alpha_i * perlin(pos);
    pos = pos * omega;
  }
  return ((value / sum_alpha) + 0.2) / 0.4;
}
