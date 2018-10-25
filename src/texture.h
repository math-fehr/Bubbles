#pragma once

#include "geom.h"

struct UniformColor {
  Color color;
};

struct CheckBoard {
  Color color1, color2;
  float n_subdivision;
};

enum class TextureType { uniform_color, checkerboard };

struct Texture {
  TextureType type;
  real ambiant_factor;
  real diffusion_factor;
  union {
    UniformColor uniform_color;
  };

  HD Color get_color(Vec2f uv) {
    switch (type) {
    case TextureType::uniform_color:
      return uniform_color.color;
    default:
      return Color{0.0f, 0.0f, 0.0f};
    }
  }
};
