#pragma once

#include "geom.h"

struct UniformColor {
  Color color;
};

struct CheckBoard {
  Color color1, color2;
  float n_subdivision;
};

enum class TextureType { uniform_color, checkboard };

struct Texture {
  TextureType type;
  real ambiant_factor;
  real diffusion_factor;
  union {
    UniformColor uniform_color;
    CheckBoard checkboard;
  };

  HD Color get_color(Vec2f uv) const {
    switch (type) {
    case TextureType::uniform_color:
      return uniform_color.color;
    case TextureType::checkboard: {
      int a = uv.x * checkboard.n_subdivision * 2.0f;
      int b = uv.y * checkboard.n_subdivision * 2.0f;
      if ((a + b) % 2) {
        return checkboard.color1;
      } else {
        return checkboard.color2;
      }
    }
    default:
      return Color{0.0f, 0.0f, 0.0f};
    }
  }
};
