#pragma once

enum class TextureType { phong };

struct Phong {
  Color color;
  float ambiant_factor;
  float diffusion_factor;
};

struct Texture {
  TextureType type;
  union {
    Phong phong;
  };
};
