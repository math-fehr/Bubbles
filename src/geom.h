#pragma once
#include <array>
#include <cmath>
#include <optional>
#include <ostream>

#define HD __host__ __device__


using real = float;

// __     __        ____
// \ \   / /__  ___|___ \
//  \ \ / / _ \/ __| __) |
//   \ V /  __/ (__ / __/
//    \_/ \___|\___|_____|

template <class T> struct Vec2 {
  T x, y;
  HD Vec2 &operator+=(const Vec2 &o) {
    x += o.x;
    y += o.y;
    return *this;
  }
  HD Vec2 &operator-=(const Vec2 &o) {
    x -= o.x;
    y -= o.y;
    return *this;
  }
  HD Vec2 &operator*=(T f) {
    x *= f;
    y *= f;
    return *this;
  }
  HD Vec2 &operator/=(T f) {
    x /= f;
    y /= f;
    return *this;
  }
  HD friend Vec2 operator+(Vec2 v, const Vec2 &v2) { return v += v2; }
  HD friend Vec2 operator-(Vec2 v, const Vec2 &v2) { return v -= v2; }
  HD friend Vec2 operator*(Vec2 v, float f) { return v *= f; }
  HD friend Vec2 operator*(float f, Vec2 v) { return v *= f; }
  HD friend Vec2 operator/(Vec2 v, float f) { return v /= f; }

  HD T operator|(const Vec2 &v) { return x * v.x + y * v.y; }
  HD T norm2() { return *this | *this; }
  HD T norm() { return sqrt(norm2()); }
  HD Vec2 normalizeeq() { return *this / norm(); }
  HD Vec2 normalize() {
    Vec2 o = *this;
    return o.normalizeeq();
  }

  HD friend std::ostream &operator<<(std::ostream &out, Vec2 v) {
    return out << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  }
};

using Vec2f = Vec2<real>;

// __     __        _____
// \ \   / /__  ___|___ /
//  \ \ / / _ \/ __| |_ \
//   \ V /  __/ (__ ___) |
//    \_/ \___|\___|____/

template <class T> struct Vec3 {
  T x, y, z;
  HD Vec3 &operator+=(const Vec3 &o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  HD Vec3 &operator-=(const Vec3 &o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
  }
  HD Vec3 &operator*=(T f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
  }
  HD Vec3 &operator/=(T f) {
    x /= f;
    y /= f;
    z /= f;
    return *this;
  }
  HD friend Vec3 operator+(Vec3 v, const Vec3 &v2) { return v += v2; }
  HD friend Vec3 operator-(Vec3 v, const Vec3 &v2) { return v -= v2; }
  HD friend Vec3 operator*(Vec3 v, float f) { return v *= f; }
  HD friend Vec3 operator*(float f, Vec3 v) { return v *= f; }
  HD friend Vec3 operator/(Vec3 v, float f) { return v /= f; }

  HD Vec3 operator^(const Vec3 &v) {
    return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
  }
  HD T operator|(const Vec3 &v) { return x * v.x + y * v.y + z * v.z; }
  HD T norm2() { return *this | *this; }
  HD T norm() { return sqrt(norm2()); }
  HD Vec3 normalizeeq() { return *this / norm(); }
  HD Vec3 normalize() {
    Vec3 o = *this;
    return o.normalizeeq();
  }

  HD friend std::ostream &operator<<(std::ostream &out, Vec3 v) {
    return out << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  }
};

using Vec3f = Vec3<real>;

//  ____
// |  _ \ __ _ _   _
// | |_) / _` | | | |
// |  _ < (_| | |_| |
// |_| \_\__,_|\__, |
//             |___/

template <class T> struct Ray {
  Vec3<T> orig, dir;
  HD Ray &normalize() { dir.normalize(); }
  HD T projindex(const Vec3<T> &vec) {
    auto dir2 = dir;
    dir2.norm();
    return (vec - orig) | dir2;
  }
  HD Vec3f projpoint(const Vec3<T> &vec) { return *this(proindex(vec)); }

  HD Vec3<T> operator()(T index) { return orig + index * dir; }

  HD friend std::ostream &operator<<(std::ostream &out, Ray v) {
    return out << v.orig << "->" << v.dir;
  }
};

using Rayf = Ray<real>;

//  __  __       _   _____
// |  \/  | __ _| |_|___ /
// | |\/| |/ _` | __| |_ \
// | |  | | (_| | |_ ___) |
// |_|  |_|\__,_|\__|____/

template <class T> struct Mat3 {
  Vec3<T> r[3];
  HD Mat3 &operator+=(const Mat3 &o) {
    r[0] += o.r[0];
    r[1] += o.r[1];
    r[2] += o.r[2];
    return *this;
  }
  HD Mat3 &operator-=(const Mat3 &o) {
    r[0] -= o.r[0];
    r[1] -= o.r[1];
    r[2] -= o.r[2];
    return *this;
  }
  HD Mat3 &operator*=(T f) {
    r[0] *= f;
    r[1] *= f;
    r[2] *= f;
    return *this;
  }
  HD Mat3 &operator/=(T f) {
    r[0] /= f;
    r[1] /= f;
    r[2] /= f;
    return *this;
  }
  HD friend Mat3 operator+(Mat3 v, const Mat3 &v2) { return v += v2; }
  HD friend Mat3 operator-(Mat3 v, const Mat3 &v2) { return v -= v2; }
  HD friend Mat3 operator*(Mat3 v, float f) { return v *= f; }
  HD friend Mat3 operator*(float f, Mat3 v) { return v *= f; }
  HD friend Mat3 operator/(Mat3 v, float f) { return v /= f; }

  HD Vec3<T> operator*(const Vec3<T> &v) {
    return Vec3<T>{r[0] | v, r[1] | v, r[2] | v};
  }

  HD friend std::ostream &operator<<(std::ostream &out, Mat3 v) {
    return out << "[" << v.r[0] << "\n " << v.r[1] << "\n" <<  v.r[2] <<"]";
  }
};

using Mat3f = Mat3<real>;

//   ____      _
//  / ___|___ | | ___  _ __
// | |   / _ \| |/ _ \| '__|
// | |__| (_) | | (_) | |
//  \____\___/|_|\___/|_|

struct Color {
  real r, g, b;
  HD inline Color &clamp() {
    if (r < 0) r = 0;
    if (g < 0) g = 0;
    if (b < 0) b = 0;
    if (r > 1) r = 1;
    if (g > 1) g = 1;
    if (b > 1) b = 1;
    return *this;
  }
  HD std::array<char, 3> to8bit() {
    clamp();
    return std::array<char, 3>{char(r * 255), char(g * 255), char(b * 255)};
  }
  HD inline Color &operator+=(const Color &o) {
    r += o.r;
    g += o.g;
    b += o.b;
    return *this;
  }
  HD inline Color &operator-=(const Color &o) {
    r -= o.r;
    g -= o.g;
    b -= o.b;
    return *this;
  }
  HD inline Color &operator*=(const Color &o) {
    r *= o.r;
    g *= o.g;
    b *= o.b;
    return *this;
  }
  HD inline Color &operator*=(real f) {
    r *= f;
    g *= f;
    b *= f;
    return *this;
  }
  HD inline Color pow(real alpha) {
    return Color{std::pow(r, alpha), std::pow(r, alpha), std::pow(r, alpha)};
  }

  HD friend inline Color operator+(Color v, const Color v2) { return v += v2; }
  HD friend inline Color operator-(Color v, const Color v2) { return v -= v2; }
  HD friend inline Color operator*(Color v, const Color v2) { return v *= v2; }
  HD friend inline Color operator*(Color v, real f) { return v *= f; }
  HD friend inline Color operator*(real f, Color v) { return v *= f; }


  HD friend std::ostream &operator<<(std::ostream &out, Color c) {
    return out << "(" << c.r << ", " << c.g << ", " << c.b <<")";
  }

};


struct Sphere {
  Vec3f center;
  real radius2;
  Color color;
  HD Sphere(Vec3f center, real radius,Color color)
    : center(center),radius2(radius*radius),color(color){}
  HD real inter(Rayf ray) {
    real pi = ray.projindex(center);
    Vec3f pn = ray(pi);
    real n2 = (center - pn).norm2();
    if (n2 > radius2){
      return -1;
    }
    else{
      return pi - sqrt(radius2 - n2);
    }
  }
};

