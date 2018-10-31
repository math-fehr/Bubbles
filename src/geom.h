#pragma once
#include <array>
#include <cmath>
#include <optional>
#include <ostream>

#define HD __host__ __device__

//                 _
//  _ __ ___  __ _| |
// | '__/ _ \/ _` | |
// | | |  __/ (_| | |
// |_|  \___|\__,_|_|

using real = float;

inline HD real clamp(real a, real mini = 0, real maxi = 1) {
  return min(maxi, max(mini, a));
}

inline HD real sign(real a) {
  return a >= 0 ? 1 : -1;
}

inline HD real signz(real a) {
  return a >= 0 ? 1 : 0;
}

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

  HD T operator|(const Vec2 &v) const { return x * v.x + y * v.y; }
  HD T norm2() const { return *this | *this; }
  HD T norm() const { return sqrt(norm2()); }
  HD Vec2 &normalize() { return *this /= norm(); }
  HD Vec2 normalized() const { return *this / norm(); }

  friend std::ostream &operator<<(std::ostream &out, Vec2 v) {
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
  HD Vec3 operator-() const { return Vec3{-x, -y, -z}; }
  HD friend Vec3 operator+(Vec3 v, const Vec3 &v2) { return v += v2; }
  HD friend Vec3 operator-(Vec3 v, const Vec3 &v2) { return v -= v2; }
  HD friend Vec3 operator*(Vec3 v, float f) { return v *= f; }
  HD friend Vec3 operator*(float f, Vec3 v) { return v *= f; }
  HD friend Vec3 operator/(Vec3 v, float f) { return v /= f; }

  HD bool operator<(Vec3 v) const { return x < v.x && y < v.y && z < v.z; }
  HD bool operator>(Vec3 v) const { return x > v.x && y > v.y && z > v.z; }
  HD bool operator<=(Vec3 v) const { return x <= v.x && y <= v.y && z <= v.z; }
  HD bool operator>=(Vec3 v) const { return x >= v.x && y >= v.y && z >= v.z; }

  HD Vec3 operator^(const Vec3 &v) const {
    return Vec3{y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
  }
  HD T operator|(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }
  HD T norm2() const { return *this | *this; }
  HD T norm() const { return sqrt(norm2()); }
  HD Vec3 &normalize() { return *this /= norm(); }
  HD Vec3 normalized() const { return *this / norm(); }

  friend std::ostream &operator<<(std::ostream &out, Vec3 v) {
    return out << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  }
};

using Vec3f = Vec3<real>;

static Vec3f X{1, 0, 0};
static Vec3f Y{0, 1, 0};
static Vec3f Z{0, 0, 1};

//  ____
// |  _ \ __ _ _   _
// | |_) / _` | | | |
// |  _ < (_| | |_| |
// |_| \_\__,_|\__, |
//             |___/

template <class T> struct Ray {
  Vec3<T> orig, dir, inv_dir; // dir must be normalized
  int sign[3];                // The sign of dir components
  Ray() = default;
  HD Ray(Vec3<T> orig, Vec3<T> dir_)
      : orig(orig), dir(dir_.normalized()), inv_dir{1.0f / dir.x, 1.0f / dir.y,
                                                    1.0f / dir.z},
        sign{dir.x > 0, dir.y > 0, dir.z > 0} {}
  HD T projindex(const Vec3<T> &vec) const { return (vec - orig) | dir; }
  HD Vec3f projpoint(const Vec3<T> &vec) const {
    return (*this)(proindex(vec));
  }

  HD Vec3<T> operator()(T index) const { return orig + index * dir; }

  friend std::ostream &operator<<(std::ostream &out, Ray v) {
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
  // WARNING
  // Initialization with braces : row vectors
  // Initialization with parenthesis : column vectors
  Vec3<T> r[3]; // row vectors
  Mat3(Vec3<T> c0, Vec3<T> c1, Vec3<T> c2)
      : r{Vec3<T>{c0.x, c1.x, c2.x}, Vec3<T>{c0.y, c2.y, c2.y},
          Vec3<T>{c0.z, c1.z, c2.z}} {}
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

  HD Vec3<T> operator*(const Vec3<T> &v) const {
    return Vec3<T>{r[0] | v, r[1] | v, r[2] | v};
  }

  friend std::ostream &operator<<(std::ostream &out, Mat3 v) {
    return out << "[" << v.r[0] << "\n " << v.r[1] << "\n" << v.r[2] << "]";
  }
};

using Mat3f = Mat3<real>;

//   ___              _                  _
//  / _ \ _   _  __ _| |_ ___ _ __ _ __ (_) ___  _ __
// | | | | | | |/ _` | __/ _ \ '__| '_ \| |/ _ \| '_ \
// | |_| | |_| | (_| | ||  __/ |  | | | | | (_) | | | |
//  \__\_\\__,_|\__,_|\__\___|_|  |_| |_|_|\___/|_| |_|

template <typename T> struct Quat {
  float w;
  Vec3<T> v;
  Quat(Vec3<T> v) : w(0), v(v) {}

  HD Quat &operator+=(const Quat &o) {
    w += o.w;
    v += o.v;
    return *this;
  }
  HD Quat &operator-=(const Quat &o) {
    w -= o.w;
    v -= o.v;
    return *this;
  }
  HD Quat &operator*=(T f) {
    w *= f;
    v *= f;
    return *this;
  }
  HD Quat &operator/=(T f) {
    w /= f;
    v /= f;
    return *this;
  }
  HD Quat operator-() const { return Quat{-w, -v}; }
  HD friend Quat operator+(Quat v, const Quat &v2) { return v += v2; }
  HD friend Quat operator-(Quat v, const Quat &v2) { return v -= v2; }
  HD friend Quat operator*(Quat v, float f) { return v *= f; }
  HD friend Quat operator*(float f, Quat v) { return v *= f; }
  HD friend Quat operator/(Quat v, float f) { return v /= f; }

  HD Quat conj() const { return Quat{v, -w}; }
  HD Quat &conjeq() const {
    w = -w;
    return *this;
  }

  HD Quat operator*(const Quat &o) const {
    return Quat{w * o.w - (v | o.v), w * o.v + o.w * v + (v ^ o.v)};
  }

  HD Vec3<T> apply(const Vec3<T> v) { return *this * v * conj(); }

  HD Mat3<T> toMat() { return Mat(apply(X), apply(Y), apply(Z)); }

  HD T norm2() const { return w * w + v.norm2(); }
  HD T norm() const { return sqrt(norm2()); }
  HD Quat &normalize() { return *this /= norm(); }
  HD Quat normalized() const { return *this / norm(); }

  friend std::ostream &operator<<(std::ostream &out, Quat q) {
    return out << q.w << " + " << q.im.x << "i + " << q.im.y << "j + " << q.im.z
               << "k";
  }
};

using Quatf = Quat<real>;

//   ____      _
//  / ___|___ | | ___  _ __
// | |   / _ \| |/ _ \| '__|
// | |__| (_) | | (_) | |
//  \____\___/|_|\___/|_|

// The discrete value of a pixel

using uchar = unsigned char;
struct RGBA {
  uchar r : 8;
  uchar g : 8;
  uchar b : 8;
  uchar a : 8;
};

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
  HD RGBA to8bit(real gamma) {
    clamp();
    return RGBA{uchar(std::pow(r, gamma) * 255),
                uchar(std::pow(g, gamma) * 255),
                uchar(std::pow(b, gamma) * 255), 0};
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
  HD inline Color &operator/=(real f) {
    r /= f;
    g /= f;
    b /= f;
    return *this;
  }
  HD inline Color pow(real alpha) {
    return Color{std::pow(r, alpha), std::pow(r, alpha), std::pow(r, alpha)};
  }

  HD friend inline Color operator+(Color v, const Color v2) { return v += v2; }
  HD friend inline Color operator-(Color v, const Color v2) { return v -= v2; }
  HD friend inline Color operator*(Color v, const Color v2) { return v *= v2; }
  HD friend inline Color operator*(Color v, real f) { return v *= f; }
  HD friend inline Color operator/(Color v, real f) { return v /= f; }
  HD friend inline Color operator*(real f, Color v) { return v *= f; }

  friend std::ostream &operator<<(std::ostream &out, Color c) {
    return out << "(" << c.r << ", " << c.g << ", " << c.b << ")";
  }
};

static Color white{1,1,1};
static Color black{0,0,0};
static Color red{1,0,0};
static Color green{0,1,0};
static Color blue{0,0,1};
