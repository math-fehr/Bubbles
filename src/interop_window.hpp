#pragma once

#include "../glad/glad.h"
#include "interop.hpp"
#include <GLFW/glfw3.h>
#include <memory>
#include <functional>
#include <unordered_map>

using KeyCallback =
    std::function<void(GLFWwindow *window, int action, int mods)>;

using CursorCallback =
  std::function<void(GLFWwindow *window, double xupd, double yupd)>;
/**
 * Custom deleter for GLFWwindow
 */
inline void glfw_window_free(GLFWwindow *window) { glfwDestroyWindow(window); }

/**
 * Class used to render the frame using CUDA and OpenGL
 */
class InteropWindow {
public:
  /**
   * The constructor create a new window, and map the buffers to CUDA arrays.
   */
  InteropWindow(unsigned width, unsigned height);
  InteropWindow(const InteropWindow &) = delete;
  InteropWindow(InteropWindow &&) = delete;
  ~InteropWindow();

public:
  // The GLFW window
  std::unique_ptr<GLFWwindow, void (*)(GLFWwindow *)> window;

  // The POD containing resources used by CUDA and OpenGL
  InteropData interop_data;

  /**
   * The callbacks called when a key is pressed.
   * The map key is the key of the button pressed as defined by GLFW
   */
  std::unordered_map<int, KeyCallback> key_callbacks;
  CursorCallback cursor_callback;
};
