#pragma once

#include "../glad/glad.h"
#include "interop.hpp"
#include <GLFW/glfw3.h>
#include <memory>

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
};
