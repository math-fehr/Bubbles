#include "interop_window.hpp"
#include "assert_cuda.hpp"

#include <iostream>

void glfw_error_callback(int error, const char *description) {
  std::cerr << description << std::endl;
}

void glfw_key_callback(GLFWwindow *window, int key, int scancode, int action,
                       int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
}

void glfw_window_size_callback(GLFWwindow *window, int width, int height) {
  InteropData *const interop =
      static_cast<InteropData *>(glfwGetWindowUserPointer(window));
  interop->set_size(width, height);
}

GLFWwindow *glfw_window_create_and_init(unsigned width, unsigned height) {
  GLFWwindow *window;

  glfwSetErrorCallback(glfw_error_callback);

  if (!glfwInit()) {
    exit(EXIT_FAILURE);
  }

  // No depth, stencil, or alpha buffer
  glfwWindowHint(GLFW_DEPTH_BITS, 0);
  glfwWindowHint(GLFW_STENCIL_BITS, 0);
  glfwWindowHint(GLFW_ALPHA_BITS, 0);

  // Use RBG
  glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

  // Give the gl context version number (the one glad use)
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

  window = glfwCreateWindow(width, height, "Bubbles", nullptr, nullptr);

  if (!window) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  // Make the window current on the current thread
  glfwMakeContextCurrent(window);

  // Set up GLAD by giving him the OpenGL context
  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

  // Don't use vsync
  glfwSwapInterval(0);

  // only copy rgb, and ignore alpha value
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

  return window;
}

InteropWindow::InteropWindow(unsigned width, unsigned height)
    : window(glfw_window_create_and_init(width, height), glfw_window_free),
      interop_data(2) {
  interop_data.set_size(width, height);

  glfwSetWindowUserPointer(window.get(), &interop_data);
  glfwSetKeyCallback(window.get(), glfw_key_callback);
  glfwSetFramebufferSizeCallback(window.get(), glfw_window_size_callback);
}

InteropWindow::~InteropWindow() {
  glfwDestroyWindow(window.get());
  glfwTerminate();
  cuda(DeviceReset());
}
