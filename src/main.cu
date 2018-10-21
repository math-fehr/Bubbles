#include "assert_cuda.hpp"
#include "interop_window.hpp"
#include "kernel.cuh"
#include <cuda_gl_interop.h>
#include "geom.h"
#include "object.h"

static void show_fps_and_window_size(GLFWwindow *window) {
  // fps counter in static variables
  static double previous_time = 0.0;
  static int frame_count = 0;

  const double current_time = glfwGetTime();
  const double elapsed = current_time - previous_time;

  if (elapsed > 0.5) {
    previous_time = current_time;

    const double fps = (double)frame_count / elapsed;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    char tmp[64];
    sprintf(tmp, "(%u x %u) - FPS: %.2f", width, height, fps);

    glfwSetWindowTitle(window, tmp);

    frame_count = 0;
  }

  frame_count++;
}

int main(int argc, char *argv[]) {

  std::vector<Object> objects;
  for(float i = 0.f; i < 9.99f; i+=0.1f) {
    Vec3f pos{i - 0.5f, i - 0.5f, i - 0.5f};
    float radius = i*0.1f;
    Color color{i/10, i/10, i/10};
    Object object;
    object.color = color;
    object.type = ObjectType::sphere;
    object.sphere = Sphere{pos,radius};
    objects.push_back(object);
  }
  Object object;
  object.color = Color{1.0f,0.0f,0.0f};
  object.type = ObjectType::plane;
  object.plane = Plane(Vec3f{1.0f,0.0f,0.0f}, 100.0f);
  objects.push_back(object);

  Object* d_objects = nullptr;
  cuda(Malloc(&d_objects, sizeof(Object) * objects.size()));
  cuda(Memcpy(d_objects, objects.data(), sizeof(Object) * objects.size(), cudaMemcpyHostToDevice));

  InteropWindow interop_window(640, 480);

  // Main loop
  while (!glfwWindowShouldClose(interop_window.window.get())) {
    show_fps_and_window_size(interop_window.window.get());

    // Execute the CUDA code
    unsigned width, height;
    std::tie(width, height) = interop_window.interop_data.get_size();
    kernel_launcher(interop_window.interop_data.get_current_cuda_array(), width,
                    height, d_objects, objects.size());

    // Switch buffers
    interop_window.interop_data.blit_buffer();
    interop_window.interop_data.change_buffer();
    glfwSwapBuffers(interop_window.window.get());

    // Get events
    glfwPollEvents();
  }

  exit(EXIT_SUCCESS);
}
