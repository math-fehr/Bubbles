#pragma once

#include "../glad/glad.h"
#include <cuda_runtime.h>

#include <array>
#include <vector>

/**
 * POD containing resources used by CUDA and OpenGL to render frames.
 */
class InteropData {
public:
  InteropData(unsigned n_buffers);
  InteropData(const InteropData &) = delete;
  InteropData(InteropData &&interop);
  ~InteropData();

  /**
   * Return (width,height) of the buffer
   */
  std::pair<unsigned, unsigned> get_size();

  /**
   * Set the size of the buffers
   */
  void set_size(unsigned width, unsigned height);

  /**
   * Get the cuda array of the current buffer
   */
  cudaArray_const_t get_current_cuda_array();

  /**
   * Clear the curent buffer
   */
  void clear_buffer(std::array<float, 4> clear_color = {1.0f, 1.0f, 1.0f,
                                                        1.0f});

  /**
   * Copy the buffer to openGL default framebuffer
   */
  void blit_buffer();

  /**
   * Change buffer
   */
  void change_buffer();

private:
  // number of buffers
  unsigned n_buffers;
  // index of current buffer
  unsigned current_buffer;

  // size of the buffers
  unsigned width;
  unsigned height;

  // GL framebuffers and renderbuffers
  std::vector<GLuint> fb;
  std::vector<GLuint> rb;

  // CUDA graphics resources and array
  std::vector<cudaGraphicsResource_t> cgr;
  std::vector<cudaArray_t> ca;
};
