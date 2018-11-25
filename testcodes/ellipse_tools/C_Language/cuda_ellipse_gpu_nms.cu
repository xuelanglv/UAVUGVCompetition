
#include "ellipse_gpu_nms.hpp"
#include "cuda_ellipse_overlaps.cuh"
#include <vector>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;



__device__ inline float devRotateIoU(float const * const region1, float const * const region2) {
  
  
  if((fabs(region1[0] - region2[0]) < 1e-5) && (fabs(region1[1] - region2[1]) < 1e-5) && (fabs(region1[2] - region2[2]) < 1e-5) && (fabs(region1[3] - region2[3]) < 1e-5) && (fabs(region1[4] - region2[4]) < 1e-5)) {
    return 1.0;
  }
  
  double elp1[5], elp2[5], result;
  for(int i=0;i<5;i++)
  {
        elp1[i] = region1[i];
        elp2[i] = region2[i];
  }
  CalculateOverlap(elp1, elp2, &result);
  
  if(result < 0) {
    result = 0.0;
  }
  return float(result);
  
}


__global__ void rotate_nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 6];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 6 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 0];
    block_boxes[threadIdx.x * 6 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 1];
    block_boxes[threadIdx.x * 6 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 2];
    block_boxes[threadIdx.x * 6 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 3];
    block_boxes[threadIdx.x * 6 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 4];
    block_boxes[threadIdx.x * 6 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 5];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 6;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devRotateIoU(cur_box, block_boxes + i * 6) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

void _rotate_nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id) {
  _set_device(device_id);

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  rotate_nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
}
