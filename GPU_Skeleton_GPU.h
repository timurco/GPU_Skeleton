//
//  GPU_Skeleton_GPU.hpp
//  GPU_Skeleton
//
//  Created by Tim Constantinov on 25.04.2023.
//
#pragma once
#ifndef GPU_Skeleton_GPU_H
#define GPU_Skeleton_GPU_H

#include <stdio.h>
#include "GPU_Skeleton.h"

#if _WIN32
#include <CL/cl.h>
#define HAS_METAL 0
#else
#include <OpenCL/cl.h>
#define HAS_METAL 1
#include <Metal/Metal.h>
#include "GPU_Skeleton_Kernel.metal.h"
#endif

#if HAS_METAL
/*
 ** Plugins must not rely on a host autorelease pool.
 ** Create a pool if autorelease is used, or Cocoa convention calls, such as
 *Metal, might internally autorelease.
 */
struct ScopedAutoreleasePool {
  ScopedAutoreleasePool() : mPool([[NSAutoreleasePool alloc] init]) {}

  ~ScopedAutoreleasePool() { [mPool release]; }

  NSAutoreleasePool *mPool;
};
#endif

inline PF_Err CL2Err(cl_int cl_result) {
  if (cl_result == CL_SUCCESS) {
    return PF_Err_NONE;
  } else {
    // set a breakpoint here to pick up OpenCL errors.
    return PF_Err_INTERNAL_STRUCT_DAMAGED;
  }
}

#define CL_ERR(FUNC) ERR(CL2Err(FUNC))

#define CUDA_CHECK(FUNC)                                                                                          \
  do {                                                                                                            \
    if (!err) {                                                                                                   \
      cudaError_t cerr = (FUNC);                                                                                  \
      if (cerr != cudaSuccess) {                                                                                  \
        (*in_dataP->utils->ansi.sprintf)(out_dataP->return_msg, "GPU Assert:\r\n%s\n", cudaGetErrorString(cerr)); \
        out_dataP->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;                                                 \
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;                                                                     \
      }                                                                                                           \
    }                                                                                                             \
  } while (0)

extern void Main_CUDA(float const* src,
                      float* dst,
                      cudaArray* d_envArray, cudaChannelFormatDesc channelDesc,
                      unsigned int srcPitch,
                      unsigned int dstPitch,
                      int is16f,
                      unsigned int width,
                      unsigned int height, float parameter, float time);

// GPU data initialized at GPU setup and used during render.
struct OpenCLGPUData {
  cl_kernel main_kernel;
};

#if HAS_METAL
struct MetalGPUData {
  id<MTLComputePipelineState> main_kernel;
};
#endif

typedef struct {
  int   mSrcPitch;
  int   mDstPitch;
  int   m16f;
  int   mWidth;
  int   mHeight;
  float mParameter;
  float mTime;
} InputKernelParams;

PF_Err GPUDeviceSetup(PF_InData *in_dataP, PF_OutData *out_dataP, PF_GPUDeviceSetupExtra *extraP);

PF_Err GPUDeviceSetdown(PF_InData *in_dataP, PF_OutData *out_dataP, PF_GPUDeviceSetdownExtra *extraP);

PF_Err SmartRenderGPU(PF_InData *in_dataP, PF_OutData *out_dataP, PF_PixelFormat pixel_format,
                      PF_EffectWorld *input_worldP, PF_EffectWorld *output_worldP, PF_EffectWorld *layer_worldP,
                      PF_SmartRenderExtra *extraP, PluginInputParams *infoP);

#endif /* GPU_Skeleton_GPU_H */
