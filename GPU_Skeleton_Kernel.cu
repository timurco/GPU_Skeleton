#ifndef SDK_GPU_SKELETON
#define SDK_GPU_SKELETON

#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE

GF_KERNEL_FUNCTION(MainKernel,
                   ((const GF_PTR(float4))(inSrc))((GF_PTR(float4))(outDst)),
                   ((int)(inSrcPitch))((int)(inDstPitch))((int)(in16f))((unsigned int)(inWidth))((unsigned int)(inHeight))
                   // Variables
                   ((float)(inParameter))((float)(inTime)),
                   // Position
                   ((uint2)(inXY)(KERNEL_XY)))
{
  if (inXY.x < inWidth && inXY.y < inHeight)
  {
    //    A    B    G    R   <---- Inverted
    // {1.0, 1.0, 1.0, 1.0}
    //    W    X    Y    Z
    float2 iResolution = {(float)inWidth, (float)inHeight};
    float2 texCoord = {(float)inXY.x, (float)inXY.y};
    float2 uv = texCoord / iResolution;

    float2 st = uv;
    uint2 offset = uint2(st * iResolution);
    offset = clamp(offset, {0, 0}, {inWidth, inHeight});
    float4 fragColor = ReadFloat4(inSrc, offset.y * inSrcPitch + offset.x, !!in16f);
    fragColor.x += inParameter;
    fragColor.y = clamp(fragColor.y + inTime * 0.02f, 0.0f, 1.0f);

    WriteFloat4(fragColor, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
  }
}
#endif

#if __NVCC__

void Main_CUDA(float const *src, float *dst, unsigned int srcPitch,
               unsigned int dstPitch, int is16f,
               unsigned int width,
               unsigned int height,
               float parameter, float time)
{
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y, 1);

  MainKernel<<<gridDim, blockDim, 0>>>(
      (float4 const *)src, (float4 *)dst, srcPitch, dstPitch, is16f, width,
      height, parameter, time);

  cudaDeviceSynchronize();
}

#endif // GF_DEVICE_TARGET_HOST

#endif
