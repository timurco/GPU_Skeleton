#ifndef SDK_GPU_SKELETON
#define SDK_GPU_SKELETON

#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_METAL
#  define make_float2(inX, inY) (float2){(float)(inX), (float)(inY)}
#  define make_uint2(inX, inY)  (uint2){(uint)(inX), (uint)(inY)}
#endif

#if GF_DEVICE_TARGET_DEVICE
GF_KERNEL_FUNCTION(MainKernel,
                   ((const GF_PTR(float4))(inSrc))((GF_PTR(float4))(outDst)),
                   ((int)(inSrcPitch))((int)(inDstPitch))((int)(in16f))((unsigned int)(inWidth))((unsigned int)(inHeight))
                   // Variables
                   ((float)(inParameter))((float)(inTime)),
                   // Position
                   ((uint2)(inXY)(KERNEL_XY)))
{
  if (inXY.x < inWidth && inXY.y < inHeight) {
    //    A    B    G    R   <---- Inverted
    // {1.0, 1.0, 1.0, 1.0}
    //    W    X    Y    Z
    float2 iResolution = {(float)inWidth, (float)inHeight};
    float2 texCoord = {(float)inXY.x, (float)inXY.y};
    float2 uv = {
      texCoord.x / iResolution.x,
      texCoord.y / iResolution.y
    };
    
    const float pixelSize = 1.0f / iResolution.x;
    
    float2 st = uv;
    st.x = st.x + cos(st.y*100.0 + inTime) * 5*pixelSize; // Wave Moves
    
    uint2 offset = {(uint)(st.x * iResolution.x),
                    (uint)(st.y * iResolution.y)};
    offset.x = clamp(offset.x, (uint)(0), inWidth);
    offset.y = clamp(offset.y, (uint)(0), inHeight);
  
    float4 fragColor = ReadFloat4(inSrc, offset.y * inSrcPitch + offset.x, !!in16f);
    fragColor.x += inParameter;

    // standard UV GLSL Gradient
    fragColor.z = uv.x;
    fragColor.y = uv.y;

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
