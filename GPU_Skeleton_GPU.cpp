//
//  GPU_Skeleton_GPU.cpp
//  GPU_Skeleton
//
//  Created by Tim Constantinov on 25.04.2023.
//

#include "GPU_Skeleton_GPU.h"

// static BOOL didSaveShaderFile = NO;

#if HAS_METAL
PF_Err NSError2PFErr(NSError *inError) {
  if (inError) {
    return PF_Err_INTERNAL_STRUCT_DAMAGED;  // For debugging, uncomment above
                                            // line and set breakpoint here
  }
  return PF_Err_NONE;
}
#endif  // HAS_METAL

static size_t RoundUp(size_t inValue, size_t inMultiple) {
  return inValue ? ((inValue + inMultiple - 1) / inMultiple) * inMultiple : 0;
}

static size_t DivideRoundUp(size_t inValue, size_t inMultiple) {
  return inValue ? (inValue + inMultiple - 1) / inMultiple : 0;
}

/**

PF_Err GPUDeviceSetup
This function creates an OpenCL/Metal/CUDA device
so that the plugin can utilize the graphics processor for image processing.
The function sets up the connection with the graphics processor and compiles the
plugin kernel.

PF_Err GPUDeviceSetdown
This function removes the device created in GPUDeviceSetup.
It frees up the resources occupied by the plugin kernel
so that other plugins can utilize them.

PF_Err SmartRenderGPU
This function performs the rendering of pixels from the input image
to the output image using the graphics processor.
The input image is read from the graphics processor memory,
and the result is written to the output image.
This function takes images, the graphics processor device, and plugin parameters
as input parameters.

*/

PF_Err GPUDeviceSetup(PF_InData *in_dataP, PF_OutData *out_dataP,
                      PF_GPUDeviceSetupExtra *extraP) {
  PF_Err err = PF_Err_NONE;
  AEGP_SuiteHandler suites(in_dataP->pica_basicP);
  auto *globalData = reinterpret_cast<GlobalData *>(
      suites.HandleSuite1()->host_lock_handle(in_dataP->global_data));

  PF_GPUDeviceInfo device_info;
  AEFX_CLR_STRUCT(device_info);

  AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
      AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite,
                                        kPFHandleSuiteVersion1, out_dataP);

  AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpuDeviceSuite =
      AEFX_SuiteScoper<PF_GPUDeviceSuite1>(
          in_dataP, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_dataP);

  gpuDeviceSuite->GetDeviceInfo(in_dataP->effect_ref,
                                extraP->input->device_index, &device_info);

  // Load and compile the kernel - a real plugin would cache binaries to disk

  if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
    // Nothing to do here. CUDA Kernel statically linked
    out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
  } else if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {
    PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(OpenCLGPUData));
    OpenCLGPUData *cl_gpu_data = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

    cl_int result = CL_SUCCESS;

    char const *k16fString = "#define GF_OPENCL_SUPPORTS_16F 0\n";

    size_t sizes[] = {strlen(k16fString),
                      strlen(kGPU_Skeleton_Kernel_OpenCLString)};
    char const *strings[] = {k16fString, kGPU_Skeleton_Kernel_OpenCLString};
    cl_context context = (cl_context)device_info.contextPV;
    cl_device_id device = (cl_device_id)device_info.devicePV;

    cl_program program;
    if (!err) {
      program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0],
                                          &result);
      CL_ERR(result);
    }

    CL_ERR(clBuildProgram(program, 1, &device,
                          "-cl-single-precision-constant -cl-fast-relaxed-math",
                          0, 0));

    if (!err) {
      cl_gpu_data->main_kernel = clCreateKernel(program, "MainKernel", &result);
      CL_ERR(result);
    }

    extraP->output->gpu_data = gpu_dataH;

    out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
  }
#if HAS_METAL
  else if (extraP->input->what_gpu == PF_GPU_Framework_METAL) {
    ScopedAutoreleasePool pool;

    // Create a library from source
    NSString *source =
        [NSString stringWithCString:kGPU_Skeleton_Kernel_MetalString
                           encoding:NSUTF8StringEncoding];
    id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

    NSError *error = nil;
    id<MTLLibrary> library =
        [[device newLibraryWithSource:source options:nil error:&error]
            autorelease];

    // An error code is set for Metal compile warnings, so use nil library as
    // the error signal
    if (!err && !library) {
      err = NSError2PFErr(error);
    }

    // For debugging only. This will contain Metal compile warnings and erorrs.
    NSString *getError = error.localizedDescription;
    if (error) {
      globalData->sceneInfo.status = "Compiling Error";
      globalData->sceneInfo.errorLog = "";

      string input = getError.UTF8String;
      regex re(R"((\d+:\d+).*(error): (.*)\n(?:\s*)(.*)\n)");
      smatch m;

      while (regex_search(input, m, re)) {
        string number = m[1].str();
        string type = m[2].str();
        string message = m[3].str();
        string code = m[4].str();
        globalData->sceneInfo.errorLog +=
            type + " [" + number + "] " + message + ": " + code + "\n";
        input = m.suffix().str();
      }
      cout << globalData->sceneInfo.errorLog << endl;
    }

    PF_Handle metal_handle =
        handle_suite->host_new_handle(sizeof(MetalGPUData));
    MetalGPUData *metal_data = reinterpret_cast<MetalGPUData *>(*metal_handle);

    // Create pipeline state from function extracted from library
    if (err == PF_Err_NONE) {
      id<MTLFunction> main_function = nil;

      NSString *func_name = [NSString stringWithCString:"MainKernel"
                                               encoding:NSUTF8StringEncoding];

      main_function = [[library newFunctionWithName:func_name] autorelease];

      if (!main_function) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
      }

      if (!main_function) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
      }

      if (!err) {
        metal_data->main_kernel =
            [device newComputePipelineStateWithFunction:main_function
                                                  error:&error];
        err = NSError2PFErr(error);
      }

      if (!err) {
        extraP->output->gpu_data = metal_handle;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;

        globalData->sceneInfo.status = "Compiled Successfully";
        FX_LOG(globalData->sceneInfo.status);
      }
    }
  }
#endif

  suites.HandleSuite1()->host_unlock_handle(in_dataP->global_data);
  return err;
}

PF_Err GPUDeviceSetdown(PF_InData *in_dataP, PF_OutData *out_dataP,
                        PF_GPUDeviceSetdownExtra *extraP) {
  PF_Err err = PF_Err_NONE;

  if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {
    PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
    OpenCLGPUData *cl_gpu_dataP = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

    (void)clReleaseKernel(cl_gpu_dataP->main_kernel);

    AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
        AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite,
                                          kPFHandleSuiteVersion1, out_dataP);

    handle_suite->host_dispose_handle(gpu_dataH);
  }

  return err;
}

PF_Err SmartRenderGPU(PF_InData *in_dataP, PF_OutData *out_dataP,
                      PF_PixelFormat pixel_format, PF_EffectWorld *input_worldP,
                      PF_EffectWorld *output_worldP,
                      PF_SmartRenderExtra *extraP, PluginInputParams *infoP) {
  PF_Err err = PF_Err_NONE;

  AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite =
      AEFX_SuiteScoper<PF_GPUDeviceSuite1>(
          in_dataP, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_dataP);

  if (pixel_format != PF_PixelFormat_GPU_BGRA128) {
    err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
  }
  A_long bytes_per_pixel = 16;

  PF_GPUDeviceInfo device_info;
  ERR(gpu_suite->GetDeviceInfo(in_dataP->effect_ref,
                               extraP->input->device_index, &device_info));

  void *src_mem = 0;
  ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, input_worldP, &src_mem));

  void *dst_mem = 0;
  ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, output_worldP,
                                 &dst_mem));

#ifdef DEBUG
  AEGP_SuiteHandler suites(in_dataP->pica_basicP);
  auto *globalData = reinterpret_cast<GlobalData *>(
      suites.HandleSuite1()->host_lock_handle(in_dataP->global_data));

  globalData->debugInfo.in_world = *input_worldP;
  globalData->debugInfo.out_world = *output_worldP;
#endif

  // read the parameters
  InputKernelParams main_params;

  main_params.mWidth = input_worldP->width;
  main_params.mHeight = input_worldP->height;
  main_params.mParameter = infoP->mParameter;
  // Assign time-related variables
  float fps = in_dataP->time_scale / in_dataP->local_time_step;
  main_params.mTime = (float)in_dataP->current_time / in_dataP->time_scale;
  FX_LOG_VAL("fps", fps);
  FX_LOG_VAL("main_params.mTime", main_params.mTime);
  FX_LOG_VAL("current_time", in_dataP->current_time);
  FX_LOG_VAL("total_time", in_dataP->total_time);
  FX_LOG_VAL("time_scale", in_dataP->time_scale);
  FX_LOG_VAL("time_step", in_dataP->time_step);
  FX_LOG_VAL("local_time_step", in_dataP->local_time_step);

  A_long src_row_bytes = input_worldP->rowbytes;
  A_long dst_row_bytes = output_worldP->rowbytes;

  main_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
  main_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
  main_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

  if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {
    PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
    OpenCLGPUData *cl_gpu_dataP = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

    cl_mem cl_src_mem = (cl_mem)src_mem;
    cl_mem cl_dst_mem = (cl_mem)dst_mem;

    cl_uint main_param_index = 0;

    // Set the arguments
    CL_ERR(clSetKernelArg(cl_gpu_dataP->main_kernel, main_param_index++,
                          sizeof(cl_mem), &cl_src_mem));
    CL_ERR(clSetKernelArg(cl_gpu_dataP->main_kernel, main_param_index++,
                          sizeof(cl_mem), &cl_dst_mem));
    CL_ERR(clSetKernelArg(cl_gpu_dataP->main_kernel, main_param_index++,
                          sizeof(int), &main_params.mSrcPitch));
    CL_ERR(clSetKernelArg(cl_gpu_dataP->main_kernel, main_param_index++,
                          sizeof(int), &main_params.mDstPitch));
    CL_ERR(clSetKernelArg(cl_gpu_dataP->main_kernel, main_param_index++,
                          sizeof(int), &main_params.m16f));
    CL_ERR(clSetKernelArg(cl_gpu_dataP->main_kernel, main_param_index++,
                          sizeof(int), &main_params.mWidth));
    CL_ERR(clSetKernelArg(cl_gpu_dataP->main_kernel, main_param_index++,
                          sizeof(int), &main_params.mHeight));
    CL_ERR(clSetKernelArg(cl_gpu_dataP->main_kernel, main_param_index++,
                          sizeof(float), &main_params.mParameter));
    CL_ERR(clSetKernelArg(cl_gpu_dataP->main_kernel, main_param_index++,
                          sizeof(float), &main_params.mTime));

    // Launch the kernel
    size_t threadBlock[2] = {16, 16};
    size_t grid[2] = {RoundUp(main_params.mWidth, threadBlock[0]),
                      RoundUp(main_params.mHeight, threadBlock[1])};

    CL_ERR(clEnqueueNDRangeKernel((cl_command_queue)device_info.command_queuePV,
                                  cl_gpu_dataP->main_kernel, 2, 0, grid,
                                  threadBlock, 0, 0, 0));
  }
#if HAS_CUDA
  else if (!err && extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
    Main_CUDA((const float *)src_mem, (float *)dst_mem, main_params.mSrcPitch,
              main_params.mDstPitch, main_params.m16f, main_params.mWidth,
              main_params.mHeight, main_params.mParameter, main_params.mTime);

    if (cudaPeekAtLastError() != cudaSuccess) {
      err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
  }
#endif
#if HAS_METAL
  else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL) {
    ScopedAutoreleasePool pool;

    Handle metal_handle = (Handle)extraP->input->gpu_data;
    MetalGPUData *metal_dataP = reinterpret_cast<MetalGPUData *>(*metal_handle);

    // Set the arguments
    id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

    id<MTLBuffer> main_param_buffer =
        [[device newBufferWithBytes:&main_params
                             length:sizeof(InputKernelParams)
                            options:MTLResourceStorageModeManaged] autorelease];

    // Launch the command
    id<MTLCommandQueue> queue =
        (id<MTLCommandQueue>)device_info.command_queuePV;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder =
        [commandBuffer computeCommandEncoder];
    id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
    id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

    MTLSize threadsPerGroup1 = {
        [metal_dataP->main_kernel threadExecutionWidth], 16, 1};
    MTLSize numThreadgroups1 = {
        DivideRoundUp(output_worldP->width, threadsPerGroup1.width),
        DivideRoundUp(output_worldP->height, threadsPerGroup1.height), 1};

    if (!err) {
      [computeEncoder setComputePipelineState:metal_dataP->main_kernel];
      [computeEncoder setBuffer:src_metal_buffer offset:0 atIndex:0];
      [computeEncoder setBuffer:dst_metal_buffer offset:0 atIndex:1];
      [computeEncoder setBuffer:main_param_buffer offset:0 atIndex:2];
      [computeEncoder dispatchThreadgroups:numThreadgroups1
                     threadsPerThreadgroup:threadsPerGroup1];

      [computeEncoder endEncoding];
      [commandBuffer commit];

      err = NSError2PFErr([commandBuffer error]);
    }
  }
#endif  // HAS_METAL

#ifdef DEBUG
  suites.HandleSuite1()->host_unlock_handle(in_dataP->global_data);
#endif

  return err;
}
