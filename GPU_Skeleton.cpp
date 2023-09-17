#if HAS_CUDA
#include <cuda_runtime.h>
// GPU_Skeleton.h defines these and are needed whereas the cuda_runtime ones are not.
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include <iomanip>
#include <iostream>
#include "GPU_Skeleton.h"
#include "GPU_Skeleton_GPU.h"

static PF_Err About(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef *output) {
  PF_SPRINTF(out_data->return_msg, "%s, v%d.%d\r%s", CONFIG_NAME, MAJOR_VERSION, MINOR_VERSION, CONFIG_DESCRIPTION);

  return PF_Err_NONE;
}

static PF_Err GlobalSetdown(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef *output) {
  PF_Err err = PF_Err_NONE;

  AEGP_SuiteHandler suites(in_data->pica_basicP);

  FX_LOG("Clean data");

  if (in_data->global_data) {
    auto *                  globalData = reinterpret_cast<GlobalData *>(suites.HandleSuite1()->host_lock_handle(in_data->global_data));
    unique_ptr<CachedImage> cachedImage(reinterpret_cast<CachedImage *>(globalData->aboutImage));

    globalData->aboutImage = nullptr;

    if (cachedImage->drawbotDataH) {
      suites.HandleSuite1()->host_unlock_handle(cachedImage->drawbotDataH);
      suites.HandleSuite1()->host_dispose_handle(cachedImage->drawbotDataH);
    }

    suites.HandleSuite1()->host_unlock_handle(in_data->global_data);
    suites.HandleSuite1()->host_dispose_handle(in_data->global_data);
  }

  return err;
}

static PF_Err GlobalSetup(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef *output) {
  PF_Err            err = PF_Err_NONE;
  AEGP_SuiteHandler suites(in_data->pica_basicP);
  out_data->my_version = PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION);

#ifdef DEBUG
  FX_LOG("DEBUG VERSION");
#endif
  FX_LOG_VAL("STAGE VERSION", STAGE_VERSION);
  FX_LOG_VAL("FS_VERSION", out_data->my_version);

  out_data->out_flags = PF_OutFlag_CUSTOM_UI | //  ABOUT
      PF_OutFlag_PIX_INDEPENDENT | PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_SEND_UPDATE_PARAMS_UI
      | PF_OutFlag_I_DO_DIALOG;

  out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_CUSTOM_UI_ASYNC_MANAGER
      | PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

  // Allocate a memory block of size sizeof(GlobalData) using the host_new_handle() function from
  // the HandleSuite1 structure, which returns a memory descriptor for the created block, and store
  // it in the variable globalDataH.
  PF_Handle globalDataH = suites.HandleSuite1()->host_new_handle(sizeof(GlobalData));
  if (!globalDataH) { return PF_Err_INTERNAL_STRUCT_DAMAGED; }

  // Store the memory descriptor inside out_data for passing it to other functions.
  out_data->global_data = globalDataH;

  // Lock the memory block using host_lock_handle() to work with it and cast it to the type
  // GlobalData*.
  GlobalData *globalData = reinterpret_cast<GlobalData *>(suites.HandleSuite1()->host_lock_handle(globalDataH));
  if (!globalData) { return PF_Err_INTERNAL_STRUCT_DAMAGED; }

  // Initialize the GlobalData structure.
  AEFX_CLR_STRUCT(*globalData);

  // Create a shared_ptr of SceneInfo object and store it in the sceneInfo field of the GlobalData
  // structure.
  globalData->sceneInfo = make_shared<SceneInfo>();
  globalData->deviceInfo = make_shared<DeviceInfo>();

// Set the Debug Info GPU type based on available frameworks.
#if HAS_CUDA
  globalData->deviceInfo->GPU = PF_GPU_Framework_CUDA;
#elif HAS_METAL
  globalData->deviceInfo->GPU = PF_GPU_Framework_METAL;
#else
  globalData->deviceInfo->GPU = PF_GPU_Framework_OPENCL;
#endif

#ifdef AE_OS_MAC
#ifdef AE_PROC_INTELx64
  globalData->deviceInfo->appleCPU = APPLE_INTEL;
#else
  globalData->deviceInfo->appleCPU = APPLE_M1;
#endif
#endif

  globalData->deviceInfo->version = in_data->version;

  // Create a unique_ptr of CachedImage object and load the image file "about_image.png" into it.
  auto cachedImage = std::make_unique<CachedImage>();
  ERR(LoadAboutImage(in_data, cachedImage.get()));

  if (!err) {
    globalData->aboutImage = reinterpret_cast<PF_Handle>(cachedImage.release());
  } else {
    PF_SPRINTF(out_data->return_msg, "Cannot Load Image");
    out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
  }

  // Unlock the memory block for further usage in other functions.
  suites.HandleSuite1()->host_unlock_handle(globalDataH);

  // For Premiere - declare supported pixel formats
  if (in_data->appl_id == 'PrMr') {
    AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite = AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite,
                                                                                                     kPFPixelFormatSuiteVersion1, out_data);

    //	Add the pixel formats we support in order of preference.
    (*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);
    (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
  } else {
    out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
  }

  return err;
}

static PF_Err ParamsSetup(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef *output) {
  PF_Err            err = PF_Err_NONE;
  AEGP_SuiteHandler suites(in_data->pica_basicP);

  PF_ParamDef def;
  AEFX_CLR_STRUCT(def);

  auto        *globalData = reinterpret_cast<GlobalData *>(suites.HandleSuite1()->host_lock_handle(in_data->global_data));
  CachedImage *cachedImage = reinterpret_cast<CachedImage *>(globalData->aboutImage);
  suites.HandleSuite1()->host_unlock_handle(in_data->global_data);

  def.ui_flags = PF_PUI_CONTROL | PF_PUI_DONT_ERASE_CONTROL;
  def.ui_width = cachedImage->width;
  def.ui_height = cachedImage->height + 45.0f;

  PF_ADD_NULL("About", GPU_SKELETON_ABOUT); // "UI");

  if (!err) {
    // Referencing Examples/UI/CCU
    PF_CustomUIInfo ci;

    AEFX_CLR_STRUCT(ci);

    ci.events = PF_CustomEFlag_COMP | PF_CustomEFlag_LAYER | PF_CustomEFlag_EFFECT;
    ci.comp_ui_width = ci.comp_ui_height = 0;
    ci.layer_ui_width = ci.layer_ui_height = 0;
    ci.preview_ui_width = ci.preview_ui_height = 0;
    ci.comp_ui_alignment = PF_UIAlignment_NONE;
    ci.layer_ui_alignment = PF_UIAlignment_NONE;
    ci.preview_ui_alignment = PF_UIAlignment_NONE;

    err = (*(in_data->inter.register_ui))(in_data->effect_ref, &ci);
  }

  AEFX_CLR_STRUCT(def);
  PF_ADD_FIXED(PARAMETER_STR, 0, MAX_PARAMETER, 0, MAX_PARAMETER, 0, 1, PF_ValueDisplayFlag_NONE, 0, Parameter_DISK_ID);

  AEFX_CLR_STRUCT(def);
  PF_ADD_COLOR(COLOR_STR, 0.0, 0.0, 0.0, Color_DISK_ID);

  AEFX_CLR_STRUCT(def);
  PF_ADD_LAYER(LAYER_STR, 0, Layer_DISK_ID);

  out_data->num_params = GPU_SKELETON_NUM_PARAMS;

  // Premiere Pro/Elements does not support this suite
  if (in_data->appl_id != 'PrMr') {
    AEFX_SuiteScoper<PF_EffectUISuite1> effect_ui_suiteP = AEFX_SuiteScoper<PF_EffectUISuite1>(in_data, kPFEffectUISuite,
                                                                                               kPFEffectUISuiteVersion1, out_data);

    ERR(effect_ui_suiteP->PF_SetOptionsButtonName(in_data->effect_ref, "URL..."));
  }

  return err;
}

static void DisposePreRenderData(void *pre_render_dataPV) {
  if (pre_render_dataPV) {
    PluginInputParams *infoP = reinterpret_cast<PluginInputParams *>(pre_render_dataPV);
    free(infoP);
  }
}

static PF_Err PreRender(PF_InData *in_dataP, PF_OutData *out_dataP, PF_PreRenderExtra *extraP) {
  PF_Err err = PF_Err_NONE;

  //    AEGP_SuiteHandler suites(in_dataP->pica_basicP);
  PF_CheckoutResult in_result;
  PF_RenderRequest  req = extraP->input->output_request;
  extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

  PluginInputParams *infoP = reinterpret_cast<PluginInputParams *>(malloc(sizeof(PluginInputParams)));

  if (infoP) {
    // Querying parameters to demoonstrate they are available at PreRender, and data can be passed
    // from PreRender to Render with pre_render_data.
    PF_ParamDef cur_param;

    ERR(PF_CHECKOUT_PARAM(in_dataP, Parameter_DISK_ID, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
    infoP->mParameter = cur_param.u.fd.value / 65536.0f;
    infoP->mParameter /= 100.0f;
    ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

    ERR(PF_CHECKOUT_PARAM(in_dataP, Color_DISK_ID, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
    infoP->mInnerStruct.color[0] = cur_param.u.cd.value.red;
    infoP->mInnerStruct.color[1] = cur_param.u.cd.value.green;
    infoP->mInnerStruct.color[2] = cur_param.u.cd.value.blue;
    infoP->mInnerStruct.color[3] = cur_param.u.cd.value.alpha;
    ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

    extraP->output->pre_render_data = infoP;
    extraP->output->delete_pre_render_data_func = DisposePreRenderData;

    // END OF PARAMETERS

    AEGP_SuiteHandler suites(in_dataP->pica_basicP);

    ERR(extraP->cb->checkout_layer(in_dataP->effect_ref, GPU_SKELETON_LAYER, Layer_DISK_ID, &req, in_dataP->current_time,
                                   in_dataP->time_step, in_dataP->time_scale, &in_result));

    ERR(extraP->cb->checkout_layer(in_dataP->effect_ref, GPU_SKELETON_INPUT, GPU_SKELETON_INPUT, &req, in_dataP->current_time,
                                   in_dataP->time_step, in_dataP->time_scale, &in_result));

    if (!err) {
      UnionLRect(&in_result.result_rect, &extraP->output->result_rect);
      UnionLRect(&in_result.max_result_rect, &extraP->output->max_result_rect);
    }

    suites.HandleSuite1()->host_unlock_handle(in_dataP->global_data);
  } else {
    err = PF_Err_OUT_OF_MEMORY;
  }

  return err;
}

static PF_Err SmartRenderCPU(PF_InData *in_data, PF_OutData *out_data, PF_PixelFormat pixel_format, PF_EffectWorld *input_worldP,
                             PF_EffectWorld *output_worldP, PF_SmartRenderExtra *extraP, PluginInputParams *infoP) {
  PF_Err err = PF_Err_NONE;

  PF_SPRINTF(out_data->return_msg,
             "%s\n\n"
             "Sorry, the current version of the plugin only supports GPU acceleration. We "
             "apologize for the inconvenience.",
             getVersionString().c_str());
  out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;

  return err;
}

static PF_Err SmartRender(PF_InData *in_data, PF_OutData *out_data, PF_SmartRenderExtra *extraP, bool isGPU) {
  PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;

  PF_EffectWorld  *input_worldP   = NULL,
                  *output_worldP  = NULL,
                  *layer_worldP   = NULL;

  // Parameters can be queried during render. In this example, we pass them from PreRender as an
  // example of using pre_render_data.
  PluginInputParams *infoP = reinterpret_cast<PluginInputParams *>(extraP->input->pre_render_data);

  if (infoP) {
    ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, GPU_SKELETON_INPUT, &input_worldP)));
    ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, GPU_SKELETON_LAYER, &layer_worldP)));

    ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

    AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
                                                                                    kPFWorldSuite,
                                                                                    kPFWorldSuiteVersion2,
                                                                                    out_data);

    PF_PixelFormat                   pixel_format = PF_PixelFormat_INVALID;
    ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

    if (isGPU) {
      ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, layer_worldP, extraP, infoP));
    } else {
      ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
    }

    ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, GPU_SKELETON_LAYER));
    ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, GPU_SKELETON_INPUT));
  } else {
    return PF_Err_INTERNAL_STRUCT_DAMAGED;
  }
  return err;
}

static PF_Err PopDialog(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef *output) {
  PF_Err err = PF_Err_NONE;

  out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;

  const char *url = "github.com/timurco";

#ifdef WIN32
  // Code for Windows
  ShellExecute(NULL, "open", url, NULL, NULL, SW_SHOWNORMAL);
#elif __APPLE__
  // Code for macOS
  char open_url_command[256];
  snprintf(open_url_command, sizeof(open_url_command), "open %s", url);
  system(open_url_command);
#endif

  return err;
}

static PF_Err HandleEvent(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef *output, PF_EventExtra *extra) {
  PF_Err err = PF_Err_NONE;

  switch (extra->e_type) {
    case PF_Event_DRAW:
      if ((*extra->contextH)->w_type == PF_Window_EFFECT) {
        ERR(DrawEvent(in_data, out_data, params, output, extra, params[1]->u.cd.value));
      } else {
        ERR(DrawCompUIEvent(in_data, out_data, params, output, extra));
      }

      break;

    default:
      break;
  }

  return err;
}

extern "C" DllExport PF_Err PluginDataEntryFunction(PF_PluginDataPtr inPtr, PF_PluginDataCB inPluginDataCallBackPtr,
                                                    SPBasicSuite *inSPBasicSuitePtr, const char *inHostName, const char *inHostVersion) {
  PF_Err result = PF_Err_INVALID_CALLBACK;

  result = PF_REGISTER_EFFECT(inPtr, inPluginDataCallBackPtr,
                              CONFIG_NAME,       // Name
                              CONFIG_MATCH_NAME, // Match Name
                              CONFIG_CATEGORY,   // Category
                              AE_RESERVED_INFO); // Reserved Info

  return result;
}

PF_Err EffectMain(PF_Cmd cmd, PF_InData *in_dataP, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef *output, void *extra) {
  PF_Err err = PF_Err_NONE;

  try {
    switch (cmd) {
      case PF_Cmd_ABOUT:
        err = About(in_dataP, out_data, params, output);
        break;
      case PF_Cmd_GLOBAL_SETUP:
        err = GlobalSetup(in_dataP, out_data, params, output);
        break;

      case PF_Cmd_GLOBAL_SETDOWN:
        err = GlobalSetdown(in_dataP, out_data, params, output);
        break;

      case PF_Cmd_PARAMS_SETUP:
        err = ParamsSetup(in_dataP, out_data, params, output);
        break;
      case PF_Cmd_GPU_DEVICE_SETUP:
        err = GPUDeviceSetup(in_dataP, out_data, (PF_GPUDeviceSetupExtra *)extra);
        break;
      case PF_Cmd_GPU_DEVICE_SETDOWN:
        err = GPUDeviceSetdown(in_dataP, out_data, (PF_GPUDeviceSetdownExtra *)extra);
        break;

      case PF_Cmd_EVENT:
        err = HandleEvent(in_dataP, out_data, params, output, reinterpret_cast<PF_EventExtra *>(extra));
        break;

      case PF_Cmd_SMART_PRE_RENDER:
        err = PreRender(in_dataP, out_data, (PF_PreRenderExtra *)extra);
        break;
      case PF_Cmd_SMART_RENDER:
        err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra *)extra, false);
        break;
      case PF_Cmd_SMART_RENDER_GPU:
        err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra *)extra, true);
        break;
      case PF_Cmd_DO_DIALOG:
        err = PopDialog(in_dataP, out_data, params, output);
        break;
      default:
        break;
    }
  } catch (PF_Err &thrown_err) {
    // Never EVER throw exceptions into AE.
    err = thrown_err;
  }
  return err;
}
