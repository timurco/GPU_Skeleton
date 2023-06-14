/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007 Adobe Systems Incorporated                       */
/* All Rights Reserved.                                            */
/*                                                                 */
/* NOTICE:  All information contained herein is, and remains the   */
/* property of Adobe Systems Incorporated and its suppliers, if    */
/* any.  The intellectual and technical concepts contained         */
/* herein are proprietary to Adobe Systems Incorporated and its    */
/* suppliers and may be covered by U.S. and Foreign Patents,       */
/* patents in process, and are protected by trade secret or        */
/* copyright law.  Dissemination of this information or            */
/* reproduction of this material is strictly forbidden unless      */
/* prior written permission is obtained from Adobe Systems         */
/* Incorporated.                                                   */
/*                                                                 */
/*******************************************************************/

#pragma once
#ifndef GPU_Skeleton_H
#define GPU_Skeleton_H

#include <math.h>

#include "AEConfig.h"
#include "AEFX_SuiteHelper.h"
#include "AEGP_SuiteHandler.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_Macros.h"
#include "GPU_Skeleton_Kernel.cl.h"
#include "Param_Utils.h"
#include "PrSDKAESupport.h"
#include "Smart_Utils.h"
#include "String_Utils.h"
#include "entry.h"

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#include <chrono>
#include <regex>

#include "AEUtil.h"
#include "Config.h"
#include "Debug.h"
#include "MiscUtil.h"

enum {
  GPU_SKELETON_INPUT = 0,
  GPU_SKELETON_ABOUT,
  GPU_SKELETON_PARAMETER,
  GPU_SKELETON_NUM_PARAMS
};

enum {
  Parameter_DISK_ID = 1,
};

typedef struct {
  float mParameter;
} PluginInputParams;

#define PARAMETER_STR "Parameter"
#define MAX_PARAMETER 256

extern "C" {

DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData *in_data, PF_OutData *out_data,
                            PF_ParamDef *params[], PF_LayerDef *output, void *extra);
}

typedef struct {
  PF_Handle drawbotDataH;
  unsigned char *data;
  int channels;
  int width;
  int height;
} CachedImage;

struct SceneInfo {
  SceneInfo() : status("Not loaded") {}
  string status;
  string errorLog;
};

enum { APPLE_NONE = 0, APPLE_M1, APPLE_INTEL };
typedef A_long AppleCPU;

struct DeviceInfo {
  AppleCPU appleCPU;
  PF_GPU_Framework GPU;
  PF_SpecVersion version;
};

//      FX_LOG_VAL("outputRect", outputRect);
//      FX_LOG_VAL("reqrect", req.rect);
//      FX_LOG_VAL("RESULT", in_result.result_rect);
//      FX_LOG_VAL("MAX_RESULT", in_result.max_result_rect);

struct DebugInfo {
  PF_LRect result_rect;
  PF_LRect output_rect;
  PF_LRect request_rect;
  PF_LRect max_result_rect;
  PF_LayerDef out_world;
  PF_LayerDef tmp_world;
  PF_LayerDef in_world;
};

struct GlobalData {
  GlobalData() : aboutImage(nullptr) {}
  PF_Handle aboutImage;
  SceneInfo sceneInfo;
  DeviceInfo deviceInfo;
  DebugInfo debugInfo;
};

PF_Err DrawEvent(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[],
                 PF_LayerDef *output, PF_EventExtra *event_extra, PF_Pixel some_color);

PF_Err LoadImageFile(PF_InData *in_data, const A_char *file_name, CachedImage *cachedImage);

PF_Err DrawCompUIEvent(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[],
                       PF_LayerDef *output, PF_EventExtra *extra);

#endif  // GPU_Skeleton_H