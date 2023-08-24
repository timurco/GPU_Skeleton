//
//  GPU_Skeleton_UI_Handler.cpp
//  GPU_Skeleton
//
//  Created by Tim Constantinov on 21.04.2023.
//

#include <codecvt>
#include <fstream>
#include <iostream>
#include <vector>

#include "GPU_Skeleton.h"
#include "MiscUtil.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef AE_OS_WIN
#include "Win/resource.h"
#define ABOUT_IMAGE MAKEINTRESOURCE(IDB_PNG1)

HMODULE GCM() {
  HMODULE hModule = NULL;
  GetModuleHandleEx(
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
    (LPCTSTR)GCM,
    &hModule);

  return hModule;
}
#else
#define ABOUT_IMAGE "about_image.png"
#endif

PF_Err LoadAboutImage(PF_InData* in_data, CachedImage* cachedImage) {
  AEGP_SuiteHandler suites(in_data->pica_basicP);
  unsigned char* imageData;

#ifdef AE_OS_WIN
  HRSRC hResource = FindResource(GCM(), ABOUT_IMAGE, MAKEINTRESOURCE(PNG));
  if (hResource == nullptr) {
    FX_LOG("Cannot find 'about' resource.");
    return PF_Err_BAD_CALLBACK_PARAM;
  }

  const DWORD imageSize = SizeofResource(GCM(), hResource);
  const void* pResourceData = LockResource(LoadResource(GCM(), hResource));
  if (pResourceData == nullptr) {
    FX_LOG("Cannot lock 'about' resource.");
    return PF_Err_BAD_CALLBACK_PARAM;
  }

  // Use stb_image.h to decode the image data
  imageData = stbi_load_from_memory((const stbi_uc*)pResourceData, imageSize,
    &cachedImage->width, &cachedImage->height,
    &cachedImage->channels, 4);

  // If the image data loading fails, handle the error.
  if (imageData == nullptr) {
    FX_LOG("Cannot load 'about' image.");
    return PF_Err_BAD_CALLBACK_PARAM;
  }

  // Convert the image from RGBA to ARGB
  // since Windows support only 4 channels ARGB (32bits)
  // info: https://community.adobe.com/t5/after-effects-discussions/image-logo-in-parameter/m-p/5197581
  cachedImage->pixelLayout = kDRAWBOT_PixelLayout_32BGR;
  unsigned char* argb_data = (unsigned char*)malloc(cachedImage->width * cachedImage->height * 4); // Allocate memory for ARGB data

  for (int y = 0; y < cachedImage->height; ++y) {
    for (int x = 0; x < cachedImage->width; ++x) {
      int rgba_index = (y * cachedImage->width + x) * 4;

      // Conversion from RGBA to BGRA
      argb_data[rgba_index + 0] = imageData[rgba_index + 2]; // B
      argb_data[rgba_index + 1] = imageData[rgba_index + 1]; // G
      argb_data[rgba_index + 2] = imageData[rgba_index + 0]; // R
      argb_data[rgba_index + 3] = imageData[rgba_index + 3]; // A
    }
  }

  // Free the original RGBA data loaded from the image
  stbi_image_free(imageData);

  // Update the data pointer to point to the new ARGB data
  imageData = argb_data;
#else
  // Get the resource path
  std::string resourcePath = AEUtil::getResourcesPath(in_data);
  FX_LOG("Resources: " << resourcePath);

  resourcePath += ABOUT_IMAGE;

  // Load the image using stb_image.h
  imageData = stbi_load(resourcePath.c_str(), &cachedImage->width, &cachedImage->height,
    &cachedImage->channels, 3);

  if (imageData == nullptr) {
    FX_LOG("Cannot load 'about' image at: " << resourcePath);
    return PF_Err_BAD_CALLBACK_PARAM;
  }
  
  cachedImage->channels = 3;
  cachedImage->pixelLayout = kDRAWBOT_PixelLayout_24BGR;
#endif

  // The following section remains unchanged from your original code
  const int numBytes = cachedImage->width * cachedImage->height * cachedImage->channels;
  cachedImage->drawbotDataH = suites.HandleSuite1()->host_new_handle(numBytes);
  unsigned char* drawbotDataP = static_cast<unsigned char*>(suites.HandleSuite1()->host_lock_handle(cachedImage->drawbotDataH));
  
  memcpy(drawbotDataP, imageData, numBytes);
  
  // Unlock and release memory
  suites.HandleSuite1()->host_unlock_handle(cachedImage->drawbotDataH);

#ifdef AE_OS_WIN
  free(imageData); // For Windows, after ARGB conversion we use malloc, so we free it.
#else
  stbi_image_free(imageData); // For non-Windows (Mac, in this case) where the image was loaded with stb_image
#endif

  return PF_Err_NONE;
}

static std::unique_ptr<DRAWBOT_UTF16Char[]> convertStringToUTF16Char(const std::string &str) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  std::wstring wstr = converter.from_bytes(str);
  size_t length = wcslen(wstr.c_str());
  std::unique_ptr<DRAWBOT_UTF16Char[]> utf16char(
      new DRAWBOT_UTF16Char[length + 1]);  // Add +1 for the null character
  AEUtil::copyConvertStringLiteralIntoUTF16(wstr.c_str(), utf16char.get());
  utf16char[length] = 0;  // Add null character at the end of the string
  return utf16char;
}

PF_Err DrawEvent(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[],
                 PF_LayerDef *output, PF_EventExtra *event_extra, PF_Pixel some_color) {
  PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
  AEGP_SuiteHandler suites(in_data->pica_basicP);

  DRAWBOT_Suites drawbotSuites;
  DRAWBOT_DrawRef drawing_ref = NULL;
  DRAWBOT_SurfaceRef surface_ref = NULL;
  DRAWBOT_SupplierRef supplier_ref = NULL;
  DRAWBOT_ImageRef img_ref = NULL;
  DRAWBOT_BrushRef brush_ref = NULL;
  DRAWBOT_BrushRef string_brush_ref = NULL;
  DRAWBOT_FontRef font_ref = NULL;
  DRAWBOT_FontRef small_font_ref = NULL;

  DRAWBOT_ColorRGBA drawbot_color;
  float fontSize = 0.0f;
  float smallFontSize = 10.0f;

  // Acquire all the drawbot suites in one go; it should be matched with the release routine.
  // You can also use C++ style AEFX_DrawbotSuitesScoper which doesn't need a release routine.
  ERR(AEFX_AcquireDrawbotSuites(in_data, out_data, &drawbotSuites));

  if (!err) {
    PF_EffectCustomUISuite1 *effectCustomUISuiteP;
    ERR(AEFX_AcquireSuite(in_data, out_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1,
                          NULL, (void **)&effectCustomUISuiteP));

    if (!err) {
      ERR(effectCustomUISuiteP->PF_GetDrawingReference(event_extra->contextH, &drawing_ref));
      AEFX_ReleaseSuite(in_data, out_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1,
                        NULL);
    }
  }

  if (!drawing_ref) {
    return err;
  }

  if (!err) {
    ERR(drawbotSuites.drawbot_suiteP->GetSupplier(drawing_ref, &supplier_ref));
    ERR(drawbotSuites.drawbot_suiteP->GetSurface(drawing_ref, &surface_ref));
  }

  auto *globalData = reinterpret_cast<GlobalData *>(suites.HandleSuite1()->host_lock_handle(in_data->global_data));
  CachedImage *cachedImage = reinterpret_cast<CachedImage *>(globalData->aboutImage);
  unsigned char* imageDataP = reinterpret_cast<unsigned char*>(suites.HandleSuite1()->host_lock_handle(cachedImage->drawbotDataH));

  ERR(drawbotSuites.supplier_suiteP->NewImageFromBuffer(
      supplier_ref, cachedImage->width, cachedImage->height,
      cachedImage->channels * cachedImage->width,
      cachedImage->pixelLayout,
      imageDataP, &img_ref));

  suites.HandleSuite1()->host_unlock_handle(cachedImage->drawbotDataH);
  suites.HandleSuite1()->host_unlock_handle(in_data->global_data);

  DRAWBOT_PointF32 in_originP;

  in_originP.x = (float)event_extra->effect_win.current_frame.left;
  in_originP.y = (float)event_extra->effect_win.current_frame.top;

  ERR(drawbotSuites.surface_suiteP->DrawImage(surface_ref, img_ref, &in_originP, 1.0f));

  /*
   WORKING WITH FONTS
   */

  // Get the default font size.
  ERR(drawbotSuites.supplier_suiteP->GetDefaultFontSize(supplier_ref, &fontSize));
  // Create the default font with the default size. Note that you can provide a different font size.
  ERR(drawbotSuites.supplier_suiteP->NewDefaultFont(supplier_ref, fontSize, &font_ref));
  ERR(drawbotSuites.supplier_suiteP->NewDefaultFont(supplier_ref, smallFontSize, &small_font_ref));

  auto status = convertStringToUTF16Char(getVersionString());

  // Draw string with white color
  drawbot_color.red = drawbot_color.green = drawbot_color.blue = 1.0;
  drawbot_color.alpha = 0.8f;

  ERR(drawbotSuites.supplier_suiteP->NewBrush(supplier_ref, &drawbot_color, &string_brush_ref));

  DRAWBOT_PointF32 text_origin;

  text_origin.x = event_extra->effect_win.current_frame.left + cachedImage->width / 2;
  text_origin.y = event_extra->effect_win.current_frame.top + cachedImage->height - 9;

  ERR(drawbotSuites.surface_suiteP->DrawString(
      surface_ref, string_brush_ref, font_ref, status.get(), &text_origin,
      kDRAWBOT_TextAlignment_Center, kDRAWBOT_TextTruncation_None, 0.0f));
  // Draw string with white color

  drawbot_color.red = drawbot_color.green = drawbot_color.blue = 0.5;
  ERR(drawbotSuites.supplier_suiteP->NewBrush(supplier_ref, &drawbot_color, &string_brush_ref));

  text_origin.x = event_extra->effect_win.current_frame.left;
  text_origin.y =
      event_extra->effect_win.current_frame.top + cachedImage->height + smallFontSize + 2;

  ERR(drawbotSuites.surface_suiteP->DrawString(
      surface_ref, string_brush_ref, small_font_ref,
      convertStringToUTF16Char("CPU: " + GET_APPLE_CPU(globalData->deviceInfo->appleCPU)).get(),
      &text_origin, kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.0f));

  text_origin.y += smallFontSize + 2;

  ERR(drawbotSuites.surface_suiteP->DrawString(
      surface_ref, string_brush_ref, small_font_ref,
      convertStringToUTF16Char("Graphics Card: " + GET_GPU(globalData->deviceInfo->GPU)).get(),
      &text_origin, kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.0f));

  text_origin.y += smallFontSize + 2;

  ERR(drawbotSuites.surface_suiteP->DrawString(
      surface_ref, string_brush_ref, small_font_ref,
      convertStringToUTF16Char("After Effects Version: " +
                               GET_AE_VERSION(globalData->deviceInfo->version))
          .get(),
      &text_origin, kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.0f));

  if (string_brush_ref) {
    ERR2(drawbotSuites.supplier_suiteP->ReleaseObject(
        reinterpret_cast<DRAWBOT_ObjectRef>(string_brush_ref)));
  }

  if (font_ref) {
    ERR2(drawbotSuites.supplier_suiteP->ReleaseObject(
        reinterpret_cast<DRAWBOT_ObjectRef>(font_ref)));
  }

  // Release/destroy the brush. Otherwise, it will lead to a memory leak.
  if (brush_ref) {
    ERR2(drawbotSuites.supplier_suiteP->ReleaseObject(
        reinterpret_cast<DRAWBOT_ObjectRef>(brush_ref)));
  }

  if (img_ref) {
    ERR2(drawbotSuites.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(img_ref)));
  }

  // Release the earlier acquired drawbot suites
  ERR2(AEFX_ReleaseDrawbotSuites(in_data, out_data));

  event_extra->evt_out_flags = PF_EO_HANDLED_EVENT;

  return err;
}

static DRAWBOT_MatrixF32 getLayer2FrameXform(PF_InData *in_data, PF_EventExtra *extra) {
  PF_FixedPoint pts[3];

  pts[0].x = FLOAT2FIX(0.0);
  pts[0].y = FLOAT2FIX(0.0);

  pts[1].x = FLOAT2FIX(1.0);
  pts[1].y = FLOAT2FIX(0.0);

  pts[2].x = FLOAT2FIX(0.0);
  pts[2].y = FLOAT2FIX(1.0);

  if ((*extra->contextH)->w_type == PF_Window_COMP) {
    for (A_short i = 0; i < 3; i++) {
      extra->cbs.layer_to_comp(extra->cbs.refcon, extra->contextH, in_data->current_time,
                               in_data->time_scale, &pts[i]);
    }
  }

  for (A_short i = 0; i < 3; i++) {
    extra->cbs.source_to_frame(extra->cbs.refcon, extra->contextH, &pts[i]);
  }

  DRAWBOT_MatrixF32 xform;

  AEFX_CLR_STRUCT(xform);

  xform.mat[0][0] = FIX_2_FLOAT(pts[1].x) - FIX_2_FLOAT(pts[0].x);
  xform.mat[0][1] = FIX_2_FLOAT(pts[1].y) - FIX_2_FLOAT(pts[0].y);

  xform.mat[1][0] = FIX_2_FLOAT(pts[2].x) - FIX_2_FLOAT(pts[0].x);
  xform.mat[1][1] = FIX_2_FLOAT(pts[2].y) - FIX_2_FLOAT(pts[0].y);

  xform.mat[2][0] = FIX_2_FLOAT(pts[0].x);
  xform.mat[2][1] = FIX_2_FLOAT(pts[0].y);

  return xform;
}

PF_Err DrawCompUIEvent(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[],
                       PF_LayerDef *output, PF_EventExtra *extra) {
  PF_Err err = PF_Err_NONE;

  AEGP_SuiteHandler suites(in_data->pica_basicP);

  DRAWBOT_Suites drawbotSuites;
  DRAWBOT_DrawRef drawingRef = NULL;
  DRAWBOT_SurfaceRef surfaceRef = NULL;

  AEFX_AcquireDrawbotSuites(in_data, out_data, &drawbotSuites);

  // Get the drawing reference by passing context to this new api
  PF_EffectCustomUISuite2 *effectCustomUISuiteP;
  err = AEFX_AcquireSuite(in_data, out_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion2,
                          NULL, (void **)&effectCustomUISuiteP);

  if (!err && effectCustomUISuiteP) {
    err = (*effectCustomUISuiteP->PF_GetDrawingReference)(extra->contextH, &drawingRef);
    AEFX_ReleaseSuite(in_data, out_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion2,
                      NULL);
  }

  if (!drawingRef) {
    return err;
  }

  ERR(suites.DrawbotSuiteCurrent()->GetSurface(drawingRef, &surfaceRef));

  auto *globalData =
      reinterpret_cast<GlobalData *>(suites.HandleSuite1()->host_lock_handle(in_data->global_data));

  bool doRenderErrorLog = !globalData->sceneInfo->errorLog.empty();

  if (!err) {
    DRAWBOT_ColorRGBA foregroundColor, shadowColor, redColor, yellowColor;
    A_LPoint shadowOffset;
    float strokeWidth, vertexSize;

    if (in_data->appl_id != 'PrMr') {
      // Currently, EffectCustomUIOverlayThemeSuite is unsupported in Premiere Pro/Elements
      ERR(suites.EffectCustomUIOverlayThemeSuite1()->PF_GetPreferredForegroundColor(
          &foregroundColor));
      ERR(suites.EffectCustomUIOverlayThemeSuite1()->PF_GetPreferredShadowColor(&shadowColor));
      ERR(suites.EffectCustomUIOverlayThemeSuite1()->PF_GetPreferredShadowOffset(&shadowOffset));
      ERR(suites.EffectCustomUIOverlayThemeSuite1()->PF_GetPreferredStrokeWidth(&strokeWidth));
      ERR(suites.EffectCustomUIOverlayThemeSuite1()->PF_GetPreferredVertexSize(&vertexSize));
    } else {
      foregroundColor = {0.9f, 0.9f, 0.9f, 1.0f};
      shadowColor = {0.0, 0, 0.0f, 0.2f};
      redColor = {0.9, 0, 0, 1};
      yellowColor = {0.9, 0.6, 0, 1};
      shadowOffset.x = -1;
      shadowOffset.y = +1;
    }

    redColor = foregroundColor;
    redColor.green = 0;
    redColor.blue = 0;

    yellowColor = foregroundColor;
    yellowColor.green = 0.3;
    yellowColor.blue = 0;

    float fontSize = 10.0;
    float largeFontScale = 1.5f;
    float padding = 3.0f;

    auto layer2FrameXform = getLayer2FrameXform(in_data, extra);

    // Setup Drawbot objects
    DRAWBOT_PointF32 origin = {padding, padding};
    DRAWBOT_SupplierRef supplierRef = NULL;
    DRAWBOT_BrushRef redBrushRef = NULL, shadowBrushRef = NULL, yellowBrushRef = NULL;
    DRAWBOT_FontRef fontRef = NULL, largeFontRef = NULL;
    DRAWBOT_Rect32 clipRect;

    // Always set the scale to one regardless viewport zoom
    double zoom =
        suites.ANSICallbacksSuite1()->hypot(layer2FrameXform.mat[0][0], layer2FrameXform.mat[0][1]);
    layer2FrameXform.mat[0][0] /= zoom;
    layer2FrameXform.mat[0][1] /= zoom;
    layer2FrameXform.mat[1][0] /= zoom;
    layer2FrameXform.mat[1][1] /= zoom;

    clipRect.top = 0;
    clipRect.left = 0;
    clipRect.width = in_data->width * zoom;
    clipRect.height = in_data->height * zoom;

    ERR(drawbotSuites.drawbot_suiteP->GetSupplier(drawingRef, &supplierRef));
    ERR(drawbotSuites.supplier_suiteP->NewBrush(supplierRef, &redColor, &redBrushRef));
    ERR(drawbotSuites.supplier_suiteP->NewBrush(supplierRef, &yellowColor, &yellowBrushRef));
    ERR(drawbotSuites.supplier_suiteP->NewBrush(supplierRef, &shadowColor, &shadowBrushRef));
    ERR(drawbotSuites.supplier_suiteP->GetDefaultFontSize(supplierRef, &fontSize));
    ERR(drawbotSuites.supplier_suiteP->NewDefaultFont(supplierRef, fontSize, &fontRef));
    ERR(drawbotSuites.supplier_suiteP->NewDefaultFont(supplierRef, fontSize * largeFontScale,
                                                      &largeFontRef));

    {  // Start Transform
      auto *surface = suites.SurfaceSuiteCurrent();

      // Apply a transform
      ERR(drawbotSuites.surface_suiteP->PushStateStack(surfaceRef));
      ERR(drawbotSuites.surface_suiteP->Transform(surfaceRef, &layer2FrameXform));

      // Apply a clip
      ERR(drawbotSuites.surface_suiteP->PushStateStack(surfaceRef));
      surface->Clip(surfaceRef, supplierRef, &clipRect);

#ifdef DEBUG
      origin.y += fontSize * largeFontScale;

      ERR(surface->DrawString(surfaceRef, redBrushRef, largeFontRef,
                              convertStringToUTF16Char("DEBUG VERSION:").get(), &origin,
                              kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.0f));

      // Array of values to be displayed
      auto debugValues = {
          make_pair("Max Result Rect: ", to_string(globalData->debugInfo.max_result_rect)),
          make_pair("Result Rect: ", to_string(globalData->debugInfo.result_rect)),
          make_pair("Request Rect: ", to_string(globalData->debugInfo.request_rect)),
          make_pair("Output Rect: ", to_string(globalData->debugInfo.output_rect)),
          make_pair("Input World: ", to_string(globalData->debugInfo.in_world)),
          make_pair("Temp World: ", to_string(globalData->debugInfo.tmp_world)),
          make_pair("Output World: ", to_string(globalData->debugInfo.out_world)),
      };

      for (const auto &debugValue : debugValues) {
        auto debugString = convertStringToUTF16Char(debugValue.first + debugValue.second);

        origin.y += fontSize + padding;
        origin.x += shadowOffset.x;
        origin.y += shadowOffset.y;

        ERR(surface->DrawString(surfaceRef, shadowBrushRef, fontRef, debugString.get(), &origin,
                                kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.0f));

        origin.x -= shadowOffset.x;
        origin.y -= shadowOffset.y;

        ERR(surface->DrawString(surfaceRef, yellowBrushRef, fontRef, debugString.get(), &origin,
                                kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.0f));
      }
#endif

      if (doRenderErrorLog) {
        // Print the type of error with larger font
        origin.y += fontSize * largeFontScale;

        origin.x += shadowOffset.x;
        origin.y += shadowOffset.y;

        auto status = convertStringToUTF16Char(globalData->sceneInfo->status);
        ERR(surface->DrawString(surfaceRef, shadowBrushRef, largeFontRef, status.get(), &origin,
                                kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.0f));

        origin.x -= shadowOffset.x;
        origin.y -= shadowOffset.y;

        ERR(surface->DrawString(surfaceRef, redBrushRef, largeFontRef, status.get(), &origin,
                                kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.0f));

        origin.y += padding * 2;

        // Print error log for each line
        for (auto &line : splitWith(globalData->sceneInfo->errorLog, "\n")) {
          origin.y += fontSize + padding;

          auto lineUTF16Char = convertStringToUTF16Char(line);

          origin.x += shadowOffset.x;
          origin.y += shadowOffset.y;

          ERR(surface->DrawString(surfaceRef, shadowBrushRef, fontRef, lineUTF16Char.get(), &origin,
                                  kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.0f));

          origin.x -= shadowOffset.x;
          origin.y -= shadowOffset.y;

          ERR(surface->DrawString(surfaceRef, redBrushRef, fontRef, lineUTF16Char.get(), &origin,
                                  kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.0f));
        }
      } /* End doRenderErrorLog */

      // Pop clipping & transofrm stacks
      ERR(drawbotSuites.surface_suiteP->PopStateStack(surfaceRef));
      ERR(drawbotSuites.surface_suiteP->PopStateStack(surfaceRef));
    } /* End Transform */

    // Release drawbot objects
    ERR(drawbotSuites.supplier_suiteP->ReleaseObject((DRAWBOT_ObjectRef)redBrushRef));
    ERR(drawbotSuites.supplier_suiteP->ReleaseObject((DRAWBOT_ObjectRef)shadowBrushRef));
    ERR(drawbotSuites.supplier_suiteP->ReleaseObject((DRAWBOT_ObjectRef)fontRef));
    ERR(drawbotSuites.supplier_suiteP->ReleaseObject((DRAWBOT_ObjectRef)largeFontRef));

    AEFX_ReleaseDrawbotSuites(in_data, out_data);

    extra->evt_out_flags = PF_EO_HANDLED_EVENT;

  } /* End doRenderSomething */

  suites.HandleSuite1()->host_unlock_handle(in_data->global_data);

  return err;
}
