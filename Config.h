#pragma once

#define DEBUG 1

#ifndef IS_PIPL
// Include PF_Stage
#include "AE_Effect.h"

#else
// For PiPL.r, define manually
#define PF_Stage_DEVELOP 0
#define PF_Stage_ALPHA 1
#define PF_Stage_BETA 2
#define PF_Stage_RELEASE 3

#endif

/* Versioning information */

#define MAJOR_VERSION 1
#define MINOR_VERSION 0

#ifdef DEBUG
#define STAGE_VERSION PF_Stage_DEVELOP
#else
#define STAGE_VERSION PF_Stage_RELEASE
#endif

#define BUILD_VERSION 0
#define BUG_VERSION 0
/* PiPL.r Version Define */
#define AEFX_VERSION (  (MAJOR_VERSION << 19) | \
                        ((MINOR_VERSION & 0xF) << 15) | \
                        ((BUG_VERSION & 0xF) << 11) | \
                        (STAGE_VERSION << 9) | \
                        (BUILD_VERSION & 0x1FF))

/**
 * Change Log
 *
 * Version 1.0.0
 * - Release
 */

#define CONFIG_NAME "GPU_Skeleton"
#define CONFIG_MATCH_NAME "ADBE TMR_GPU_Skeleton"
#define CONFIG_CATEGORY "TimurKo"
#define CONFIG_DESCRIPTION "(c) 2023 Tim Constantinov.\rhttps://github.com/timurco"
  
