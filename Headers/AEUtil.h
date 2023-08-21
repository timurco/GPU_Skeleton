/*
 Modified from: https://github.com/baku89/ISF4AE
*/

#pragma once

#include <string>

#include "AEFX_SuiteHelper.h"
#include "AEGP_SuiteHandler.h"
#include "AE_Effect.h"
#include "AE_Macros.h"
#include "AEConfig.h"

using namespace std;

#ifdef AE_OS_WIN
//  global compilation flag configuring windows sdk headers
//  preventing inclusion of min and max macros clashing with <limits>
#define NOMINMAX 1
//  override byte to prevent clashes with <cstddef>
#define byte win_byte_override
#include <Windows.h>
//  Undefine min max macros so they won't collide with <limits> header content.
#undef min
#undef max
//  Undefine byte macros so it won't collide with <cstddef> header content.
#undef byte
#endif

namespace AEUtil {

string getResourcesPath(PF_InData *in_data);

// Other AE-specific utils
void copyConvertStringLiteralIntoUTF16(const wchar_t *inputString, A_UTF16Char *destination);

}  // namespace AEUtil
