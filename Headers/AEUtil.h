/*
 Modified from: https://github.com/baku89/ISF4AE
*/

#pragma once

#include <string>

#include "AEFX_SuiteHelper.h"
#include "AEGP_SuiteHandler.h"
#include "AE_Effect.h"
#include "AE_Macros.h"

using namespace std;

namespace AEUtil {

string getResourcesPath(PF_InData *in_data);

// Other AE-specific utils
void copyConvertStringLiteralIntoUTF16(const wchar_t *inputString, A_UTF16Char *destination);

}  // namespace AEUtil
