/*
 Modified from: https://github.com/baku89/ISF4AE
*/

#include "AEUtil.h"
#include <codecvt>
#include "AE_EffectCB.h"
#include "MiscUtil.h"

namespace AEUtil {

string getResourcesPath(PF_InData *in_data) {
  // initialize and compile the shader objects
  A_UTF16Char pluginFolderPath[AEFX_MAX_PATH];
  PF_GET_PLATFORM_DATA(PF_PlatData_EXE_FILE_PATH_W, &pluginFolderPath);

#ifdef AE_OS_WIN
#error "The resource path is only relevant for macOS"
#endif
#ifdef AE_OS_MAC
  NSUInteger length = 0;
  A_UTF16Char *tmp = pluginFolderPath;
  while (*tmp++ != 0) {
    ++length;
  }
  NSString *newStr = [[NSString alloc] initWithCharacters:pluginFolderPath length:length];
  string resourcePath([newStr UTF8String]);
  resourcePath += "/Contents/Resources/";
#endif
  return resourcePath;
}

/**
 * Function to convert and copy string literals to A_UTF16Char.
 * On Win: Pass the input directly to the output
 * On Mac: All conversion happens through the CFString format
 */
void copyConvertStringLiteralIntoUTF16(const wchar_t *inputString, A_UTF16Char *destination) {
#ifdef AE_OS_MAC
  size_t length = wcslen(inputString);
  CFRange range = {0, AEGP_MAX_PATH_SIZE};
  range.length = length;
  CFStringRef inputStringCFSR =
      CFStringCreateWithBytes(kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(inputString),
                              length * sizeof(wchar_t), kCFStringEncodingUTF32LE, FALSE);
  CFStringGetBytes(inputStringCFSR, range, kCFStringEncodingUTF16, 0, FALSE,
                   reinterpret_cast<UInt8 *>(destination), length * (sizeof(A_UTF16Char)), NULL);
  destination[length] = 0;  // Set NULL-terminator, since CFString calls don't set it
  CFRelease(inputStringCFSR);
#elif defined AE_OS_WIN
  size_t length = wcslen(inputString);
  wcscpy_s(reinterpret_cast<wchar_t *>(destination), length + 1, inputString);
#endif
}

}  // namespace AEUtil
