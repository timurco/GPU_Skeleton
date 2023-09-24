#ifdef DEBUG
#ifndef IS_PIPL
#include <chrono>
#include <iostream>
#endif

#ifndef _WIN32
static void OutputDebugString(const char* msg) { std::cout << msg; }
static void OutputDebugStringA(const char* msg) { std::cout << msg; }
#endif

#ifdef _WIN32
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define FX_LOG_HEADER "[GPU_SKELETON]: "
#define FX_LOG_FOOTER "\n============\n"
#define FX_LOG(log) do { std::ostringstream oss; oss << FX_LOG_HEADER << log << FX_LOG_FOOTER; OutputDebugStringA(oss.str().c_str()); } while(0)
#define FX_LOG_VAL(log, val) do { std::ostringstream oss; oss << FX_LOG_HEADER << log << " = " << val << FX_LOG_FOOTER; OutputDebugStringA(oss.str().c_str()); } while(0)
#define FX_LOG_ERR(msg, ...) \
  FX_LOG("ERROR :: " << std::string(__FILENAME__) << " [" << std::to_string(__LINE__) << "] :: " << msg, ##__VA_ARGS__)

#define FX_LOG_TIME_START(name) auto name = chrono::system_clock::now();
#define FX_LOG_TIME_END(name, message) \
  FX_LOG(message << " time ="          \
  << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - name).count() << "ms");

#define FX_LOG_RECT(label, rect) \
  FX_LOG(label << "=(" << rect.left << ", " << rect.top << ", " << rect.right << ", " << rect.bottom << ")");

#else
#define FX_LOG(log)
#define FX_LOG_VAL(log, val)
#define FX_LOG_TIME_START(name)
#define FX_LOG_TIME_END(name, message)
#define FX_LOG_RECT(label, rect)

#endif
