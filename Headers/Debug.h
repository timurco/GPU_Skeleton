#ifdef DEBUG
#ifndef IS_PIPL
#include <chrono>
#include <iostream>
#endif

#define FX_LOG_HEADER "[GPU_SKELETON]: "
#define FX_LOG_FOOTER "\n============\n"
#define FX_LOG(log) cout << FX_LOG_HEADER << log << FX_LOG_FOOTER << endl
#define FX_LOG_VAL(log, val) cout << FX_LOG_HEADER << log << " = " << val << FX_LOG_FOOTER << endl

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
