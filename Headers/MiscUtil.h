//
//  MiscUtil.hpp
//  GPU_Skeleton
//
//  Created by Tim Constantinov on 22.04.2023.
//
#pragma once

#ifndef MiscUtil_hpp
#define MiscUtil_hpp

using namespace std;

#define GET_GPU(gpu)                                   \
  (gpu == PF_GPU_Framework_NONE     ? string("None")   \
   : gpu == PF_GPU_Framework_OPENCL ? string("OpenCL") \
   : gpu == PF_GPU_Framework_METAL  ? string("Metal")  \
   : gpu == PF_GPU_Framework_CUDA   ? string("CUDA")   \
                                    : string("Unknown"))

#define GET_APPLE_CPU(cpu)                        \
  (cpu == APPLE_M1      ? string("Apple Silicon") \
   : cpu == APPLE_INTEL ? string("Apple Intel")   \
                        : string("Unknown"))

#define GET_AE_VERSION(version) to_string(version.major) + "." + to_string(version.minor)

#include <stdio.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../Config.h"

string getVersionString();
vector<string> splitWith(string s, string delimiter);
// ostream& operator<<(ostream& os, const PF_Rect& rect);
string to_string(const PF_Rect &rect);
string to_string(const PF_LayerDef &world);

#endif /* MiscUtil_hpp */
