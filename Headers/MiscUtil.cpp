//
//  MiscUtil.cpp
//  GPU_Skeleton
//
//  Created by Tim Constantinov on 22.04.2023.
//

#include "MiscUtil.h"

string getVersionString() {
  string stageStr;
  switch (STAGE_VERSION) {
    case PF_Stage_DEVELOP:
      stageStr = "Develop";
      break;
    case PF_Stage_ALPHA:
      stageStr = "Alpha";
      break;
    case PF_Stage_BETA:
      stageStr = "Beta";
      break;
    case PF_Stage_RELEASE:
      stageStr = "Release";
      break;
    default:
      stageStr = "Unknown";
      break;
  }

  return string(CONFIG_NAME) + " v" + to_string(MAJOR_VERSION) + "." + to_string(MINOR_VERSION) +
         "." + to_string(BUILD_VERSION) + " " + stageStr;
}

vector<string> splitWith(string s, string delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  string token;
  vector<string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));

  return res;
}

// ostream& operator<<(ostream& os, const PF_Rect& rect) {
//   os << " { left: " << rect.left << ", top: " << rect.top
//   << ", right: " << rect.right << ", bottom: " << rect.bottom << " }";
//   return os;
// }

string to_string(const PF_Rect &rect) {
  ostringstream oss;
  oss << "[ left: " << rect.left << ", top: " << rect.top << ", right: " << rect.right
      << ", bottom: " << rect.bottom << " ]  ";
  return oss.str();
}

string to_string(const PF_LayerDef &world) {
  ostringstream oss;
  oss << "[ x: " << world.origin_x << ", y: " << world.origin_y << ", width: " << world.width
      << ", height: " << world.height << ", rowbytes: " << world.rowbytes
      << ", extend:" << to_string(world.extent_hint) << " ]";
  return oss.str();
}
