#pragma once

#include "logging.hpp"
#include "yolo_app.hpp"

AppConfig MakeDefaultAppConfig();
void InitYoloRuntimeConfig();
const app::logging::LogConfig &GetYoloLogConfig();
