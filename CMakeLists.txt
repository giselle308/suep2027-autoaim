cmake_minimum_required(VERSION 3.10)
project(CameraApp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WORKSPACE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_ROOT ${WORKSPACE_ROOT}/2027rm_ws)

file(GLOB_RECURSE CGRAPH_SOURCES
    ${PROJECT_ROOT}/include/CGraph/src/*.cpp
)
add_library(cgraph_runtime OBJECT ${CGRAPH_SOURCES})
target_include_directories(cgraph_runtime PUBLIC
    ${PROJECT_ROOT}/include/CGraph/src
)

# 检测系统架构
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH_DIR "64")
else()
    set(ARCH_DIR "32")
endif()

# 头文件路径（SDK 头文件在项目根目录 include 下）
include_directories(${PROJECT_ROOT}/include)

# 库文件路径（SDK 库在项目根目录 lib 下）
link_directories(${PROJECT_ROOT}/lib/${ARCH_DIR})

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# YAML_CPP
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

# 源文件
set(SOURCES ${PROJECT_ROOT}/src/camera.cpp)

# 创建可执行文件
add_executable(camera_app ${SOURCES})

add_executable(detector_app
    ${PROJECT_ROOT}/src/detector.cpp
)

add_executable(cgraph_camera_detector
    $<TARGET_OBJECTS:cgraph_runtime>
    ${PROJECT_ROOT}/src/camera.cpp
    ${PROJECT_ROOT}/src/detector.cpp
    ${PROJECT_ROOT}/src/cgraph_camera_detector.cpp
)

target_include_directories(cgraph_camera_detector PRIVATE
    ${PROJECT_ROOT}/include/CGraph/src
    ${PROJECT_ROOT}/src
)

target_compile_definitions(camera_app PRIVATE
    CAMERA_CONFIG_PATH="${PROJECT_ROOT}/config/config.yaml"
)

# 链接库文件
target_link_libraries(camera_app
    MvCameraControl
    MvCameraControlWrapper
    MvCamLVision
    ${OpenCV_LIBS}
    yaml-cpp
)

target_link_libraries(detector_app
    ${OpenCV_LIBS}
    yaml-cpp
)

target_link_libraries(cgraph_camera_detector
    MvCameraControl
    MvCameraControlWrapper
    MvCamLVision
    ${OpenCV_LIBS}
    yaml-cpp
    pthread
)

target_compile_definitions(cgraph_camera_detector PRIVATE
    CAMERA_CONFIG_PATH="${PROJECT_ROOT}/config/config.yaml"
    CAMERA_INFO_PATH="${PROJECT_ROOT}/config/camera_info.yaml"
    CAMERA_NODE_LIBRARY
    DETECTOR_NODE_LIBRARY
)

add_custom_target(all_apps
    DEPENDS camera_app detector_app cgraph_camera_detector
)

# 设置运行时库路径
set_target_properties(camera_app PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib/${ARCH_DIR}"
)