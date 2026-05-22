cmake_minimum_required(VERSION 3.16)
project(CameraApp LANGUAGES CXX)

if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "Available build types" FORCE)
endif()

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else()
    message(STATUS "IPO/LTO not enabled: ${ipo_error}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CAMERAAPP_PERF_SYMBOLS "Build with debug symbols and frame pointers for perf" OFF)
if(CAMERAAPP_PERF_SYMBOLS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -fno-omit-frame-pointer")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
endif()

set(WORKSPACE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_ROOT ${WORKSPACE_ROOT}/2027rm_ws)
set(APP_SRC_DIR ${PROJECT_ROOT}/src/app)
set(CAMERA_SRC_DIR ${PROJECT_ROOT}/src/camera)
set(COMMON_SRC_DIR ${PROJECT_ROOT}/src/common)
set(DISPLAY_SRC_DIR ${PROJECT_ROOT}/src/display)
set(POSE_SRC_DIR ${PROJECT_ROOT}/src/pose)
set(SERIAL_SRC_DIR ${PROJECT_ROOT}/src/serial)
set(VISION_SRC_DIR ${PROJECT_ROOT}/src/vision)
set(APP_INCLUDE_DIRS
    ${PROJECT_ROOT}/src
    ${APP_SRC_DIR}
    ${CAMERA_SRC_DIR}
    ${COMMON_SRC_DIR}
    ${DISPLAY_SRC_DIR}
    ${POSE_SRC_DIR}
    ${SERIAL_SRC_DIR}
    ${VISION_SRC_DIR}
)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH_DIR "64")
else()
    set(ARCH_DIR "32")
endif()

set(CAMERA_SDK_LIB_DIR ${PROJECT_ROOT}/lib/${ARCH_DIR})
set(CAMERA_SDK_LIBS
    MvCameraControl
    MvCameraControlWrapper
    MvCamLVision
)

set(OPENCV_412_DIR "/usr/local/opencv4.12.0/lib/cmake/opencv4")
if(EXISTS "${OPENCV_412_DIR}/OpenCVConfig.cmake" AND NOT OpenCV_DIR)
    set(OpenCV_DIR "${OPENCV_412_DIR}" CACHE PATH "OpenCV 4.12.0 config directory" FORCE)
endif()
find_package(OpenCV 4.12.0 EXACT REQUIRED
    COMPONENTS core imgproc dnn calib3d highgui videoio imgcodecs
)
message(STATUS "Using OpenCV ${OpenCV_VERSION}: ${OpenCV_DIR}")
find_package(yaml-cpp REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(spdlog REQUIRED)

get_target_property(OPENCV_CORE_LOCATION opencv_core IMPORTED_LOCATION_RELEASE)
if(NOT OPENCV_CORE_LOCATION)
    get_target_property(OPENCV_CORE_LOCATION opencv_core IMPORTED_LOCATION)
endif()
get_filename_component(OPENCV_LIBRARY_DIR "${OPENCV_CORE_LOCATION}" DIRECTORY)

file(GLOB_RECURSE CGRAPH_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_ROOT}/include/CGraph/src/*.cpp
)
add_library(cgraph_runtime OBJECT ${CGRAPH_SOURCES})
target_compile_features(cgraph_runtime PUBLIC cxx_std_20)
target_include_directories(cgraph_runtime PUBLIC
    ${PROJECT_ROOT}/include
    ${PROJECT_ROOT}/include/CGraph/src
    ${APP_INCLUDE_DIRS}
)

function(configure_common_target target_name)
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    target_include_directories(${target_name} PRIVATE
        ${PROJECT_ROOT}/include
        ${PROJECT_ROOT}/include/CGraph/src
        ${APP_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_directories(${target_name} PRIVATE
        ${CAMERA_SDK_LIB_DIR}
    )
    target_link_libraries(${target_name} PRIVATE
        ${OpenCV_LIBS}
        yaml-cpp
        spdlog::spdlog
        pthread
    )
endfunction()

add_executable(camera_app
    ${CAMERA_SRC_DIR}/camera.cpp
    ${COMMON_SRC_DIR}/profiling.cpp
)
configure_common_target(camera_app)
target_link_libraries(camera_app PRIVATE ${CAMERA_SDK_LIBS})
target_compile_definitions(camera_app PRIVATE
    CAMERA_CONFIG_PATH="${PROJECT_ROOT}/config/config.yaml"
)

set(DETECTOR_SRC ${APP_SRC_DIR}/detector.cpp)
if(EXISTS ${DETECTOR_SRC})
    add_executable(detector_app
        ${DETECTOR_SRC}
    )
    configure_common_target(detector_app)
endif()

add_executable(cgraph_camera_detector
    $<TARGET_OBJECTS:cgraph_runtime>
    ${APP_SRC_DIR}/startup_main.cpp
    ${APP_SRC_DIR}/yolo_config_loader.cpp
    ${CAMERA_SRC_DIR}/camera.cpp
    ${COMMON_SRC_DIR}/profiling.cpp
    ${COMMON_SRC_DIR}/thread_affinity.cpp
    ${DISPLAY_SRC_DIR}/display_node.cpp
    ${POSE_SRC_DIR}/RGO_Tracker.cpp
    ${POSE_SRC_DIR}/pnp_solver.cpp
    ${SERIAL_SRC_DIR}/serial_imu_node.cpp
    ${SERIAL_SRC_DIR}/serial_protocol.cpp
    ${VISION_SRC_DIR}/postprocess.cpp
    ${VISION_SRC_DIR}/yolo.cpp
    ${VISION_SRC_DIR}/yolo_openvino.cpp
)
configure_common_target(cgraph_camera_detector)
target_link_libraries(cgraph_camera_detector PRIVATE
    ${CAMERA_SDK_LIBS}
    openvino::runtime
)
target_compile_definitions(cgraph_camera_detector PRIVATE
    CAMERA_CONFIG_PATH="${PROJECT_ROOT}/config/config.yaml"
    CAMERA_INFO_PATH="${PROJECT_ROOT}/config/camera_info.yaml"
    YOLO_CONFIG_PATH="${PROJECT_ROOT}/config/yolo_config.yaml"
    CAMERA_NODE_LIBRARY
)

set(ALL_APP_TARGETS camera_app cgraph_camera_detector)
if(TARGET detector_app)
    list(APPEND ALL_APP_TARGETS detector_app)
endif()

add_custom_target(all_apps
    DEPENDS ${ALL_APP_TARGETS}
)

add_custom_target(configure_release
    COMMAND ${CMAKE_COMMAND}
        -S ${WORKSPACE_ROOT}
        -B ${WORKSPACE_ROOT}/2027rm_ws/build
        -DCMAKE_BUILD_TYPE=Release
    WORKING_DIRECTORY ${WORKSPACE_ROOT}
    USES_TERMINAL
)

add_custom_target(build_release
    COMMAND ${CMAKE_COMMAND}
        --build ${WORKSPACE_ROOT}/2027rm_ws/build
        --target cgraph_camera_detector
        -j
    DEPENDS configure_release
    WORKING_DIRECTORY ${WORKSPACE_ROOT}
    USES_TERMINAL
)

add_custom_target(run_camera_app
    COMMAND camera_app
    DEPENDS camera_app
    WORKING_DIRECTORY ${PROJECT_ROOT}
    USES_TERMINAL
)

if(TARGET detector_app)
    add_custom_target(run_detector_app
        COMMAND detector_app
        DEPENDS detector_app
        WORKING_DIRECTORY ${PROJECT_ROOT}
        USES_TERMINAL
    )
endif()

add_custom_target(run_cgraph_camera_detector
    COMMAND cgraph_camera_detector
    DEPENDS cgraph_camera_detector
    WORKING_DIRECTORY ${PROJECT_ROOT}
    USES_TERMINAL
)

set_target_properties(camera_app cgraph_camera_detector PROPERTIES
    BUILD_RPATH "${CAMERA_SDK_LIB_DIR};${OPENCV_LIBRARY_DIR}"
    INSTALL_RPATH "${CAMERA_SDK_LIB_DIR};${OPENCV_LIBRARY_DIR}"
)
