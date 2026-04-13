cmake_minimum_required(VERSION 3.16)
project(CameraApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(WORKSPACE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_ROOT ${WORKSPACE_ROOT}/2027rm_ws)

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

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

file(GLOB_RECURSE CGRAPH_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_ROOT}/include/CGraph/src/*.cpp
)
add_library(cgraph_runtime OBJECT ${CGRAPH_SOURCES})
target_include_directories(cgraph_runtime PUBLIC
    ${PROJECT_ROOT}/include
    ${PROJECT_ROOT}/include/CGraph/src
    ${PROJECT_ROOT}/src
)

function(configure_common_target target_name)
    target_include_directories(${target_name} PRIVATE
        ${PROJECT_ROOT}/include
        ${PROJECT_ROOT}/include/CGraph/src
        ${PROJECT_ROOT}/src
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_directories(${target_name} PRIVATE
        ${CAMERA_SDK_LIB_DIR}
    )
    target_link_libraries(${target_name} PRIVATE
        ${OpenCV_LIBS}
        yaml-cpp
        pthread
    )
endfunction()

add_executable(camera_app
    ${PROJECT_ROOT}/src/camera.cpp
)
configure_common_target(camera_app)
target_link_libraries(camera_app PRIVATE ${CAMERA_SDK_LIBS})
target_compile_definitions(camera_app PRIVATE
    CAMERA_CONFIG_PATH="${PROJECT_ROOT}/config/config.yaml"
)

set(DETECTOR_SRC ${PROJECT_ROOT}/src/detector.cpp)
if(EXISTS ${DETECTOR_SRC})
    add_executable(detector_app
        ${DETECTOR_SRC}
    )
    configure_common_target(detector_app)
endif()

add_executable(cgraph_camera_detector
    $<TARGET_OBJECTS:cgraph_runtime>
    ${PROJECT_ROOT}/src/camera.cpp
    ${PROJECT_ROOT}/src/yolo.cpp
    ${PROJECT_ROOT}/src/startup_main.cpp
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
    BUILD_RPATH "${CAMERA_SDK_LIB_DIR}"
    INSTALL_RPATH "${CAMERA_SDK_LIB_DIR}"
)