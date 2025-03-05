cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(melotts)

set(CMAKE_CXX_STANDARD 14)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2 -fdata-sections -ffunction-sections")
endif()

# Remove this include since it brings in Axera dependencies
# include(cmake/msp_dependencies.cmake)

# onnxruntime
include_directories(onnxruntime/include/onnxruntime)
include_directories(onnxruntime/include/onnxruntime/core/session)
link_directories(onnxruntime/lib)

# Remove these
# include_directories(${MSP_INC_DIR})
# link_directories(${MSP_LIB_DIR})

include_directories(src)
aux_source_directory(src SRC)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/bin)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp ${SRC})

# Remove Axera libraries from linking
target_link_libraries(${PROJECT_NAME} onnxruntime onnxruntime_providers_shared)

file(COPY onnxruntime/lib/libonnxruntime.so DESTINATION ${CMAKE_INSTALL_PREFIX})
file(COPY onnxruntime/lib/libonnxruntime.so.1.14.0 DESTINATION ${CMAKE_INSTALL_PREFIX})
file(COPY onnxruntime/lib/libonnxruntime_providers_shared.so DESTINATION ${CMAKE_INSTALL_PREFIX})

install(TARGETS ${PROJECT_NAME}
        RUNTIME
            DESTINATION ./)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/"
)