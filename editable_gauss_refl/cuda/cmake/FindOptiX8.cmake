# FindOptiX8.cmake

# Try to locate the Optix installation using the OPTIX_HOME environment variable
find_path(OPTIX_HOME include/optix.h 
    PATHS
        ENV OPTIX_HOME
        ENV OPTIX_ROOT
        ${CMAKE_CURRENT_LIST_DIR}/../csrc/third_party/optix  # default OptiX path
    DOC "Path to Optix installation.")

if(${OPTIX_HOME} STREQUAL "OptiX8_HOME-NOTFOUND")
	if (${OptiX8_FIND_REQUIRED})
        message(FATAL_ERROR "OPTIX_HOME not defined")
	elseif(NOT ${OptiX8_FIND_QUIETLY})
        message(STATUS "OPTIX_HOME not defined")
	endif()
endif()

# Include
find_path(OptiX8_INCLUDE_DIR 
	NAMES optix.h
    PATHS "${OPTIX_HOME}/include"
	NO_DEFAULT_PATH
	)
find_path(OptiX8_INCLUDE_DIR
	NAMES optix.h
	)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX8 DEFAULT_MSG 
	OptiX8_INCLUDE_DIR)

set(OptiX8_INCLUDE_DIRS ${OptiX8_INCLUDE_DIR})
if(WIN32)
	set(OptiX8_DEFINITIONS NOMINMAX)
endif()
mark_as_advanced(OptiX8_INCLUDE_DIRS OptiX8_DEFINITIONS)

add_library(OptiX8 INTERFACE)
target_compile_definitions(OptiX8 INTERFACE ${OptiX8_DEFINITIONS})
target_include_directories(OptiX8 INTERFACE ${OptiX8_INCLUDE_DIRS})
if(NOT WIN32)
    target_link_libraries(OptiX8 INTERFACE dl)
endif()