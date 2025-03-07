#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

find_program(TECOCC
    NAMES tecocc
    HINTS ENV SDAA_HOME
    PATHS ENV PATH
    PATH_SUFFIXES bin
    DOC "tecocc location."
    NO_DEFAULT_PATH
)
message(STATUS "tecocc path: ${TECOCC}")

set(SDAA_INCLUDES $ENV{SDAA_HOME}/include)
message(STATUS "sdaa include dir: ${SDAA_INCLUDES}")

macro(sdaa_add_library sdaa_target)
  set(sources)
  set(headers)
  set(output_files)
  set(sdaa_build_shared_lib)
  set(sdaa_static_whole_archive FALSE)

  foreach(arg ${ARGN})
    if("${arg}" STREQUAL "STATIC")
      set(sdaa_build_shared_lib STATIC)
    elseif("${arg}" STREQUAL "SHARED")
      set(sdaa_build_shared_lib SHARED)
    elseif("${arg}" STREQUAL "EXPORT_ALL")
      set(sdaa_static_whole_archive TRUE)
    else()
      get_filename_component(file_dir "${arg}" DIRECTORY)

      if(IS_ABSOLUTE "${file_dir}")
        set(source "${arg}")
      else()
        set(source "${CMAKE_CURRENT_SOURCE_DIR}/${arg}")
      endif()

      list(APPEND sources ${source})

      if(${arg} MATCHES "\\.h(pp)?$")
        list(APPEND headers ${source})
      endif()
    endif()
  endforeach()

  set(generate_target_name ${sdaa_target}_generate)
  set(generate_path "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${generate_target_name}.dir")
  message(STATUS "generate_path ${generate_path}")

  foreach(source ${sources})
    if(${source} MATCHES "\\.scpp$")
      file(RELATIVE_PATH relative_file_path ${CMAKE_CURRENT_SOURCE_DIR} ${source})
      get_filename_component(basename ${relative_file_path} NAME)
      get_filename_component(relative_file_dir ${relative_file_path} DIRECTORY)
      if(NOT EXISTS ${generate_path}/${relative_file_dir})
        file(MAKE_DIRECTORY ${generate_path}/${relative_file_dir})
      endif()

      set(output_file "${generate_path}/${relative_file_dir}/${basename}.o")
      add_custom_command(OUTPUT ${output_file}
        DEPENDS ${source} ${headers}
        COMMAND "${TECOCC}"
                "${source}"
                -DUSE_EXPERIMENTAL_SDAA_API
                -O2
                -flto
                -fPIC
                -c
                -o
                "${output_file}"
        )
      add_custom_target(${basename}_target
        DEPENDS ${output_file})
      set_source_files_properties(${output_file} PROPERTIES GENERATED TRUE)
      list(APPEND output_files ${output_file})
    endif()
  endforeach()

  if("${sdaa_build_shared_lib}" STREQUAL "STATIC")
    set(library_name lib${sdaa_target}.a)
    set(sdaa_static_option -sdaa-static-lib)
  else() # SHARED or empty
    set(library_name lib${sdaa_target}.so)
    set(sdaa_static_option -Wl,-soname,${library_name})
  endif()

  set(library_file ${CMAKE_CURRENT_BINARY_DIR}/${library_name})
  add_custom_command(OUTPUT ${library_file}
    DEPENDS ${output_files}
    COMMAND "${TECOCC}"
            "${output_files}"
            -DUSE_EXPERIMENTAL_SDAA_API
            -flto
            --sdaa-link
            -fPIC
            -shared
            ${sdaa_static_option}
            -o
            "${library_file}"
    COMMAND_EXPAND_LISTS)
  add_custom_target(${generate_target_name} ALL
    DEPENDS ${library_file})

  if("${sdaa_build_shared_lib}" STREQUAL "STATIC")
    add_library(${sdaa_target} STATIC IMPORTED)
    add_dependencies(${sdaa_target} ${generate_target_name})
    set_target_properties(${sdaa_target}
                          PROPERTIES IMPORTED_LOCATION ${library_file})
    if(sdaa_static_whole_archive)
      message(WARNING "${library_file} is repeated twice on the link command")
      set(_sdaa_link_libs -Wl,--whole-archive,${library_file},--no-whole-archive)
      set_target_properties(${sdaa_target}
                            PROPERTIES INTERFACE_LINK_LIBRARIES ${_sdaa_link_libs})
    endif()
  else() # SHARED
    add_library(${sdaa_target} SHARED IMPORTED)
    add_dependencies(${sdaa_target} ${generate_target_name})
    target_link_directories(${sdaa_target} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${sdaa_target}
                          PROPERTIES IMPORTED_LOCATION ${library_file})
  endif()
endmacro()

add_custom_target(plugin)

include_directories(/opt/tecoai/include)

set(TARGET_NAME "TecoInferPlugin")
set(SHARED_TARGET ${TARGET_NAME})
set(FIND_SDAART "/opt/tecoai/lib64")
set(FIND_PluginUtil "/root/miniconda3/envs/tvm-build/lib/python3.8/site-packages/tvm")

include_directories(/opt/tecoai/include)
include_directories(/opt/tecoai/include/dyn/include)
include_directories(/opt/tecoai/include/dyn/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(PLUGIN_SOURCES)
set(PLUGIN_SCPP_SOURCES)

set(PLUGIN_LISTS
    pluginAdd
)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

sdaa_add_library(plugin_kernel STATIC ${PLUGIN_SCPP_SOURCES})

message(STATUS "PLUGIN_SOURCES: ${PLUGIN_SOURCES}")
message(STATUS "GENERATED_SCPP_FILES: ${plugin_kernel}")

add_library(${SHARED_TARGET} SHARED
        ${PLUGIN_SOURCES}
)

set(PLUGIN_UTIL_PATH ${CMAKE_CURRENT_SOURCE_DIR}../)
target_include_directories(${SHARED_TARGET}
        PUBLIC ${PLUGIN_UTIL_PATH})

set_target_properties(${SHARED_TARGET} PROPERTIES
        CXX_STANDARD "17"
        CXX_STANDARD_REQUIRED "YES"
        CXX_EXTENSIONS "NO"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugin/example"
)

target_link_libraries(${SHARED_TARGET} 
    -Wl,--whole-archive plugin_kernel -Wl,--no-whole-archive
    ${FIND_SDAART}/libsdaart.so
    ${FIND_PluginUtil}/libTecoInferPluginUtil.so 
)

add_dependencies(${SHARED_TARGET} TecoInferPluginUtil)
