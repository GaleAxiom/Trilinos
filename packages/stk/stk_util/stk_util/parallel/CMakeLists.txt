# Copyright 2002 - 2008, 2010, 2011 National Technology Engineering
# Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
# 
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
# 
#     * Neither the name of NTESS nor the names of its contributors
#       may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

SET(HEADERS "")
SET(SOURCES "")

IF (STK_HAS_MPI)

FILE(GLOB HEADERS CONFIGURE_DEPENDS *.hpp *.h)
FILE(GLOB SOURCES CONFIGURE_DEPENDS *.cpp)

ELSE()

FILE(GLOB HEADERS
     CommBuffer.hpp
     OutputStreams.hpp
     Parallel.hpp
     ParallelComm.hpp
     ParallelReduce.hpp
     ParallelReduceBool.hpp
     ParallelVectorConcat.hpp
     mpi_filebuf.hpp
    )
FILE(GLOB SOURCES
     CommBuffer.cpp
     OutputStreams.cpp
     Parallel.cpp
     ParallelComm.cpp
     ParallelReduce.cpp
     mpi_filebuf.cpp
    )

ENDIF()

IF(HAVE_STK_Trilinos)
  TRIBITS_ADD_LIBRARY(stk_util_parallel
    NOINSTALLHEADERS ${HEADERS}
    SOURCES ${SOURCES}
    DEPLIBS stk_util_util
  )
ELSE()
  add_library(stk_util_parallel ${SOURCES})

  IF(STK_HAS_MPI)
    target_link_libraries(stk_util_parallel PUBLIC MPI::MPI_CXX)
    message("MPI_LIBRARIES: ${MPI_LIBRARIES}")
  ENDIF()

  target_link_libraries(stk_util_parallel PUBLIC stk_util_util)

  IF(${PROJECT_NAME}_ENABLE_SEACASAprepro_lib)
    find_package(SEACASAprepro_lib REQUIRED)
    target_link_libraries(stk_util_parallel PUBLIC SEACASAprepro_lib::aprepro_lib)
  ENDIF()
ENDIF()

target_include_directories(stk_util_parallel PUBLIC
  $<BUILD_INTERFACE:${STK_TOPLEVEL_SOURCE_DIR}/stk_util>
  $<INSTALL_INTERFACE:include>
)

INSTALL(FILES ${HEADERS} DESTINATION
    ${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}/stk_util/parallel/)

IF(NOT HAVE_STK_Trilinos)
  INSTALL(TARGETS stk_util_parallel EXPORT stkTargets DESTINATION ${STK_INSTALL_LIBDIR})
ENDIF()
