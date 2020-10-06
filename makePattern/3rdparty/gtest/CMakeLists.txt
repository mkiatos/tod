cmake_minimum_required(VERSION 2.8.8)
project(gtest_builder C CXX)
include(ExternalProject)

IF ( MSVC )
ExternalProject_Add(googletest
	URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
    CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
               -Dgtest_force_shared_crt=ON
     PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
# Disable install step
    INSTALL_COMMAND ""
)

# Specify include dir
ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIRS ${source_dir}/include PARENT_SCOPE)

ExternalProject_Get_Property(googletest binary_dir)
set(GTEST_LIBS_DIR ${binary_dir} )
set(GTEST_LIBRARIES_GTEST debug ${GTEST_LIBS_DIR}/DebugLibs/gtest.lib optimized ${GTEST_LIBS_DIR}/ReleaseLibs/gtest.lib )
set(GTEST_LIBRARIES_GTEST_MAIN debug ${GTEST_LIBS_DIR}/DebugLibs/gtest_main.lib optimized ${GTEST_LIBS_DIR}/ReleaseLibs/gtest_main.lib )

set(GTEST_LIBRARIES ${GTEST_LIBRARIES_GTEST} ${GTEST_LIBRARIES_GTEST_MAIN} PARENT_SCOPE)


ELSE ( MSVC )

ExternalProject_Add(googletest
	URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
    INSTALL_COMMAND ""
)

# Specify include dir
ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIRS ${source_dir}/include PARENT_SCOPE)

# Specify MainTest's link libraries
ExternalProject_Get_Property(googletest binary_dir)
set(GTEST_LIBS_DIR ${binary_dir} )
set(GTEST_LIBRARIES ${GTEST_LIBS_DIR}/libgtest.a ${GTEST_LIBS_DIR}/libgtest_main.a PARENT_SCOPE)


ENDIF ( MSVC )

