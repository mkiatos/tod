project(calibration)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Debug)

find_package(PCL 1.7.2 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VTK 6 REQUIRED)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

aux_source_directory(./src SRC_LIST)
file(GLOB INCLUDE_LIST "./include/*.h" "./include/*.hpp" "./include/cuda/*.h")

include_directories(
    ./include
    /usr/local/include/eigen3        
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}        
)

add_executable(${PROJECT_NAME} ${SRC_LIST} ${INCLUDE_LIST})

link_directories(./lib)

target_link_libraries(calibration
    ${BOOST_LIBRARIES}
    ${OpenCV_LIBS}
    ${VTK_LIBRARIES}
    ${PCL_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/lib/libapriltags.a
)
