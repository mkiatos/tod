cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

PROJECT(detect)

find_package( OpenCV 2.4.9 REQUIRED )
find_package( PCL 1.7.2 REQUIRED )

aux_source_directory(./src SRC_LIST)
file(GLOB INCLUDE_LIST "./include/*.h" "./include/*.hpp")

include_directories(
./include
${Boost_INCLUDE_DIR}
${PCL_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)

add_executable( ${PROJECT_NAME} ${SRC_LIST} ${INCLUDE_LIST})

target_link_libraries( 
${PROJECT_NAME}
${OpenCV_LIBS}
${PCL_LIBRARIES}
${Boost_LIBRARIES}
)
