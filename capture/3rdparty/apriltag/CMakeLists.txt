PROJECT(apriltag)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common/)

ADD_DEFINITIONS("-std=gnu99 -Wall -Wno-unused-parameter -Wno-unused-function -fPIC")

SET ( APRILTAG_SOURCES 
	apriltag.c apriltag_quad_thresh.c tag16h5.c tag25h7.c tag25h9.c tag36h10.c tag36h11.c tag36artoolkit.c g2d.c 
	common/zarray.c common/zhash.c common/zmaxheap.c common/unionfind.c common/matd.c common/image_u8.c common/pnm.c common/image_f32.c common/image_u32.c common/workerpool.c common/time_util.c 
	common/svd22.c common/homography.c common/string_util.c common/getopt.c
)

#linking 

ADD_LIBRARY(apriltag STATIC ${APRILTAG_SOURCES})
TARGET_LINK_LIBRARIES(apriltag pthread m)
SET_TARGET_PROPERTIES(apriltag PROPERTIES DEBUG_POSTFIX d)



