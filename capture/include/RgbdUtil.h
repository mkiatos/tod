#ifndef _RGBD_UTIL_H
#define _RGBD_UTIL_H


#include <iostream>

#include "opencv2/highgui/highgui.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "PinholeCamera.h"


cv::Mat maxFilter(const cv::Mat &depth);

void depthToPointCloud(const cv::Mat &rgb, const cv::Mat &depth, const PinholeCamera &camera,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

void cloudViewer(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);


#endif
