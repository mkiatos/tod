#ifndef _POINT_CLOUD_REGISTRATION_
#define _POINT_CLOUD_REGISTRATION_


#include <iostream>

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

#include "opencv2/highgui/highgui.hpp"

#include <eigen3/Eigen/Geometry>

#include "RgbdUtil.h"


class PointCloudRegistration{
public:
    PointCloudRegistration(float cx, float cy, float r, float baseDelta_);

    void filterCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                     float cx, float cy, float r, float baseDelta_);

    void pairAlign(pcl::PointCloud<pcl::PointXYZRGB>::Ptr globalCloud,
                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

    void addCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr globalCloud,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                  Eigen::Matrix4d pose);

private:
    float cx_, cy_, r_, baseDelta_;
};


#endif
