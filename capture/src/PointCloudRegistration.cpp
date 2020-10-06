#include "PointCloudRegistration.h"

#include <pcl/io/ply_io.h>


PointCloudRegistration::PointCloudRegistration(float cx, float cy, float r, float baseDelta):
    cx_(cx), cy_(cy), r_(r), baseDelta_(baseDelta){}


void PointCloudRegistration::filterCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                         float cx, float cy, float r, float baseDelta)
{
    pcl::PointIndices::Ptr  outliers(new pcl::PointIndices());

    float dist, x, y, z;

    for(size_t i = 0; i < cloud->points.size(); i++)
    {
        x = cloud->points[i].x;
        y = cloud->points[i].y;
        z = cloud->points[i].z;

        dist = sqrt( (x - cx)*(x - cx) + (y - cy)*(y - cy) );

        if(dist > r || z > -baseDelta)
        {
            outliers->indices.push_back(i);
        }
    }

    //!< Extract the planar inliers from the point cloud
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(outliers);
    extract.setNegative(true);
    extract.filter(*cloud);
}


void PointCloudRegistration::pairAlign(pcl::PointCloud<pcl::PointXYZRGB>::Ptr globalCloud,
                                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr alignedCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

    Eigen::Matrix4f finalTransform = Eigen::Matrix4f::Identity();

    //!< downsample
    pcl::VoxelGrid<pcl::PointXYZRGB> downsample;
    downsample.setLeafSize(0.01f, 0.01f, 0.01f);

    downsample.setInputCloud(globalCloud);
    downsample.filter(*target);

    downsample.setInputCloud(cloud);
    downsample.filter(*source);


    //!< apply icp
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);

    icp.setTransformationEpsilon (1e-6);
    icp.setMaxCorrespondenceDistance (0.005);

    icp.setMaximumIterations (200);

    icp.align(*alignedCloud, finalTransform);
    pcl::transformPointCloud(*cloud, *cloud, finalTransform);
}


void PointCloudRegistration::addCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr globalCloud,
                                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                      Eigen::Matrix4d pose)
{
    Eigen::Matrix4d poseInv = pose.inverse();
    pcl::transformPointCloud(*cloud, *cloud, poseInv);

    filterCloud(cloud, cx_, cy_, r_, baseDelta_);

    //cloudViewer(cloud);

    if(globalCloud->empty())
    {
        *globalCloud = *cloud;
    }
    else
    {
        pairAlign(globalCloud, cloud);

        *globalCloud = (*globalCloud) + (*cloud);
    }
}
