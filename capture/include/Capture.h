#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/PointIndices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/poisson.h>
#include <pcl/io/ply_io.h>

#include "opencv2/highgui/highgui.hpp"

#include <vector>

#include "PoseEstimation.h"
#include "PinholeCamera.h"
#include "PointCloudRegistration.h"
#include "RgbdUtil.h"
#include "OffScreenRenderer.h"


class CapturePipeline{
public:

    struct Parameters{
        PinholeCamera camera_;
        float baseDelta;

        Parameters(): camera_(PinholeCamera(525.0, 525.0, 320.0, 240.0, cv::Size(640, 480))), baseDelta(0.01){}
    };


    CapturePipeline(boost::filesystem::path &objectPath, CapturePipeline::Parameters params):
                    objectPath_(objectPath), params_(params){}

    void findPoses();

    void mergePointClouds();

    void makeMasks();

private:
    boost::filesystem::path objectPath_;
    Parameters params_;
};

