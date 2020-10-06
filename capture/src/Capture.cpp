	#include "Capture.h"


namespace fs = boost::filesystem;


int noOfFiles(boost::filesystem::path &path)
{
    fs::directory_iterator end_iter;

    int count = 0;

    if(fs::exists(path) && fs::is_directory(path))
    {
        for(fs::directory_iterator dir_iter(path) ; dir_iter != end_iter ; ++dir_iter)
        {
            if (fs::is_regular_file(dir_iter->status()) ) count++;
        }
    }
    return count;
}


void fitCircle(const std::vector<cv::Point2f> &pts, float &cx, float &cy, float &r)
{
    float xMax, yMax, xMin, yMin;
    xMax = xMin = pts[0].x;
    yMax = yMin = pts[0].y;

    for(int i = 1; i<pts.size(); i++)
    {
        if(pts[i].x > xMax) xMax = pts[i].x;
        if(pts[i].x < xMin) xMin = pts[i].x;
        if(pts[i].y > yMax) yMax = pts[i].y;
        if(pts[i].y < yMin) yMin = pts[i].y;
    }

    cx = ( xMax + xMin ) / 2;
    cy = ( yMax + yMin ) / 2;

    if(xMax - cx >= yMax - cy) r = xMax - cx;
    else r = yMax - cy;
}




void CapturePipeline::findPoses()
{
    fs::path framePath = objectPath_ / "rgb";
    int frames = noOfFiles(framePath);

    AprilTagDetector::AprilTagParameters apriltagParams;
    AprilTagDetector tagDetector;
    PoseEstimator estimator;

    std::vector<cv::Point2f> pointsList;

    for(int i = 0; i < frames; i++)
    {
        cv::Mat rgb = cv::imread((framePath / str(boost::format("rgb%d.png") % i)).string());

        std::vector<cv::Point2f> pts;
        std::vector<cv::Point3f> objs;

        if(tagDetector.findPoints(pts, objs, rgb, apriltagParams) == false) continue;

        Eigen::Affine3d tr = estimator.estimate(pts, objs, params_.camera_);

        Eigen::Matrix4d pose = tr.matrix();

        fs::path posePath = objectPath_ / str(boost::format("poses/pose%d.txt") % i);

        std::ofstream strm(posePath.string().c_str());
        strm << pose << std::endl;

        for(int j = 0; j < objs.size(); j++)
        {
            strm << objs[j].x << ' ' << objs[j].y << std::endl;
            pointsList.push_back(cv::Point2f(objs[j].x, objs[j].y));
        }
    }

    float cx, cy, r;
    fitCircle(pointsList, cx , cy, r);

    std::ofstream strm( (objectPath_ / "turntable.txt").string().c_str() );
    strm << cx << ' ' << cy << ' ' << r << std::endl;
}


void readPose(std::string pathPose, Eigen::Matrix4d &pose)
{
    std::ifstream strm(pathPose.c_str());

    pose = Eigen::Matrix4d::Identity();

    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            strm >> pose(i,j);
        }
    }
}


void findLargestCluster(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    int maxSize = 0;

    //!< Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(0.001); //!< 1mm
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(cloud->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);

    for(std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin(); it != clusterIndices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr largestCluster (new pcl::PointCloud<pcl::PointXYZRGB>);
        for(std::vector<int>::const_iterator pit = it->indices.begin(); pit!=it->indices.end(); ++pit)
        {
            largestCluster->points.push_back(cloud->points[*pit]);
        }
        //!< keep the largest cluster
        if(largestCluster->points.size() > maxSize)
        {
            maxSize = largestCluster->points.size();
            *cloud = *largestCluster;
        }
    }
}


void generateMesh(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PolygonMesh &mesh)
{
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize (0.005f, 0.005f, 0.005f);
    vg.filter(*cloud);

    //!< normal surface estimation
    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNumberOfThreads(8);
    ne.setInputCloud(cloud);
    ne.setRadiusSearch(0.01);

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    ne.setViewPoint(centroid[0], centroid[1], centroid[2]);
    pcl::PointCloud<pcl::Normal>::Ptr cloudNormals (new pcl::PointCloud<pcl::Normal>());
    ne.compute(*cloudNormals);

    //!< change the direction of normals
    for(size_t i=0;i<cloudNormals->size();i++)
    {
        cloudNormals->points[i].normal_x *= -1;
        cloudNormals->points[i].normal_y *= -1;
        cloudNormals->points[i].normal_z *= -1;
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudWithNormals (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    pcl::concatenateFields (*cloud, *cloudNormals, *cloudWithNormals);

    //!< poisson reconsrtuction
    pcl::Poisson<pcl::PointXYZRGBNormal> poisson;
    poisson.setDepth(9);
    poisson.setInputCloud(cloudWithNormals);
    poisson.reconstruct(mesh);
}


void CapturePipeline::mergePointClouds()
{
    fs::path framePath = objectPath_ / "rgb";
    int frames = noOfFiles(framePath);

    std::ifstream strm( (objectPath_ / "turntable.txt").string().c_str() );
    float cx, cy, r;
    strm >> cx >> cy >> r;

    PointCloudRegistration preg(cx, cy, r, params_.baseDelta);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr globalCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

    for(int i = 0; i < frames; i++)
    {
        fs::path posePath = objectPath_ / str(boost::format("poses/pose%d.txt") % i);
        fs::path depthPath = objectPath_ / str(boost::format("depth/depth%d.png") % i);
        fs::path rgbPath = objectPath_ / str(boost::format("rgb/rgb%d.png") % i);

        if( !fs::exists(posePath) ) continue;

        cv::Mat depth = cv::imread(depthPath.string(), CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat rgb = cv::imread(rgbPath.string(), CV_LOAD_IMAGE_COLOR);

        //!< load pose
        Eigen::Matrix4d pose;
        readPose(posePath.string(), pose);

        //!< create point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        depthToPointCloud(rgb, maxFilter(depth), params_.camera_, cloud);

        preg.addCloud(globalCloud, cloud, pose);
    }

    findLargestCluster(globalCloud);

    //!< compute the 3d centroid of point cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*globalCloud, centroid);

    //!<
    Eigen::Matrix4d transformAxes = Eigen::Matrix4d::Identity();
    transformAxes(0,3) = centroid[0];
    transformAxes(1,3) = centroid[1];
    transformAxes(2,3) = centroid[2];

    //!< rotate Z by 90 degrees
    Eigen::Matrix4d rotateZ = Eigen::Matrix4d::Identity();
    rotateZ(2,2) = -1;
    transformAxes = transformAxes * rotateZ;

    std::ofstream out( (objectPath_ / "trasnformAxes.txt").string().c_str() );
    out << transformAxes;

    //Eigen::Matrix4d poseInv = transformAxes.inverse();
    //pcl::transformPointCloud(*globalCloud, *globalCloud, poseInv);

    //cloudViewer(globalCloud);
    ////////

    pcl::PolygonMesh mesh;
    generateMesh(globalCloud, mesh);

    fs::path plyPath = objectPath_ / "ply";

    pcl::io::savePLYFile((plyPath / "global_cloud.ply").string(), *globalCloud);
    pcl::io::savePLYFile((plyPath / "model.ply").string(), mesh);

}


void CapturePipeline::makeMasks()
{
    fs::path framePath = objectPath_ / "rgb";
    int frames = noOfFiles(framePath);

    fs::path modelPath = objectPath_ / "ply/model.ply";
    OffScreenRenderer rdr(params_.camera_, modelPath);

    for(int i = 0; i < frames; i++)
    {
        fs::path posePath = objectPath_ / "poses";
        fs::path dcPath = objectPath_ / "dc";
        fs::path masksPath = objectPath_ / "masks";

        if( !fs::exists(posePath / str(boost::format("pose%d.txt") % i)) ) continue;

        Eigen::Matrix4d pose;
        readPose((posePath / str(boost::format("pose%d.txt") % i)).string(), pose);

        rdr.render(pose);

        cv::Mat depth = rdr.getDepth();
        cv::imwrite((dcPath / str(boost::format("dc%d.png") % i)).string(), depth);

        cv::Mat rgb = rdr.getRgb();

        //!< gray scale mask
        cv::Mat gray;
        cvtColor(rgb, gray, CV_BGR2GRAY);

        cv::Mat mask = cv::Mat(rgb.size(), CV_8UC1);
        cv::threshold(gray, mask, 30, 255, 0);

        //cv::Mat dilateMat = cv::Mat::zeros(5, 5, CV_8U);
        //dilateMat.at<uchar>(0,0) = 1;
        //dilateMat.at<uchar>(0,4) = 1;
        //dilateMat.at<uchar>(4,0) = 1;
        //dilateMat.at<uchar>(4,4) = 1;
        //cv::dilate(mask, mask, dilateMat, cv::Point(-1, -1), 1, 1, 1);
        //cv::imwrite((masksPath / str(boost::format("mask%d.png") % i)).string() , mask);

        //!< binary mask
        cv::Mat binaryMask = cv::Mat(gray.size(), CV_8U);
        cv::threshold(mask, binaryMask, 30, 1, 0);
        cv::imwrite((masksPath / str(boost::format("binaryMask%d.png") % i)).string() , binaryMask);

        cv::Mat masked;
        rgb = cv::imread( (objectPath_ / str(boost::format("rgb/rgb%d.png") % i) ).string(), CV_LOAD_IMAGE_COLOR);
        rgb.copyTo(masked, binaryMask);
        cv::imwrite( (masksPath / str(boost::format("masked%d.png") % i)).string() , masked);
    }
}
