#include "Tod.h"


namespace fs = boost::filesystem;


struct ViewPointDescriptor
{
    std::vector<Eigen::Vector4d> kp_;
    std::vector<cv::Mat> sift_;
    Eigen::Matrix4d pose_;
    int id_;
};


int noOfFiles(fs::path &p)
{
    fs::directory_iterator end_iter;

    int count = 0;

    if(fs::exists(p) && fs::is_directory(p))
    {
        for(fs::directory_iterator dir_iter(p) ; dir_iter != end_iter ; ++dir_iter)
        {
            if (fs::is_regular_file(dir_iter->status())) count++;
        }
    }
    return count;
}


void ObjectDetector::train(boost::filesystem::path &dir, PinholeCamera &cam)
{
    fs::directory_iterator dit(dir), dend;
    for(; dit != dend; ++dit)
    {
        if (!fs::is_directory(dit->path())) continue;

        fs::path p = dit->path();
        trainSingleObject(p, cam);
    }
}


void ObjectDetector::trainSingleObject(boost::filesystem::path &dir, PinholeCamera &cam)
{
    fs::path outPath = dir / "todData.bin";
    
    std::ofstream strm(outPath.c_str(), std::ios::binary);
    
    BinaryStream ar(strm);
    
    fs::path framesPath = dir / "rgb";
    int noOfViews = noOfFiles(framesPath);
    ar << noOfViews;
    
    for(int i = 0; i < noOfViews; i++)
    {
        fs::path posePath = dir / str(boost::format("poses/pose%d.txt") % i);
        fs::path rgbPath = dir / str(boost::format("rgb/rgb%d.png") % i);
        fs::path maskPath = dir / str(boost::format("masks/binaryMask%d.png") % i);
        fs::path dcPath = dir / str(boost::format("dc/dc%d.png") % i);
        
        cv::Mat rgb = cv::imread(rgbPath.string(), CV_LOAD_IMAGE_COLOR);
        cv::Mat mask = cv::imread(maskPath.string(), CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat dc = cv::imread(dcPath.string(), CV_16U);

        ViewPointDescriptor vpd;

        vpd.id_ = i;

        //!< read pose
        std::ifstream input(posePath.string().c_str());
        for(int x = 0; x < 4; x++)
        {
            for(int y = 0; y < 4; y++)
            {
                input >> vpd.pose_(x, y);
            }
        }

        Eigen::Matrix4d ipose = vpd.pose_.inverse();

        //!< compute sift features
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;

        cv::SIFT sift(0, 3, 0.04, 10, 1.6);

        sift.detect(rgb, keypoints, mask);
        sift.compute(rgb, keypoints, descriptor);

        //cv::Mat output;
        //cv::drawKeypoints(rgb, keypoints, output);
        //cv::imshow("rgb", output);
        //while(cvWaitKey(0)==27)break;

        for(int j = 0; j < keypoints.size(); j++)
        {
            const cv::KeyPoint &kp = keypoints[j];
            unsigned short depthValue = dc.at<unsigned short>(kp.pt.y, kp.pt.x);
            cv::Point3d bp = cam.backProject(kp.pt, depthValue);

            Eigen::Vector4d q = ipose * Eigen::Vector4d(bp.x, bp.y, bp.z, 1.0);

            vpd.kp_.push_back(q);
            vpd.sift_.push_back(descriptor.row(j));
        }

        ar << vpd.id_ ;

        ar << vpd.kp_ << vpd.sift_ ;

        std::cout<< i <<std::endl;
    }
}
