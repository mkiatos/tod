#include "Tod.h"

#include "XtionGrabber.h"

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>


void cloudViewer(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::visualization::PCLVisualizer* viewer(new pcl::visualization::PCLVisualizer);
    viewer->setBackgroundColor(0,0,0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "cloud");
    viewer->addCoordinateSystem(1.0);
    //viewer->initCameraParameters();
    while(!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}


int main(int argv, char** argc)
{

    PinholeCamera camera(525.0, 525.0, 320.0, 240.0, cv::Size(640, 480));
    ObjectDetector::DetectionParameters params;

    //!< give the the path of the object base
    boost::filesystem::path resultDir("/home/marios/Desktop/project/objects_res");
    boost::filesystem::path pathDir("/home/marios/Desktop/project/objects_base");
    boost::filesystem::path calibDir("/home/marios/Desktop/project/calibration/build/calibration.txt");
/*
    //!< give the name of the object to be detected
    std::string object = "pringles";
    int isCylindrical = 1;
    double thickness = 0.075;
=
    std::string object = "amita";
    int isCylindrical = 0;
    double thickness = 0.075;

    std::string object = "washNgo";
    int isCylindrical = 0;
    double thickness = 0.063;

 

    std::string object = "lucozade";
    int isCylindrical = 1;
    double thickness = 0.06;

    
    std::string object = "depon";
    int isCylindrical = 1;
    double thickness = 0.05;
*/
	 std::string object = "box";
    int isCylindrical = 0;
    double thickness = 0.05;

    //!< read from base
    ObjectDetector detector;
    detector.readFromBase(pathDir.string());
    
    XtionGrabber grabber;
	int flag = 1;
    while(1)
    {
        grabber.grab();
        cv::Mat rgb = grabber.getRgb();
        cv::Mat depth = grabber.getDepth();

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        depthToPointCloud(rgb, depth, cloud);

        std::vector<ObjectDetector::Result> hypotheses;

        if(detector.detect(rgb, cloud, camera, params, hypotheses) == true)
        {
            std::cout<<"Objects Found"<<std::endl;

            Eigen::Matrix4d finalPose;

            if( detector.findSingleObject(calibDir, hypotheses, object, 
                finalPose, isCylindrical) == false ) continue;

            std::ofstream strmPose((resultDir / "pose.txt").string().c_str());
            strmPose << finalPose;
            strmPose.close();

            //!< convert to quarteninon and translation
            Eigen::Matrix3f rmat;
            rmat << finalPose(0,0), finalPose(0,1), finalPose(0,2),
                    finalPose(1,0), finalPose(1,1), finalPose(1,2),
                    finalPose(2,0), finalPose(2,1), finalPose(2,2);

            Eigen::Quaternionf q(rmat);

            Eigen::Vector3d tvec;
            tvec << finalPose(0,3), finalPose(1,3), finalPose(2,3);
            Eigen::Translation3d p(tvec);

            //!< draw in rgb image the bbox of detected object
            detector.draw(rgb, camera, hypotheses);

            cv::imwrite( (resultDir / "result.png").string(), rgb );            

            Eigen::Matrix4d ipose = finalPose.inverse();
            pcl::transformPointCloud(*cloud, *cloud, ipose);

            pcl::io::savePLYFile((resultDir / "cloud.ply").string(), *cloud);
            //cloudViewer(cloud);

            std::ofstream strm( (resultDir / "result.txt").string().c_str() );
            strm << p.x() << ' ' << p.y() << ' ' << p.z() << ' '
                 << q.x() << ' ' << q.y() << ' ' << q.z() << ' '<< q.w()
                 << ' ' << thickness << ' ' << isCylindrical;
            
            strm.close();

			std::cout<<"Continue? (0-> to quit, 1->to continue)"<<std::endl;
			std::cin>>flag;
            if(flag==0)break;

        }
        std::cout << "objects not found" << std::endl;
    }

    return 0;
}
