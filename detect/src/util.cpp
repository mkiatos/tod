#include <pcl/io/vtk_lib_io.h>
#include <pcl/ros/conversions.h>
#include <pcl/point_types.h>

#include "opencv2/highgui/highgui.hpp"

#include "boost/random.hpp"

void getModelBoundingBox(pcl::PolygonMesh mesh, Eigen::Vector3d &vMin, Eigen::Vector3d &vMax)
{
    vMax.x() = vMax.y() = vMax.z() = -DBL_MAX;
    vMin.x() = vMin.y() = vMin.z() = DBL_MAX;
	
	//!< convert polygonMesh to pointCloud
	pcl::PointCloud<pcl::PointXYZ> cloud;
    //pcl::fromROSMsg(mesh.cloud, cloud);
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);
        
	for(int i = 0; i < mesh.polygons.size(); i++)
	{
		pcl::Vertices polygon = mesh.polygons[i];
		std::vector<uint32_t> vertex = polygon.vertices;

		for(int j = 0; j < vertex.size(); j++)
		{
			pcl::PointXYZ &p  = cloud.points[mesh.polygons[i].vertices[j]];
            vMax.x() = std::max( vMax.x(), double(p.x) );
            vMax.y() = std::max( vMax.y(), double(p.y) );
            vMax.z() = std::max( vMax.z(), double(p.z) );

            vMin.x() = std::min( vMin.x(), double(p.x) );
            vMin.y() = std::min( vMin.y(), double(p.y) );
            vMin.z() = std::min( vMin.z(), double(p.z) );
		}
	}
	
}



float computeOverlap(std::vector<pcl::PointXYZ> &pts, const Eigen::Matrix4d &pose, const std::pair<Eigen::Vector3d, Eigen::Vector3d> &modelBox)
{
    float count = 0;

    Eigen::Matrix4d ipose = pose.inverse();

    for(int i = 0; i < pts.size(); i++)
	{
        double x = pts[i].x;
        double y = pts[i].y;
        double z = pts[i].z;

        Eigen::Vector4d q = ipose * Eigen::Vector4d(x, y, z, 1);

        if(modelBox.first.x() > q.x() && modelBox.second.x() < q.x()) continue;
        if(modelBox.first.y() > q.y() && modelBox.second.y() < q.y()) continue;
        if(modelBox.first.z() > q.z() && modelBox.second.z() < q.z()) continue;

        count++;
	}

    return count/100.0;
}



float computeOverlap(const Eigen::Matrix4d &pose1, const std::pair<Eigen::Vector3d, Eigen::Vector3d> &modelBox1,
                     const Eigen::Matrix4d &pose2, const std::pair<Eigen::Vector3d, Eigen::Vector3d> &modelBox2)
{


    boost::mt19937 rng;

    boost::random::uniform_real_distribution<double> rnx(modelBox1.first.x(), modelBox1.second.x());
    boost::random::uniform_real_distribution<double> rny(modelBox1.first.y(), modelBox1.second.y());
    boost::random::uniform_real_distribution<double> rnz(modelBox1.first.z(), modelBox1.second.z());

    Eigen::Matrix4d a = pose1.inverse() * pose2;

    float count = 0;

    for(int i = 0; i < 100; i++)
    {
        double x = rnx(rng);
        double y = rny(rng);
        double z = rnz(rng);

        Eigen::Vector4d q = a * Eigen::Vector4d(x, y, z, 1);

        if(modelBox2.first.x() > q.x() && modelBox2.second.x() < q.x()) continue;
        if(modelBox2.first.y() > q.y() && modelBox2.second.y() < q.y()) continue;
        if(modelBox2.first.z() > q.z() && modelBox2.second.z() < q.z()) continue;

        count++;
    }

    return count/100.0;
}



void depthToPointCloud(const cv::Mat &rgb, const cv::Mat &depth, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    int focalLengthX = 525;
    int focalLengthY = 525;
    double centerX = 320;
    double centerY = 240;

    cloud->width = rgb.cols;
    cloud->height = rgb.rows;
    cloud->points.resize (cloud->width * cloud->height);

    unsigned short depth_value;
    float x,y,z;
    for(int i=0;i<depth.rows;i++)
    {
        for(int j=0;j<depth.cols;j++)
        {
            depth_value = depth.at<unsigned short>(i,j);
            if(depth_value != 0)
            {
                //!< Z = depth(row, col) / 1000;
                //!< X = (col - centerX) * Z / focalLengthX;
                //!< Y = (row - centerY) * Z / focalLengthY;
                z = float(depth_value) / 1000.0f;
                x = (j - centerX) * z / focalLengthX;
                y = (i - centerY) * z / focalLengthY;

                cloud->points[i*rgb.rows+j].x = x;
                cloud->points[i*rgb.rows+j].y = y;
                cloud->points[i*rgb.rows+j].z = z;

                //!< save rgb data
                //uint8_t b = rgb.at<cv::Vec3b>(i,j)[0];
                //uint8_t g = rgb.at<cv::Vec3b>(i,j)[1];
                //uint8_t r = rgb.at<cv::Vec3b>(i,j)[2];
                //cloud->points[i*rgb.rows+j].rgb = (r << 16 | g << 8 | b);
            }
        }
    }
}
