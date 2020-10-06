#ifndef _TOD_H_
#define _TOD_H_


#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/vtk_lib_io.h>

#include "Eigen/Geometry"

#include "BinaryStream.h"

#include "OffScreenRenderer.h"


extern void getModelBoundingBox(pcl::PolygonMesh mesh, Eigen::Vector3d &vMin, Eigen::Vector3d &vMax);
extern float computeOverlap(std::vector<pcl::PointXYZ> &pts, const Eigen::Matrix4d &pose, const std::pair<Eigen::Vector3d, Eigen::Vector3d> &modelBox);
extern float computeOverlap(const Eigen::Matrix4d &pose1, const std::pair<Eigen::Vector3d, Eigen::Vector3d> &modelBox1,
                     const Eigen::Matrix4d &pose2, const std::pair<Eigen::Vector3d, Eigen::Vector3d> &modelBox2);
extern void depthToPointCloud(const cv::Mat &rgb, const cv::Mat &depth, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);


struct TrainingData;


struct Region;


class ObjectDetector{
	private:	
		std::auto_ptr<TrainingData> data_;
	
	public:	
        struct DetectionParameters
		{
			float minDistanceAbovePlane_;
            float maxDinstaceAbovePlane_;
            float poseVerifyDistThresh_;     
			float poseVerifySiftThresh_;
			int numMostLikelyLabels_;
						
			DetectionParameters()
			{
				minDistanceAbovePlane_ = 0.01;
				maxDinstaceAbovePlane_ = 0.5;
                numMostLikelyLabels_ = 1; //! 2
				poseVerifyDistThresh_ = 0.01;
                poseVerifySiftThresh_ = 150;
            }
            
		};
		
		
		struct Result
		{
			int objId_;
			int nMatches_;
			Eigen::Matrix4d pose_;
			double thickness_;
		};
		
	
        ObjectDetector(){}

        void train(boost::filesystem::path &dir, PinholeCamera &cam);

        void trainSingleObject(boost::filesystem::path &dir, PinholeCamera &cam);
	
		void readFromBase(std::string pathDir);
		
		void computeDescriptors(const cv::Mat &rgb);
		
		void getDescriptorsWithinMask(std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptor,
                                      const cv::Mat &mask, const cv::Rect &rect);
		
		void findPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<Eigen::Vector4d> &planes);
		
		void pointsAbovePlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
							  const Eigen::Vector4d &plane, float tmin, float tmax,
							  pcl::IndicesPtr &indices);
											
		void findClusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                          const pcl::IndicesPtr &indices, std::vector<Region> &regions);
                          
        void findPose(const std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &keypoints,
                      std::vector<cv::Point3f> &kp, PinholeCamera &cam, Eigen::Matrix4d &pose);
                      
        int geometricPoseVerification(cv::Mat &descriptor, std::vector<cv::KeyPoint> keypoints,
                                      PinholeCamera &cam, Eigen::Matrix4d &pose,
                                      const DetectionParameters &params, int idx);
                          
        void matchCluster(const cv::Mat &rgb, const cv::Mat &mask, const cv::Rect &rect,
                          PinholeCamera &cam, const DetectionParameters &params,
                          Result &res);                         
                                  
        bool sceneConsistency(const Result &res, const pcl::PointCloud<pcl::PointXYZ> &cloud,
                              PinholeCamera &cam, const cv::Mat &objMask, const cv::Rect rect);
                              
        void mergeHypotheses(std::vector<Result> &hypotheses);     
		
		bool detect(const cv::Mat &rgb, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                    PinholeCamera &cam, const DetectionParameters &params,
                    std::vector<Result> &hypotheses);

        bool findSingleObject(boost::filesystem::path &calibPath, std::vector<Result> &hypotheses,
                              std::string &object,Eigen::Matrix4d &finalPose, int isCylindrical);
                                        
        void draw(cv::Mat &rgb, PinholeCamera &cam, std::vector<Result> &results);
};


#endif
