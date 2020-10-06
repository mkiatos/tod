#include "Tod.h"
#include <time.h>
namespace fs = boost::filesystem;


struct TrainingDataForObject{

    std::vector<cv::Point3f> keypoints_;
    std::vector<cv::Mat> descriptors_;

    std::vector<int> descriptorIndex_;

    boost::shared_ptr<cv::FlannBasedMatcher> siftMatcher_;
    pcl::search::KdTree<pcl::PointXYZ> tree_;

    std::pair<Eigen::Vector3d, Eigen::Vector3d> modelBox_;

    fs::path modelPath_, objectPath_;

};


typedef boost::unordered_multimap<std::pair<int, int>, int> KeypointIndex;


struct TrainingData{

    std::vector<std::string> labels_;
    std::vector<TrainingDataForObject> odata_;
    std::vector<cv::KeyPoint> kp_;
    cv::Mat desc_;
    KeypointIndex kpIndex_;
};


struct Region{
    cv::Mat mask_;
    cv::Rect rect_;
    int regionId_;

    Region(const cv::Size &size)
    {
        mask_ = cv::Mat::zeros(size, CV_8UC1);
    }
};



void ObjectDetector::readFromBase(std::string pathDir)
{
    std::cout<<"reading from base..."<<std::endl;

    data_.reset(new TrainingData);

    boost::filesystem::path directory(pathDir);
    boost::filesystem::directory_iterator iter(directory), end;

    for(; iter != end; ++iter)
    {
        //!< load trainig data and polygon mesh model
        boost::filesystem::path dataPath = iter->path() / "todData.bin" ;
        if ( ! boost::filesystem::exists(dataPath)) continue;

        boost::filesystem::path modelPath = iter->path() / "ply/model.ply" ;
        if (!boost::filesystem::exists(modelPath)) continue;


        std::string label = iter->path().stem().string();

        data_->labels_.push_back(label);
        TrainingDataForObject obj;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::PolygonMesh mesh;
        Eigen::Vector3d vMin, vMax;
        pcl::io::loadPolygonFilePLY(modelPath.string().c_str(), mesh);
        getModelBoundingBox(mesh, vMin, vMax);
        
        //Eigen::Matrix4d transformAxes = Eigen::Matrix4d::Identity();
		//transformAxes(0,3) = (vMax[0] - vMin[0]) / 2;
		//transformAxes(1,3) = (vMax[1] - vMin[1]) / 2;
		//transformAxes(2,3) = (vMax[2] - vMin[2]) / 2;
		//std::ofstream out( (iter->path() / "trasnformAxes.txt").string().c_str() );
		//out << transformAxes;

        obj.modelPath_ = modelPath.string();
        obj.modelBox_ = std::make_pair(vMin, vMax);
        obj.objectPath_ = iter->path();

        std::ifstream strm(dataPath.c_str(), std::ios::binary);
        BinaryStream ar(strm);

        int noOfViews;
        ar >> noOfViews;

        for(int i = 0; i < noOfViews; i++)
        {
            int idx;	
            ar >> idx;
            obj.descriptorIndex_.push_back(idx);

            std::vector<Eigen::Vector4d> kp;
            ar >> kp;
            for(int j = 0; j < kp.size(); j++)
            {
                cloud->points.push_back(pcl::PointXYZ(kp[j].x(), kp[j].y(), kp[j].z()));
                obj.keypoints_.push_back(cv::Point3f(kp[j].x(), kp[j].y(), kp[j].z()));
            }

            std::vector<cv::Mat> desc;
            ar >> desc;
            obj.descriptors_.insert(obj.descriptors_.end(), desc.begin(), desc.end());
        }
        
        obj.siftMatcher_.reset(new cv::FlannBasedMatcher);
        obj.siftMatcher_->add(obj.descriptors_);
        obj.siftMatcher_->train();
        
        obj.tree_.setInputCloud(cloud);

        data_->odata_.push_back(obj);
        std::cout<< label << " loaded" <<std::endl;
    }
}


void ObjectDetector::computeDescriptors(const cv::Mat &rgb)
{
    data_->kp_.clear();
    data_->desc_ = cv::Mat();
    data_->kpIndex_.clear();

    cv::SIFT sift(0, 3,0.04, 10, 1.6);
    sift.detect(rgb, data_->kp_);
    sift.compute(rgb, data_->kp_, data_->desc_);

    for(int i = 0; i < data_->kp_.size(); i++)
    {
        cv::KeyPoint &kp = data_->kp_[i];
        data_->kpIndex_.insert( std::make_pair( std::make_pair(kp.pt.x, kp.pt.y), i ) );
    }
}



void ObjectDetector::getDescriptorsWithinMask(std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptor,
                                              const cv::Mat &mask, const cv::Rect &rect)
{
    std::vector<int> indices;

    for(int y = 0; y < rect.y + rect.height; y++)
    {
        for(int x = 0; x < rect.x + rect.width ; x++)
        {
            if(mask.at<uchar>(y, x) == 255)
            {
                std::pair<KeypointIndex::iterator, KeypointIndex::iterator> itRange;
                itRange = data_->kpIndex_.equal_range(std::make_pair(x, y));

                for(KeypointIndex::iterator it = itRange.first; it != itRange.second; ++it)
                    indices.push_back(it->second);
            }
        }
    }

    descriptor = cv::Mat(indices.size(), data_->desc_.cols, data_->desc_.type());

    for(int i = 0; i < indices.size(); i++)
    {
        keypoints.push_back(data_->kp_[indices[i]]);
        data_->desc_.row(indices[i]).copyTo(descriptor.row(i));
    }

}



void ObjectDetector::findPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<Eigen::Vector4d> &planes)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
    *tempCloud = *cloud;

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setProbability(0.99);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setMaxIterations(500);

    pcl::ExtractIndices<pcl::PointXYZ> extract;

    while(1)
    {
        pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliersPlane (new pcl::PointIndices);

        seg.setInputCloud(tempCloud);
        seg.segment(*inliersPlane, *coeffs);

        if(inliersPlane->indices.size() < 60000) break;

        Eigen::Vector4d cand(coeffs->values[0], coeffs->values[1], coeffs->values[2], coeffs->values[3]);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(tempCloud);
        extract.setIndices(inliersPlane);
        extract.setNegative(true);
        extract.filter(*cloudFiltered);
        *tempCloud = *cloudFiltered;

        planes.push_back(cand);
    }

}



void ObjectDetector::pointsAbovePlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                      const Eigen::Vector4d &plane, float tmin, float tmax,
                                      pcl::IndicesPtr &indices)
{
    for(int i = 0; i < cloud->height; i++)
    {
        for(int j = 0; j < cloud->width; j++)
        {
            pcl::PointXYZ p = cloud->at(j, i);

            Eigen::Vector4d p_(p.x, p.y, p.z, 1.0);

            double dist = p_.dot(plane);

            if(dist > tmin  &&  dist < tmax)
            {
                indices->push_back(i * cloud->width + j);
            }
        }
    }
}



void ObjectDetector::findClusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                  const pcl::IndicesPtr &indices, std::vector<Region> &regions)
{

    //!< Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.01); //!< 1cm
    ec.setMinClusterSize(1000);
    ec.setMaxClusterSize(cloud->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.setIndices(indices);
    ec.extract(cluster_indices);

    //!< set the initial x,y parameters fof rect creation
    int xmin, ymin, xmax, ymax;
    xmin = ymin = 0;
    xmax = ymax = 255;

    int clusterId = 1;

    for(int i = 0; i < cluster_indices.size(); i++)
    {
        Region region( cv::Size(cloud->width, cloud->height) );
        pcl::PointIndices cluster = cluster_indices[i];
        for(int j = 0; j < cluster.indices.size(); j++)
        {
            int id = cluster.indices[j];
            int y = id / cloud->height;
            int x = id % cloud->height;
            region.mask_.at<uchar>(y, x) = 255;

            xmin = std::min(xmin, x);
            ymin = std::min(ymin, y);
            xmax = std::max(xmax, x);
            ymax = std::max(ymax, y);
        }

        region.rect_ = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
        region.regionId_ = clusterId;
        regions.push_back(region);

        clusterId++;
    }

}



void ObjectDetector::findPose(const std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &keypoints,
                              std::vector<cv::Point3f> &kp, PinholeCamera &cam, Eigen::Matrix4d &pose)
{
    std::vector<cv::Point2f> imagePts;
    std::vector<cv::Point3f> objectPts;

    for(int j = 0; j < matches.size(); j++)
    {
        cv::KeyPoint &kpTest = keypoints[matches[j].queryIdx];
        cv::Point3f &kpObject = kp[matches[j].imgIdx];

        imagePts.push_back(kpTest.pt);
        objectPts.push_back(cv::Point3f(kpObject.x, kpObject.y, kpObject.z));
    }

    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    cv::solvePnPRansac(objectPts, imagePts, cam.getMatrix(), cam.getDistortionCoeffs(), rvec, tvec,
                       false, //!< useExtrinsicGuess
                       500,   //!< number of iterations
                       2,     //!< reprojection error
                       100,   //!< number of inliers
                       inliers);

    cv::Mat R = cv::Mat(cv::Size(3,3), CV_64F);
    cv::Rodrigues(rvec, R);

    Eigen::Matrix3d r;
    Eigen::Vector3d t;
    for(int i = 0; i < 3 ; i++)
    {
        for(int j = 0; j < 3 ; j++)
        {
            r(i,j) = R.at<double>(i,j);
        }
        t(i) = tvec.at<double>(0,i);
    }

    Eigen::Affine3d tr = Eigen::Translation3d(t) * r ;

    pose = tr.matrix();
}



int ObjectDetector::geometricPoseVerification(cv::Mat &descriptor, std::vector<cv::KeyPoint> keypoints,
                                              PinholeCamera &cam, Eigen::Matrix4d &pose,
                                              const DetectionParameters &params, int idx)
{
    TrainingDataForObject &obj = data_->odata_[idx];

    OffScreenRenderer rdr(cam, obj.modelPath_);
    rdr.render(pose);
    

    cv::Mat depth = rdr.getDepth();

    Eigen::Matrix4d ipose = pose.inverse();

    //!< We project the 2D locations xmr of all SIFT features fmr
    //!< detected in Im onto the 3D object model
    std::vector<cv::Point3f> projPoints;
    std::vector<int> projId;

    for(int i = 0; i < keypoints.size(); i++)
    {
        cv::KeyPoint &kp = keypoints[i];
        unsigned short depthValue = depth.at<unsigned short>(kp.pt.y, kp.pt.x);
        cv::Point3d bp = cam.backProject(kp.pt, depthValue);

        Eigen::Vector4d q = ipose * Eigen::Vector4d(bp.x, bp.y, bp.z, 1.0);

        projPoints.push_back(cv::Point3f(q.x(), q.y(), q.z()));
        projId.push_back(i);
    }

    std::vector<cv::Point2f> imPts;
    std::vector<cv::Point3f> modelPts;

    for(int i = 0; i < projPoints.size(); i++)
    {
        std::vector<int> kIndices;
        std::vector<float> kSqrDistances;

        int neighbors = obj.tree_.radiusSearch(pcl::PointXYZ(projPoints[i].x, projPoints[i].y, projPoints[i].z),
                                               params.poseVerifyDistThresh_, kIndices, kSqrDistances, 20);

        double minDist = DBL_MAX;
        int bestId = -1;
        for(int j = 0; j < neighbors; j++)
        {
            double dist = cv::norm(descriptor.row(projId[i]) - obj.descriptors_[kIndices[j]]);

            if(dist < minDist)
            {
                minDist = dist;
                bestId = kIndices[j];
            }
        }

        if(minDist > params.poseVerifySiftThresh_) continue;
        imPts.push_back(keypoints[projId[i]].pt);
        modelPts.push_back(obj.keypoints_[bestId]);
    }

    if(imPts.size() > 3)
    {
        cv::Mat rmat = cv::Mat(3, 3, CV_64F);
        cv::Mat tmat = cv::Mat(1, 3, CV_64F);
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                rmat.at<double>(i, j) = pose(i, j);
            }
            tmat.at<double>(0, i) = pose(0, i);
        }

        cv::Mat rvec;
        cv::Rodrigues(rmat, rvec);

        std::vector<int> inliers;
        std::cout<<imPts.size()<<std::endl;
        cv::solvePnPRansac(modelPts, imPts, cam.getMatrix(), cam.getDistortionCoeffs(), rvec, tmat,
                           false, //!< useExtrinsicGuess
                           500,   //!< number of iterations
                           2,     //!< reprojection error
                           100,   //!< number of inliers
                           inliers);
		
		std::ofstream out("points.txt");
		for(int i = 0; i < imPts.size(); i++)
		{
			out << imPts << " " ;
		}
	    
	    out.close();
		
        cv::Rodrigues(rvec, rmat);

        Eigen::Matrix3d r;
        Eigen::Vector3d t;
        for(int i = 0; i < 3 ; i++)
        {
            for(int j = 0; j < 3 ; j++)
            {
                r(i,j) = rmat.at<double>(i,j);
            }
            t(i) = tmat.at<double>(0,i);
        }

        Eigen::Affine3d tr = Eigen::Translation3d(t) * r ;

        pose = tr.matrix();

    }
    return imPts.size();
}



void ObjectDetector::matchCluster(const cv::Mat &rgb, const cv::Mat &mask, const cv::Rect &rect,
                                  PinholeCamera &cam, const DetectionParameters &params,
                                  Result &res)
{
    res.objId_ = -1;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    getDescriptorsWithinMask(keypoints, descriptor, mask, rect);

    if (keypoints.empty()) return;

    std::vector<double> logP;//!< the log likelihood of each object
    std::vector<int> logIndex;

    for(int i = 0; i < data_->labels_.size(); i++)
    {
        TrainingDataForObject &obj = data_->odata_[i];

        //!< find the nearest neighbors
        std::vector<cv::DMatch> matches;
        obj.siftMatcher_->match(descriptor, matches);

        double dist = 0.0;
        for(int j = 0; j < matches.size(); j++)
        {
            double d = cv::norm(descriptor.row(j) - obj.descriptors_[matches[j].imgIdx]);
            dist += d;
        }
        dist /= matches.size();

        logP.push_back(dist);
        logIndex.push_back(i);
    }


    //!< sort and find min
    for(int i = 0; i < logIndex.size() - 1; i++)
    {
        for(int j = 0; j < logIndex.size()- i - 1; j++)
        {
            if (logP[j+1] < logP[j])
            {
                std::swap(logP[j], logP[j+1]);
                std::swap(logIndex[j], logIndex[j+1]);
            }
        }
    }


    int nSift = std::min(params.numMostLikelyLabels_, (int)logP.size());


    int maxMatches = -1;
    Eigen::Matrix4d bestPose;
    int bestObj;
	double thickness;
	
    //!< find pose
    for(int i = 0; i < nSift; i++)
    {
        int idx = logIndex[i];

        TrainingDataForObject &obj = data_->odata_[idx];

        std::vector<cv::DMatch> matches;
        obj.siftMatcher_->match(descriptor, matches);

        Eigen::Matrix4d pose;
        findPose(matches, keypoints, obj.keypoints_, cam, pose);

        int nMatches = geometricPoseVerification(descriptor, keypoints, cam, pose, params, idx);

        if(nMatches > maxMatches)
        {
            maxMatches = nMatches;
            bestPose = pose;
            bestObj = idx;
            thickness = std::abs(obj.modelBox_.first.x() - obj.modelBox_.second.x());
        }
    }

    if(maxMatches == 0)
    {
        res.objId_ = -1;
    }
    else
    {
        //store results
        res.objId_ = bestObj;
        res.nMatches_ = maxMatches;
        res.pose_ = bestPose;
        res.thickness_ = thickness;
    }
}



bool ObjectDetector::sceneConsistency(const Result &res, const pcl::PointCloud<pcl::PointXYZ> &cloud,
                                      PinholeCamera &cam, const cv::Mat &objMask, const cv::Rect rect)
{
    TrainingDataForObject &obj = data_->odata_[res.objId_];

    std::vector<pcl::PointXYZ> points;

    for(int y = rect.y; y < rect.y + rect.height; y++)
    {
        for(int x = rect.x; x < rect.x + rect.width; x++)
        {
            if(objMask.at<uchar>(y, x) == 255)
            {
                int id = y * cloud.width + x;
                points.push_back(pcl::PointXYZ(cloud.points[id].x, cloud.points[id].y, cloud.points[id].z));
            }
        }
    }

    float percentage = computeOverlap(points, res.pose_, obj.modelBox_);

    if(percentage > 0.3) return true;
    else return false;
}



void ObjectDetector::mergeHypotheses(std::vector<Result> &hypotheses)
{
    std::vector<Result>::iterator it(hypotheses.begin());

    for( ; it != hypotheses.end(); ++it)
    {
        Result res1 = *it;

        std::vector<Result>::iterator rit(it);
        ++rit;
        for( ; rit != hypotheses.end(); )
        {
            if (rit == it) break ;

            Result res2 = *rit;

            if(res1.objId_ == res2.objId_)
            {
                float overlap = computeOverlap(res1.pose_, data_->odata_[res1.objId_].modelBox_,
                        res2.pose_, data_->odata_[res2.objId_].modelBox_);

                if(overlap > 0.2)
                {
                    if(res1.nMatches_ < res2.nMatches_) std::swap(res1, res2);

                    rit = hypotheses.erase(rit);

                    if (rit == hypotheses.end()) break ;
                }
                else ++rit;
            }
            else ++rit;
        }
    }
}



bool ObjectDetector::detect(const cv::Mat &rgb, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                            PinholeCamera &cam, const DetectionParameters &params,
                            std::vector<Result> &hypotheses)
{
    clock_t tStart = clock();
    computeDescriptors(rgb);


    //!< find planes
    std::vector<Eigen::Vector4d> planes;
    findPlanes(cloud, planes);
	
    for(int i = 0; i < 1; i++)
    {
        //!< find what part of the point cloud lie above each candidate plane
        pcl::IndicesPtr indices(new std::vector <int>);
        pointsAbovePlane(cloud, planes[i], params.minDistanceAbovePlane_, params.maxDinstaceAbovePlane_, indices);

        //!< find clusters on the remaining point cloud to obtain individual object point clouds
        std::vector<Region> regions;
        findClusters(cloud, indices, regions);

        for(int j = 0; j < regions.size(); j++)
        {
            Result res;

            matchCluster(rgb, regions[j].mask_, regions[j].rect_, cam, params, res);

            if (res.objId_ >= 0)
            {
                if(sceneConsistency(res, *cloud, cam, regions[j].mask_, regions[j].rect_) == false) continue;
                hypotheses.push_back(res);

                pcl::PointCloud<pcl::PointXYZ>::Ptr objCloud (new pcl::PointCloud<pcl::PointXYZ>);
                for(int m = 0; m < regions[j].mask_.rows; m++)
                {
                    for(int n = 0; n < regions[j].mask_.cols; n++)
                    {
                        if(regions[j].mask_.at<uchar>(m,n) == 255)
                        {   int id = m*cloud->width+n;

                            objCloud->points.push_back(pcl::PointXYZ(cloud->points[id].x, cloud->points[id].y, cloud->points[id].z));
                        }
                    }
                }
                pcl::io::savePLYFile("obj.ply", *objCloud);

            }
        }
    }

    mergeHypotheses(hypotheses);

    std::cout<< "time elapsed " << 1000*(clock() - tStart)/CLOCKS_PER_SEC << "ms" << std::endl;


    for(int i = 0; i < hypotheses.size(); i++)
    {
        Result &res = hypotheses[i];
        int objId = res.objId_;
        std::string label = data_->labels_[objId];
        std::cout<<label<<std::endl;
    }

    if(hypotheses.size() > 0) return true;
    else return false;
}


bool ObjectDetector::findSingleObject(boost::filesystem::path &calibPath, std::vector<Result> &hypotheses,
                                      std::string &object,Eigen::Matrix4d &finalPose, int isCylindrical)
{
    //!< load calibration matrix
    std::ifstream strm(calibPath.c_str());

    Eigen::Matrix4d calibMat;

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            strm >> calibMat(i,j);
        }
    }

    //!<!!!!
    calibMat = Eigen::Matrix4d::Identity();
    
    Eigen::Matrix4d rotationMatrix = Eigen::Matrix4d::Zero();
    rotationMatrix(0,1) = 1;
    rotationMatrix(1,0) = -1;
    rotationMatrix(2,2) = 1;
    rotationMatrix(3,3) = 1;

    for(int i = 0; i < hypotheses.size(); i++)
    {
        int objId = hypotheses[i].objId_;
        TrainingDataForObject obj = data_->odata_[objId];
        std::string label = data_->labels_[objId];
        std::cout<<label<<std::endl;

        if(label.c_str() == object)
        {
            //!< load transformation axes matrix
            std::ifstream strm( (obj.objectPath_ / "trasnformAxes.txt").string().c_str() );

            Eigen::Matrix4d transformAxes;

            for(int x = 0; x < 4; x++)
            {
                for(int y = 0; y < 4; y++)
                {
                    strm >> transformAxes(x,y);
                }
            }

            //!< calculate the final pose
            //!< finalPose = calibrationMatrix * objectPose * axesTransfotmation;
            finalPose = calibMat * hypotheses[i].pose_ * transformAxes * rotationMatrix;

            return true;
        }
    }

    return false;
}



void ObjectDetector::draw(cv::Mat &rgb, PinholeCamera &cam, std::vector<ObjectDetector::Result> &results)
{
    for(int i = 0; i < results.size(); i++)
    {
        Result &res = results[i];

        int objId = res.objId_;
        TrainingDataForObject obj = data_->odata_[objId];
        std::string label = data_->labels_[objId];

        OffScreenRenderer rdr(cam, obj.modelPath_);
        rdr.render(res.pose_);

        cv::Mat rdrRgb = rdr.getRgb();
        cv::Mat gray;
        cvtColor(rdrRgb, gray, CV_BGR2GRAY);

        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

        for(int j = 0; j < contours.size(); j++)
        {
            cv::drawContours(rgb, contours, j, cv::Scalar(255, 128, 0), 2, 8, hierarchy, 0, cv::Point());
        }

        cv::Rect r = cv::boundingRect(contours[0]);

        cv::putText(rgb, label, cv::Point(r.tl()), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2) ;
    }
}
