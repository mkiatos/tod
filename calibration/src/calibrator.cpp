#include <calibrator.h>
#include <vector>
#include <fstream>

//opencv
#include <opencv/highgui.h>

struct Comparator {
    bool operator()(const AprilTags::TagDetection &d1, const AprilTags::TagDetection &d2){
        return d1.id < d2.id;
    }
}comparator;



Eigen::Matrix4f Calibrator::doCalibration(const std::vector<Eigen::Vector3f> &robot_points, const std::vector<Eigen::Vector3f> &tag_points)
{
    cv::Mat meanRobot = cv::Mat(3, 1, CV_32FC1, cv::Scalar::all(0));

    int n = robot_points.size();

    for (int i=0;i<n;i++){
        for (int j=0;j<3;j++){
            meanRobot.at<float>(j, 0) += (robot_points[i](j)/(float)n);
        }
    }    

    cv::Mat xtionPoints(n, 3, CV_32FC1, cv::Scalar::all(0));

    cv::Mat meanXtion = cv::Mat(3, 1, CV_32FC1, cv::Scalar::all(0));

    for (unsigned int i=0;i<n;i++)
    {

        xtionPoints.at<float>(i, 0) = tag_points[i](0) ;
        xtionPoints.at<float>(i, 1) = tag_points[i](1) ;
        xtionPoints.at<float>(i, 2) = tag_points[i](2) ;

        meanXtion.at<float>(0, 0) += tag_points[i](0)/n;
        meanXtion.at<float>(1, 0) += tag_points[i](1)/n;
        meanXtion.at<float>(2, 0) += tag_points[i](2)/n;

    }   


    cv::Mat H = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Pa = cv::Mat(3, 1, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Pb = cv::Mat(1, 3, CV_32FC1, cv::Scalar::all(0));

    for(int i=0; i<n; ++i){
        for(int j=0; j<3; ++j){
            Pa.at<float>(j, 0) = xtionPoints.at<float>(i, j) - meanXtion.at<float>(j, 0);
            Pb.at<float>(0, j) = robot_points[i](j) - meanRobot.at<float>(j, 0);
        }
        H += Pa * Pb;
    }
    std::cout << "H: " << H << std::endl;

    cv::SVD svd(H, cv::SVD::FULL_UV);
    cv::Mat tr(4, 4, CV_32FC1, cv::Scalar::all(0)) ;
    cv::Mat V = svd.vt.t();
    double det = cv::determinant(V);

    if(det < 0){
        for(int i=0; i<V.rows; ++i)
            V.at<float>(i,3) *= -1;
    }

    cv::Mat R = V * svd.u.t();
    std::cout << "R: " << R << std::endl;

    cv::Mat t = (-1)*R*meanXtion + meanRobot;

    Eigen::Matrix4f res;
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
            res(i, j) = R.at<float>(i, j);
    for(int i=0; i<3; ++i){
        res(3, i) = 0;
        res(i, 3) = t.at<float>(i);
    }
    res(3,3) = 1;

    return res;

}






void Calibrator::calibrate(const cv::Mat &rgb, const cv::Mat &depth, const std::string &calib_file, const std::string &output_file){


    AprilTags::TagDetector *tagDetector = new AprilTags::TagDetector(AprilTags::tagCodes36h11);

    cv::Mat image_gray;
    cv::cvtColor(rgb, image_gray, CV_BGR2GRAY);
    vector<AprilTags::TagDetection> detections = tagDetector->extractTags(image_gray);

    cv::Mat rgb_tags;
    rgb.copyTo(rgb_tags);
    if(detections.size() > 0){
        for(int i=0; i<detections.size(); ++i){
            for(int j=0; j<4; ++j)
                cv::circle(rgb_tags, cv::Point(detections[i].p[j].first, detections[i].p[j].second), 4, cv::Scalar(0,0,255), 2);
            detections[i].draw(rgb_tags);
        }
    }

    if(detections.size() < nmarkers_) {
        std::cout << "Only " << detections.size() << " of " << nmarkers_ << " markers detected" << std::endl;
        return;
    }

    std::cout << "Press s to stop or any key to continue" << std::endl;
    while(true){
		
        cv::imshow("tags", rgb_tags);
        cv::imwrite("tags.png", rgb_tags);
        cv::imwrite("rgb.png", rgb);
        int k = cv::waitKey(30);
        if( (char)k == 's')
            return;

        if( k > 0)
            break;
    }
    std::sort(detections.begin(), detections.end(), comparator);

    std::ifstream fin(calib_file.c_str());
    if(!fin){
        std::cout << "Cannot find file: " << calib_file << std::endl;
        return;
    }

    std::vector<Eigen::Vector3f> robot_points(nmarkers_ * 4);
    //we read 16 points, 4 for each marker
    for(int i=0; i<nmarkers_ * 4; ++i){
        for(int j=0; j<3; ++j){
            float p;
            fin >> p;
            robot_points[i](j) = p;
        }
    }
    fin.close();

    std::vector<Eigen::Vector3f> tag_points(nmarkers_ * 4);
    for(int i=0; i<nmarkers_; ++i){
//        if(detections[i].id != i){
//            std::cout << "Wrong id recognized..." << std::endl;
//            return;
//        }
        for(int j=0; j<4; ++j){
            int imgx = detections[i].p[j].first;
            int imgy = detections[i].p[j].second;
            float Z = (float)depth.at<unsigned short>(imgy, imgx) / 1000.0f;
            float X = (imgx - cx_) / fx_ * Z;
            float Y = (imgy - cy_) / fy_ * Z;
            tag_points[i*4 + j] << X, Y, Z;
        }
    }


    Eigen::Matrix4f trans = doCalibration(robot_points, tag_points);
    if(trans(2,3) < 0)
        for(int i=0; i<4; ++i)
            trans(2, i) *= -1;

    std::ofstream fout(output_file.c_str());
    fout << trans;
    fout.close();

    std::cout << "Calibration finished succefully" << std::endl;


}
