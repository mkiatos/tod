#include <calibrator.h>

//opencv
#include "opencv2/opencv.hpp"

int main(){

    cv::VideoCapture capture(CV_CAP_OPENNI);
    if(!capture.isOpened()){
        std::cout << "Cannot open capture object" << std::endl;
        return -1;
    }

    cv::Mat rgb, depth;

    capture.set( CV_CAP_PROP_OPENNI_REGISTRATION, CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION_ON );

    while(true){
        capture.grab();
        capture.retrieve( depth, CV_CAP_OPENNI_DEPTH_MAP );
        capture.retrieve( rgb, CV_CAP_OPENNI_BGR_IMAGE );
        cv::imshow("rgb", rgb);
        if(cv::waitKey(30) > 0)
            break;
    }

    Calibrator calib;
    calib.calibrate(rgb, depth, "input_points.txt", "calibration.txt");

    return 0;

}
