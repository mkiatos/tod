#include <iostream>

#include "opencv2/highgui/highgui.hpp"

class XtionGrabber{
public:

	XtionGrabber() {}
	
	void grab();
	
	void init();
	
	cv::Mat getRgb();
	
	cv::Mat getDepth();
	
private:
	cv::VideoCapture capture_;
	cv::Mat rgb_, depth_;
};
