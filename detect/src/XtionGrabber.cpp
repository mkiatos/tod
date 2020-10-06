#include "XtionGrabber.h"

void XtionGrabber::grab()
{
	capture_.open(CV_CAP_OPENNI);
	capture_.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ);
	
	if(!capture_.grab())
	{
		std::cout<<"xtion cannot grab images"<<std::endl;
	}
	else if (capture_.retrieve(depth_, CV_CAP_OPENNI_DEPTH_MAP) & capture_.retrieve(rgb_, CV_CAP_OPENNI_BGR_IMAGE))
	{	
        //cv::imshow("rgb", rgb_);
	}
}


void XtionGrabber::init()
{ 
	if (!capture_.isOpened()) 
		std::cout<<"Xtion is not opened"<<std::endl;
	else 
		std::cout<<"Xtion is ready for use!"<<std::endl;
  
	while(1)
	{
		grab();
		if (cv::waitKey(30) == 27) break; ///press Esc to terminate the procedure 
	}
}


cv::Mat XtionGrabber::getRgb()
{
	return rgb_;
}


cv::Mat XtionGrabber::getDepth()
{
	return depth_;
}
