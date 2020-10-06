#include "Capture.h"

int main(int argc, char** argv)
{
  
  boost::filesystem::path objectPath("/home/marios/Vision/code/objects_base/washNgo");
  CapturePipeline::Parameters params;
std::cout<<"Main"<<std::endl;
  CapturePipeline capture(objectPath, params);

  capture.findPoses();
  //capture.mergePointClouds();
  //capture.makeMasks();
  
  return 0;
}
