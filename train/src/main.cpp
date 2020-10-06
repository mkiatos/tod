#include "Tod.h"

int main(int argc, char** argv)
{
    PinholeCamera cam(525.0, 525.0, 320.0, 240.0, cv::Size(640, 480));
    boost::filesystem::path dir("/home/marios/Desktop/project/objects_base/amita");

    ObjectDetector detector;
    detector.trainSingleObject(dir, cam);

    return 0;
}
