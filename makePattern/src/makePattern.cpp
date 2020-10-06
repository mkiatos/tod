#include <iostream>

#include <fstream>

#include <opencv2/opencv.hpp>

#include "boost/filesystem.hpp"
#include <boost/format.hpp>

#include <apriltag.h>
#include <tag36h11.h>


void makePattern36H11(const std::string &tagFolder, const std::string &outSvg, const cv::Size &boardSize, float tileSize, float tileOffset)
{
    using namespace boost::filesystem ;

    float tx = boardSize.width * (tileSize + tileOffset) ;
    float ty = boardSize.height * (tileSize + tileOffset) ;

    float bs = tileSize/8 ;

    std::ofstream strm(outSvg.c_str()) ;

    strm << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"" ;
    strm << tx*100 << "cm\" height=\"" << ty*100 << "cm\" viewBox=\"0 0 " ;
    strm << tx << ' ' << ty << "\">\n<g fill=\"black\">" ;

    for(int i=0, k=0 ; i<boardSize.height ; i++)
        for(int j=0 ; j<boardSize.width ; j++, k++)
        {
            path dir(tagFolder) ;

            path p = dir / str(boost::format("tag36_11_%05d.png") % k) ;

            if ( !exists(p) ) continue ;

            cv::Mat cc = cv::imread(p.string()) ;

            float x0 = j * (tileSize + tileOffset) ;
            float y0 = i * (tileSize + tileOffset) ;

            strm << "<g>\n" ;
            for( int y=0 ; y<8 ; y++)
                for( int x=0 ; x<8 ; x++)
                {

                    if ( cc.at<cv::Vec3b>(y+1, x+1)[0] == 0 )
                    {
                        float xx = x0 + x * bs ;
                        float yy = y0 + y * bs ;

                        strm << "<rect x=\"" << xx << "\" y=\"" << yy << "\" width=\"" << bs << "\" height=\"" << bs << "\"/>\n" ;
                    }
                }
            strm << "</g>\n" ;
        }

    strm << "</g></svg>" ;

    strm.flush() ;
}


int main(int argc, char** argv)
{
    makePattern36H11("/home/robo/Marios/projects/tod/makePattern/3rdparty/apriltag/tag36h11/",
                     "/home/robo/Marios/projects/tod/makePattern/pattern.svg",
                     cv::Size(2, 2), 0.08, 0.01);

    return 0;
}
