#include "RgbdUtil.h"

cv::Mat maxFilter(const cv::Mat &depth)
{
    int w = depth.cols, h = depth.rows ;

    cv::Mat_<ushort> dmat(depth), res = cv::Mat_<ushort>::zeros(h, w) ;

    for(int i=1 ; i<h-1 ; i++)
        for(int j=1 ; j<w-1 ; j++)
        {
            ushort p[9] ;

            p[0] = dmat[i-1][j-1] ; p[1] = dmat[i-1][j] ; p[2] = dmat[i-1][j+1] ;
            p[3] = dmat[i][j-1] ; p[4] = dmat[i][j] ; p[5] = dmat[i][j+1] ;
            p[6] = dmat[i+1][j-1] ; p[7] = dmat[i+1][j] ; p[8] = dmat[i+1][j+1] ;

            ushort pmin = p[0], pmax = p[0], pmid = p[4] ;

            for( int k=1 ; k<9 ; k++ )
            {
                pmin = std::min(pmin, p[k]) ;
                pmax = std::max(pmax, p[k]) ;
            }

            if ( std::max(abs(pmax - pmid), abs(pmin - pmid)) < 15 )
                res[i][j] = dmat[i][j] ;
            else res[i][j] = 0 ;

        }

    return res ;
}


void depthToPointCloud(const cv::Mat &rgb, const cv::Mat &depth, const PinholeCamera &camera,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
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
                x = (j - camera.cx()) * z / camera.fx();
                y = (i - camera.cy()) * z / camera.fy();

                cloud->points[i*rgb.rows+j].x = x;
                cloud->points[i*rgb.rows+j].y = y;
                cloud->points[i*rgb.rows+j].z = z;

                //!< save rgb data
                uint8_t b = rgb.at<cv::Vec3b>(i,j)[0];
                uint8_t g = rgb.at<cv::Vec3b>(i,j)[1];
                uint8_t r = rgb.at<cv::Vec3b>(i,j)[2];
                cloud->points[i*rgb.rows+j].rgb = (r << 16 | g << 8 | b);
            }
        }
    }
}


void cloudViewer(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::visualization::PCLVisualizer* viewer(new pcl::visualization::PCLVisualizer);
    viewer->setBackgroundColor(0,0,0);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, "cloud");
    viewer->addCoordinateSystem(1.0);
    //viewer->initCameraParameters();
    while(!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}
