#ifndef _OFF__SCREEN_RENDERER_
#define _OFF__SCREEN_RENDERER_


#include <iostream>

#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkCamera.h>

#include "opencv2/highgui/highgui.hpp"

#include "boost/filesystem.hpp"

#include "PinholeCamera.h"


class OffScreenRenderer{
	private:		      
        boost::filesystem::path modelPath_;
        PinholeCamera camera_;
        vtkSmartPointer<vtkCamera> vtkCamera_;
		vtkSmartPointer<vtkPolyData>  polyData_;	
		vtkSmartPointer<vtkRenderWindow> renderWin_;	
	public:
        OffScreenRenderer(PinholeCamera &camera, boost::filesystem::path &modelPath_);
		
		void initCamera(vtkSmartPointer<vtkCamera> camera);
		
		void transform(vtkSmartPointer<vtkCamera> camera, Eigen::Matrix4d &pose);
		
		void saveRendering(const vtkSmartPointer<vtkRenderWindow> &renderWin,
                           const vtkSmartPointer<vtkRenderer> &renderer,
                           const vtkSmartPointer<vtkCamera> &cam);
                                  
        void render(Eigen::Matrix4d &pose);
        
        cv::Mat getRgb();
        
        cv::Mat getDepth();
};


#endif
