#include "OffScreenRenderer.h"


OffScreenRenderer::OffScreenRenderer(PinholeCamera &camera, boost::filesystem::path &modelPath):
                                     camera_(camera), modelPath_(modelPath)
{
    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFilePLY(modelPath_.string(), mesh);
    pcl::VTKUtils::mesh2vtk(mesh, polyData_);  
}


void OffScreenRenderer::initCamera(vtkSmartPointer<vtkCamera> camera)
{
    //!< set virtual camera parameters
    const double zNear = 0.001;//!< near clipping plane
    const double zFar = 100.0;//!< far clipping plane
    camera->SetClippingRange(zNear, zFar);

    double fovy = 2 * atan( (camera_.height()/2) / camera_.fy() );//!< camera view angle
    camera->SetViewAngle( fovy * 180/M_PI );
}


void OffScreenRenderer::transform(vtkSmartPointer<vtkCamera> camera, Eigen::Matrix4d &pose)
{
    //!< approximate the transform of xtion to virtual camera
    //!< change the y,z axes
    //!< |1  0  0  0|
    //!< |0 -1  0  0|
    //!< |0  0 -1  0|
    //!< |0  0  0  1|
    vtkSmartPointer<vtkTransform> scale = vtkSmartPointer<vtkTransform>::New();
    vtkSmartPointer<vtkMatrix4x4> scaleMat = vtkSmartPointer<vtkMatrix4x4>::New();
    scaleMat -> Identity();
    scaleMat -> SetElement(1, 1, -1);
    scaleMat -> SetElement(2, 2, -1);
    scale -> Concatenate(scaleMat);
    scale -> Concatenate(camera->GetViewTransformMatrix());
    camera -> ApplyTransform(scale);

    //!< |        |        ]
    //!< |  R^T   |-R^T * t]
    //!< |        |        ]
    //!< |0  0  0 |     1  ]
    vtkSmartPointer<vtkMatrix4x4> vtkPose = vtkSmartPointer<vtkMatrix4x4>::New();
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            vtkPose -> SetElement(i, j, pose(i,j));
        }
    }

    vtkSmartPointer<vtkTransform> poseTransform = vtkSmartPointer<vtkTransform>::New();
    poseTransform -> Identity();
    poseTransform -> SetMatrix(vtkPose);
    poseTransform -> Inverse();
    camera -> ApplyTransform(poseTransform);
}



void OffScreenRenderer::render(Eigen::Matrix4d &pose)
{
    //!< initiate virtual camera, setting the correct parameters
    vtkCamera_ = vtkSmartPointer<vtkCamera>::New();
    initCamera(vtkCamera_);
    transform(vtkCamera_, pose);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper -> SetInputData(polyData_);
    mapper -> Update();

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor -> SetMapper(mapper);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();

    renderWin_ = vtkSmartPointer<vtkRenderWindow>::New();
    renderWin_ -> SetSize(camera_.width(), camera_.height());
    renderWin_ -> AddRenderer(renderer);

    renderer -> AddActor(actor);
    renderer -> SetActiveCamera(vtkCamera_);
    renderWin_ -> Render();
}


cv::Mat OffScreenRenderer::getRgb()
{
    cv::Mat rgb = cv::Mat::zeros(camera_.height(), camera_.width(), CV_64FC3);

    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter -> SetInput(renderWin_);
    windowToImageFilter -> SetMagnification(1);
    windowToImageFilter -> SetInputBufferTypeToRGB();

    vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
    writer -> SetFileName("screenshot.png");
    writer -> SetInputConnection(windowToImageFilter->GetOutputPort());
    writer -> Write();
    rgb = cv::imread("screenshot.png", CV_LOAD_IMAGE_COLOR);

    return rgb;
}


cv::Mat OffScreenRenderer::getDepth()
{
    cv::Mat depth = cv::Mat::zeros(camera_.height(), camera_.width(), CV_16UC1);

    //!< save depth mat
    float* z = new float[camera_.width() * camera_.height()]();
    renderWin_->GetZbufferData(0, 0, camera_.width() - 1, camera_.height() - 1, z);

    double zNear, zFar;
    vtkCamera_ -> GetClippingRange(zNear, zFar);

    for(int x = 0; x < camera_.width(); x++)
    {
        for(int y = 0; y < camera_.height(); y++)
        {
            float value = z[y * camera_.width() + x];
            if(value != 1.0)
            {
                double depthValue = -1 * (zFar * zNear) / (value * (zFar - zNear) - zFar);//?
                depth.at<int16_t>(y, x) = (int16_t) (depthValue * 1000);
            }
        }
    }
    cv::flip(depth, depth, 0);

    return depth;
}
