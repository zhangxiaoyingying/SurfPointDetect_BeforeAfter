
#include "stdafx.h"
/*
#include <iostream>

#include <iomanip>

#include "opencv2/core/core.hpp"

#include "opencv2/objdetect/objdetect.hpp"

#include "opencv2/features2d/features2d.hpp"

#include "opencv2/highgui/highgui.hpp"

#include "opencv2/calib3d/calib3d.hpp"

#include "opencv2/nonfree/nonfree.hpp"

#include "opencv2/imgproc/imgproc_c.h"

#include "opencv2/legacy/legacy.hpp"

#include "opencv2/legacy/compat.hpp"

 

using namespace cv;

using namespace std;

 

int main()

{

    Mat leftImg=imread("E:\\test\\111.png");

    Mat rightImg=imread("E:\\test\\222.png");

    if(leftImg.data==NULL||rightImg.data==NULL)

        return 0;

 

    //转化成灰度图

    Mat leftGray;

    Mat rightGray;

    cvtColor(leftImg,leftGray,CV_BGR2GRAY);

    cvtColor(rightImg,rightGray,CV_BGR2GRAY);

 

    //获取两幅图像的共同特征点

    int minHessian=400;

    SurfFeatureDetector detector(minHessian);

    vector<KeyPoint> leftKeyPoints,rightKeyPoints;

    detector.detect(leftGray,leftKeyPoints);

    detector.detect(rightGray,rightKeyPoints);

    SurfDescriptorExtractor extractor;

    Mat leftDescriptor,rightDescriptor;

    extractor.compute(leftGray,leftKeyPoints,leftDescriptor);

    extractor.compute(rightGray,rightKeyPoints,rightDescriptor);

    FlannBasedMatcher matcher;

    vector<DMatch> matches;

    matcher.match(leftDescriptor,rightDescriptor,matches);    

    int matchCount=leftDescriptor.rows;

    if(matchCount>15)

    {

        matchCount=15;

        //sort(matches.begin(),matches.begin()+leftDescriptor.rows,DistanceLessThan);

        sort(matches.begin(),matches.begin()+leftDescriptor.rows);

    }    

    vector<Point2f> leftPoints;

    vector<Point2f> rightPoints;

    for(int i=0; i<matchCount; i++)

    {

        leftPoints.push_back(leftKeyPoints[matches[i].queryIdx].pt);

        rightPoints.push_back(rightKeyPoints[matches[i].trainIdx].pt);

    }

 

    //获取左边图像到右边图像的投影映射关系

    Mat homo=findHomography(leftPoints,rightPoints);

    Mat shftMat=(Mat_<double>(3,3)<<1.0,0,leftImg.cols, 0,1.0,0, 0,0,1.0);

 

    //拼接图像

    Mat tiledImg;

    warpPerspective(leftImg,tiledImg,shftMat*homo,Size(leftImg.cols+rightImg.cols,rightImg.rows));

    rightImg.copyTo(Mat(tiledImg,Rect(leftImg.cols,0,rightImg.cols,rightImg.rows)));

 

    //保存图像

    imwrite("tiled.jpg",tiledImg);

        

    //显示拼接的图像

    imshow("tiled image",tiledImg);

    waitKey(3000);
	system("pause");
    return 0;

}

*/


//surf 拼接图像
/*
int main()
{
//读取图像
Mat leftImg=imread("E:\\test\\111.png");
Mat rightImg=imread("E:\\test\\Lena1.jpg");
if(leftImg.data==NULL||rightImg.data==NULL)
	return -1;

//转化成灰度图
Mat leftGray;
Mat rightGray;
cvtColor(leftImg,leftGray,CV_BGR2GRAY);
cvtColor(rightImg,rightGray,CV_BGR2GRAY);

//获取两幅图像的共同特征点
int minHessian=400;
SurfFeatureDetector detector(minHessian);
vector<KeyPoint> leftKeyPoints,rightKeyPoints;
detector.detect(leftGray,leftKeyPoints);
detector.detect(rightGray,rightKeyPoints);
SurfDescriptorExtractor extractor;
Mat leftDescriptor,rightDescriptor;
extractor.compute(leftGray,leftKeyPoints,leftDescriptor);
extractor.compute(rightGray,rightKeyPoints,rightDescriptor);
FlannBasedMatcher matcher;
vector<DMatch> matches;
matcher.match(leftDescriptor,rightDescriptor,matches);	
int matchCount=leftDescriptor.rows;
if(matchCount>15)
{
	matchCount=15;
	//sort(matches.begin(),matches.begin()+leftDescriptor.rows,DistanceLessThan);
	sort(matches.begin(),matches.begin()+leftDescriptor.rows);
}	
vector<Point2f> leftPoints;
vector<Point2f> rightPoints;
for(int i=0; i<matchCount; i++)
{
	leftPoints.push_back(leftKeyPoints[matches[i].queryIdx].pt);
	rightPoints.push_back(rightKeyPoints[matches[i].trainIdx].pt);
}

//获取左边图像到右边图像的投影映射关系
Mat homo=findHomography(leftPoints,rightPoints);
Mat shftMat=(Mat_<double>(3,3)<<1.0,0,leftImg.cols, 0,1.0,0, 0,0,1.0);

//拼接图像
Mat tiledImg;
warpPerspective(leftImg,tiledImg,shftMat*homo,Size(leftImg.cols+rightImg.cols,rightImg.rows));
rightImg.copyTo(Mat(tiledImg,Rect(leftImg.cols,0,rightImg.cols,rightImg.rows)));

//保存图像
imwrite("tiled.jpg",tiledImg);
	
//显示拼接的图像
imshow("tiled image",tiledImg);
waitKey(0);

return 0;
}

*/

/*
int _tmain(int argc, char * argv[])
{
    Mat img1 = imread("E:\\test\\111.png");
    Mat img2 = imread("E:\\test\\222.png");
  
    if (img1.empty() || img2.empty())
    {
        cout << "Can't read image"<< endl;
        return -1;
    }
    imgs.push_back(img1);
    imgs.push_back(img2);
  
    Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
  //   使用stitch函数进行拼接
    Mat pano;
    Stitcher::Status status = stitcher.stitch(imgs, pano);
    imwrite(result_name, pano);
    Mat pano2=pano.clone();
  //   显示源图像，和结果图像
    imshow("全景图像", pano);
    if(waitKey()==27)
        return 0;
}
*/