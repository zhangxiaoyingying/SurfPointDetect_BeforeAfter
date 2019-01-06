////// sift.cpp : 定义控制台应用程序的入口点。

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"//因为在属性中已经配置了opencv等目录，所以把其当成了本地目录一样
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp" //在opencv中使用sift算子需要加头文件"opencv2/nonfree/nonfree.hpp"，
									  //注意这个是非免费的，Sift算法的专利权属于哥伦比亚大学，如果在商业软件中使用，可能有风险。
#include "opencv2/legacy/legacy.hpp"


#include "opencv2/imgproc/imgproc.hpp"



using namespace cv;
using namespace std;
Mat di;


////
//////小测试
//////int main()
//////{
//////	IplImage* img=cvLoadImage("E:\\test\\1.tif");//图像大小为940M，tif，打开出错!!!!
//////	//IplImage* img=cvLoadImage("E:\\test\\zhang.tif");//图像大小为130K，格式为tif，可以直接打开
//////	//IplImage* img=cvLoadImage("E:\\fusiontest\\2.bmp");//图像大小为769K，格式为bmp,可以直接打开
//////	cvNamedWindow("Example1",1);
//////	cvShowImage("Example1",img);
//////	cvWaitKey();
//////	cvReleaseImage(&img);
//////	cvDestroyWindow("Example1");
//////}
////
////
/////*
////////Harris角点检测算法
////int main(int argc,char** argv)
////{
////     IplImage* pImg;
////     IplImage* pHarrisImg;
////     IplImage* grayImage;
////     IplImage* dst8;
////    double minVal=0.0, maxVal=0.0;
////     double scale, shift;
////     double min=0, max=255;
////     if((pImg=cvLoadImage("e:\\test\\lena.bmp",1))!=NULL)
////     {
////         cvNamedWindow("source",1);
////         cvShowImage("source",pImg);
////         pHarrisImg=cvCreateImage(cvGetSize(pImg),IPL_DEPTH_32F,1);
////         //there we should define IPL_DEPTH_32F rather than IPL_DEPTH_8U
////         grayImage=cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);
////         dst8=cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);//this is for the result image
////         grayImage->origin=pImg->origin;  //there make sure the same  origin between grayImage and pImg
////
////         cvCvtColor(pImg,grayImage,CV_BGR2GRAY);//cause harris need gray scale image,we should convert RGB 2 gray
////
////         int block_size=7;
////         //do harris algorithm
////         cvCornerHarris(grayImage,pHarrisImg,block_size,7,0.04);//Harris角点检测
////
////         //convert scale so that we see the clear image
////         cvMinMaxLoc(pHarrisImg,&minVal,&maxVal,NULL,NULL,0);
////
////          std::cout<<minVal<<std::endl;
////          std::cout<<maxVal<<std::endl;
////
////         scale=(max-min)/(maxVal-minVal);
////         shift=-minVal*scale+min;
////         cvConvertScale(pHarrisImg,dst8,scale,shift);
////         cvNamedWindow("Harris",1);
////         cvShowImage("Harris",dst8);
////         cvWaitKey(0);
////         cvDestroyWindow("source");
////         cvDestroyWindow("Harris");
////         cvReleaseImage(&dst8);
////         cvReleaseImage(&pHarrisImg);
////         return 0;
////     }
////     return 1;
////}
////*/
////
//////// sift_test.cpp : 定义控制台应用程序的入口点。
//////int main(int argc,char* argv[])
//////{
//////   /* Mat image01=imread("e:\\test\\lena.bmp");  
//////    Mat image02=imread("e:\\test\\lena1.bmp");  
//////    Mat img_1,img_2;  
//////    GaussianBlur(image01,img_1,Size(3,3),0.5);  
//////    GaussianBlur(image02,img_2,Size(3,3),0.5); */
//////
//////   Mat img_1=imread("E:\\test\\Lena.bmp");//宏定义时CV_LOAD_IMAGE_GRAYSCALE=0，也就是读取灰度图像
//////   Mat img_2=imread("E:\\test\\Lena1.bmp");//一定要记得这里路径的斜线方向，这与Matlab里面是相反的
//////
//////
//////	
//////
//////if(!img_1.data || !img_2.data)//如果数据为空
//////    {
//////        cout<<"opencv error"<<endl;
//////		waitKey(100);
//////		system("pause");
//////return -1;
//////    }
//////    cout<<"open right"<<endl;
//////	
//////
////////第一步，用SIFT算子检测关键点
//////	//SurfFeatureDetector detector();
//////    SiftFeatureDetector detector(200);//构造函数采用内部默认的，参数越小，点越少！！
//////    std::vector<KeyPoint> keypoints_1,keypoints_2;//构造2个专门由点组成的点向量用来存储特征点
//////
//////    detector.detect(img_1,keypoints_1);//将img_1图像中检测到的特征点存储起来放在keypoints_1中
//////    detector.detect(img_2,keypoints_2);//同理
//////
////////在图像中画出特征点
//////    Mat img_keypoints_1,img_keypoints_2;
//////
//////    drawKeypoints(img_1,keypoints_1,img_keypoints_1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);//在内存中画出特征点
//////    drawKeypoints(img_2,keypoints_2,img_keypoints_2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
//////
//////    imshow("sift_keypoints_1",img_keypoints_1);//显示特征点
//////    imshow("sift_keypoints_2",img_keypoints_2);
//////
////////计算特征向量
//////    SiftDescriptorExtractor extractor;//定义描述子对象
//////
//////    Mat descriptors_1,descriptors_2;//存放特征向量的矩阵
//////
//////    extractor.compute(img_1,keypoints_1,descriptors_1);//计算特征向量
//////    extractor.compute(img_2,keypoints_2,descriptors_2);
//////
////////用burte force进行匹配特征向量
//////    BruteForceMatcher<L2<float>>matcher;//定义一个burte force matcher对象
//////    vector<DMatch>matches;
//////    matcher.match(descriptors_1,descriptors_2,matches);
//////
////////绘制匹配线段
//////    Mat img_matches;
//////    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_matches);//将匹配出来的结果放入内存img_matches中
//////
////////显示匹配线段
//////    imshow("sift_Matches",img_matches);//显示的标题为Matches
//////    waitKey(0);
//////return 0;
//////system("pause");
//////}
////
////
/////*
// surf_test.cpp : 定义控制台应用程序的入口点。

#include <iostream>
#include <fstream>
int main(int argc,char* argv[])
{
    //Mat img_1=imread("E:\\test\\111.png");//宏定义时CV_LOAD_IMAGE_GRAYSCALE=0，也就是读取灰度图像
    //Mat img_2=imread("E:\\test\\222.png");//一定要记得这里路径的斜线方向，这与Matlab里面是相反的
////	Mat img_1=imread("e:\\张英测试报告\\0303旋转缩放\\2_UP.jpg");
//	Mat img_2=imread("e:\\张英测试报告\\0303旋转缩放\\1_1000.jpg");//使用Photoshop进行透视处理的图像，配准结果差！！





	


	Mat img_1=imread("e:\\test_t\\A2.jpg");	//大图
	Mat img_2=imread("e:\\test_t\\B2.jpg");	//大图

if(!img_1.data || !img_2.data)//如果数据为空
    {
        cout<<"opencv error"<<endl;
return -1;
    }
    cout<<"open right"<<endl;

//第一步，用SURF算子检测关键点

	  int minHessian=5000;//400和800是实验中用的较多的值
    SurfFeatureDetector detector(minHessian);//参数为图像Hessian矩阵判别式的阈值，这个值即参数越大，表示特征点越少，越稳定

    std::vector<KeyPoint> keypoints_1,keypoints_2;//构造2个专门由点组成的点向量用来存储特征点

    detector.detect(img_1,keypoints_1);//将img_1图像中检测到的特征点存储起来放在keypoints_1中
    detector.detect(img_2,keypoints_2);//同理

//在图像中画出特征点
    Mat img_keypoints_1,img_keypoints_2;

    drawKeypoints(img_1,keypoints_1,img_keypoints_1,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(img_2,keypoints_2,img_keypoints_2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);

    imshow("surf_keypoints_1",img_keypoints_1);
    imshow("surf_keypoints_2",img_keypoints_2);

//计算特征向量
    SurfDescriptorExtractor extractor;//定义描述子对象

    Mat descriptors_1,descriptors_2;//存放特征向量的矩阵

    extractor.compute(img_1,keypoints_1,descriptors_1);
    extractor.compute(img_2,keypoints_2,descriptors_2);

//用burte force进行匹配特征向量
    BruteForceMatcher<L2<float>>matcher;//定义一个burte force matcher对象
    vector<DMatch>matches;
    matcher.match(descriptors_1,descriptors_2,matches);

//绘制匹配线段
    Mat img_matches;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_matches);//将匹配出来的结果放入内存img_matches中

	ofstream outFile1,outFile2;//创建对象
	outFile1.open("descriptor1.txt");//打开文本
	outFile2.open("descriptor2.txt");
	//输出左边图像的匹配点描述子
	for (int n = 0; n < 64; n++)
	{
		outFile1 << descriptors_1.at<int>(matches[0].queryIdx,n) << endl;
	}
	//输出右边图像的匹配点描述子
	for (int n = 0; n < 64; n++)
	{
		outFile2 << descriptors_2.at<int>(matches[0].trainIdx, n) << endl;
	}
	outFile1.close();//关闭文本
	outFile2.close();

	
	//显示匹配线段
    imshow("surf_Matches",img_matches);//显示的标题为Matches
    waitKey(0);
    return 0;
	system("pause");
}

////
////
/////*
//////透视变换代码（opencv 源代码）
////int main()
////{
////	CvPoint2D32f srcTri[4], dstTri[4];
////	CvMat*       warp_mat = cvCreateMat (3, 3, CV_32FC1);
////	IplImage*    src = NULL;
////	IplImage*    dst = NULL;
////
////	src = cvLoadImage ("E:\\test\\2.png", 1);
////	dst = cvCloneImage (src);
////	dst->origin = src->origin;
////	cvZero (dst);
////
////	//密集透视变换采用3*3矩阵和4个点的数组
////	srcTri[0].x = 0;
////	srcTri[0].y = 0;
////	srcTri[1].x = src->width - 1;
////	srcTri[1].y = 0;
////	srcTri[2].x = 0;
////	srcTri[2].y = src->height - 1;
////	srcTri[3].x = src->width - 1;
////	srcTri[3].y = src->height - 1;
////
////	//乘的系数确定对应点的位置，因此可以设定
////	dstTri[0].x = src->width * 0.05;
////	dstTri[0].y = src->height * 0.33;
////	dstTri[1].x = src->width * 0.9;
////	dstTri[1].y = src->height * 0.25;
////	dstTri[2].x = src->width * 0.2;
////	dstTri[2].y = src->height * 0.7;
////	dstTri[3].x = src->width * 0.8;
////	dstTri[3].y = src->height * 0.9;
////
////	//dstTri = srcTri * warp_mat 
////
////	cvGetPerspectiveTransform (srcTri, dstTri, warp_mat);
////	cvWarpPerspective (src, dst, warp_mat);
////
////	cvNamedWindow("src", 1);
////	cvShowImage("src", src);
////	//保存图像
////	cvSaveImage("E:\\test\\2.jpg",dst);
////
////	cvNamedWindow ("Affine_Transform", 1);
////	cvShowImage ("Affine_Transform", dst);
////	
////
////
////	cvWaitKey (0);
////
////	cvReleaseImage (&src);
////	cvReleaseImage (&dst);
////	cvReleaseMat (&warp_mat);
////
////	return 0;
////}
////*/
////
//
//
//
////****************************************//
////*********opencv stitch的拼接,速度特别慢********//
////***************************************//
///*
//#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/stitching/stitcher.hpp>
//bool try_use_gpu = false;
//vector<Mat> imgs;
////string result_name = "dst1.jpg";
//int main(int argc, char * argv[])
//{
//    Mat img1 = imread("E:\\test\\123.png");
//    Mat img2 = imread("E:\\test\\123.tif");
//
//    imshow("p1", img1);
//    imshow("p2", img2);
//
//    if (img1.empty() || img2.empty())
//    {
//        cout << "Can't read image" << endl;
//        return -1;
//    }
//    imgs.push_back(img1);
//    imgs.push_back(img2);
//
//
//    Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
//    // 使用stitch函数进行拼接
//    Mat pano;
//    Stitcher::Status status = stitcher.stitch(imgs, pano);
//    if (status != Stitcher::OK)
//    {
//        cout << "Can't stitch images, error code = " << int(status) << endl;
//        return -1;
//		system("pause");
//    }
// //   imwrite(result_name, pano);
//    Mat pano2 = pano.clone();
//    // 显示源图像，和结果图像
//    imshow("全景图像", pano);
//    if (waitKey() == 27)
//        return 0;
//}
//*/
//
//
//
//
//////surf 和图像拼接
////int main(int argc,char *argv[])    
////{    
////	Mat image01=imread("E:\\test\\xiaotu\\33.png");   
////    Mat image02=imread("E:\\test\\xioatu\\44.png");    //当使用ps使用旋转角度过大的图像，配准就会出问题，算法不能用，但是这个斜的角度没法确定
////                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
////    imshow("参考图",image01);  
////   imshow("测试图",image02);  
////  
////    //灰度图转换  
////    Mat image1,image2;    
////    cvtColor(image01,image1,CV_RGB2GRAY);  
////   cvtColor(image02,image2,CV_RGB2GRAY);  
//// 
////  
////   //提取特征点    
////    SurfFeatureDetector surfDetector(800);  // 海塞矩阵阈值  
////    vector<KeyPoint> keyPoint1,keyPoint2;    
////    surfDetector.detect(image1,keyPoint1);    
////   surfDetector.detect(image2,keyPoint2);    
////  
////    //特征点描述，为下边的特征点匹配做准备    
////    SurfDescriptorExtractor SurfDescriptor;    
////   Mat imageDesc1,imageDesc2;    
////    SurfDescriptor.compute(image1,keyPoint1,imageDesc1);    
////    SurfDescriptor.compute(image2,keyPoint2,imageDesc2);      
////  
////    //获得匹配特征点，并提取最优配对     
////    FlannBasedMatcher matcher;  
////   vector<DMatch> matchePoints;    
////    matcher.match(imageDesc1,imageDesc2,matchePoints,Mat());  
////   sort(matchePoints.begin(),matchePoints.end()); //特征点排序    
////  
////    //获取排在前N个的最优匹配特征点  
////    vector<Point2f> imagePoints1,imagePoints2;      
////    for(int i=0;i<10;i++)  
////    {         
////      imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);       
////        imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);       
////   }  
////    //获取图像1到图像2的投影映射矩阵 尺寸为3*3  
////    Mat homo=findHomography(imagePoints1,imagePoints2,CV_RANSAC);  
////   ////也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
////    //Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);     
////        cout<<"变换矩阵为：\n"<<homo<<endl<<endl; //输出映射矩阵  
////    //图像配准  
////   Mat imageTransform1,imageTransform2;  
//// 
////  warpPerspective(image01,imageTransform1,homo,Size(image01.cols,image01.rows));
////    imshow("经过透视矩阵变换后",imageTransform1);  
////      
////   waitKey();    
////    return 0;    
////} 
//
//
//
////
///**
// * @file SURF_Homography
// * @brief SURF detector + descriptor + FLANN Matcher + FindHomography
// * @author A. Huaman
// */
////********这个程序中有画框、匹配点数统计、显示等可以借鉴！！****//
////**************************************************//
//
////int main()
////{
////	initModule_nonfree();//初始化模块，使用SIFT或SURF时用到 
////	Ptr<FeatureDetector> detector = FeatureDetector::create( "SURF" );//创建SIFT特征检测器，可改成SURF/ORB
////	Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SURF" );//创建特征向量生成器，可改成SURF/ORB
////	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );//创建特征匹配器  
////	if( detector.empty() || descriptor_extractor.empty() )  
////		cout<<"fail to create detector!";  
////
////	//读入图像  
////	Mat img1 = imread("E:\\test\\xiaotu\\44.png");	//小图、对象
////	Mat img2 = imread("E:\\test\\xiaotu\\1.tif");  //大图，参考图
////	  
////	//特征点检测  
////	double t = getTickCount();//当前滴答数  
////	vector<KeyPoint> m_LeftKey,m_RightKey;  
////	detector->detect( img1, m_LeftKey );//检测img1中的SIFT特征点，存储到m_LeftKey中  
////	detector->detect( img2, m_RightKey );  
////	cout<<"图像1特征点个数:"<<m_LeftKey.size()<<endl;  
////	cout<<"图像2特征点个数:"<<m_RightKey.size()<<endl;  
////	
////	
////	//根据特征点计算特征描述子矩阵，即特征向量矩阵  
////	Mat descriptors1,descriptors2;  
////	descriptor_extractor->compute( img1, m_LeftKey, descriptors1 );  
////	descriptor_extractor->compute( img2, m_RightKey, descriptors2 );  
////	
////	t = ((double)getTickCount() - t)/getTickFrequency();  
////	cout<<"SIFT算法用时t："<<t<<"秒"<<endl;  
////	//cout<<"图像1特征描述矩阵大小："<<descriptors1.size()  
////	//	<<"，特征向量个数："<<descriptors1.rows<<"，维数："<<descriptors1.cols<<endl;  
////	//cout<<"图像2特征描述矩阵大小："<<descriptors2.size()  
////	//	<<"，特征向量个数："<<descriptors2.rows<<"，维数："<<descriptors2.cols<<endl;  
////
////	//画出特征点  
////	Mat img_m_LeftKey,img_m_RightKey;  
////	drawKeypoints(img1,m_LeftKey,img_m_LeftKey,Scalar::all(-1),0);  
////	drawKeypoints(img2,m_RightKey,img_m_RightKey,Scalar::all(-1),0);  
////	//imshow("Src1",img_m_LeftKey);  
////	//imshow("Src2",img_m_RightKey);  
////
////	//特征匹配  
////	vector<DMatch> matches;//匹配结果  
////	descriptor_matcher->match( descriptors1, descriptors2, matches );//匹配两个图像的特征矩阵  
////	cout<<"Match个数："<<matches.size()<<endl;  
////
////	//计算匹配结果中距离的最大和最小值  
////	//距离是指两个特征向量间的欧式距离，表明两个特征的差异，值越小表明两个特征点越接近  
////	double max_dist = 0;  
////	double min_dist = 100;  
////	for(int i=0; i<matches.size(); i++)  
////	{  
////		double dist = matches[i].distance;  
////		if(dist < min_dist) min_dist = dist;  
////		if(dist > max_dist) max_dist = dist;  
////	}  
////	cout<<"最大距离："<<max_dist<<endl;  
////	cout<<"最小距离："<<min_dist<<endl;  
////
////	//筛选出较好的匹配点  
////	vector<DMatch> goodMatches;  
////	for(int i=0; i<matches.size(); i++)  
////	{  
////		if(matches[i].distance < 0.2 * max_dist)  
////		{  
////			goodMatches.push_back(matches[i]);  
////		}  
////	}  
////	cout<<"goodMatch个数："<<goodMatches.size()<<endl;  
////
////	//画出匹配结果  
////	Mat img_matches;  
////	//红色连接的是匹配的特征点对，绿色是未匹配的特征点  
////	drawMatches(img1,m_LeftKey,img2,m_RightKey,goodMatches,img_matches,  
////		Scalar::all(-1)/*CV_RGB(255,0,0)*/,CV_RGB(0,255,0),Mat(),2);  
////
////	imshow("MatchSIFT",img_matches);  
////	IplImage result=img_matches;
////
////	waitKey(0);
////
////
////	//RANSAC匹配过程
////	vector<DMatch> m_Matches=goodMatches;
////	// 分配空间
////	int ptCount = (int)m_Matches.size();
////	Mat p1(ptCount, 2, CV_32F);
////	Mat p2(ptCount, 2, CV_32F);
////
////	// 把Keypoint转换为Mat
////	Point2f pt;
////	for (int i=0; i<ptCount; i++)
////	{
////		pt = m_LeftKey[m_Matches[i].queryIdx].pt;
////		p1.at<float>(i, 0) = pt.x;
////		p1.at<float>(i, 1) = pt.y;
////
////		pt = m_RightKey[m_Matches[i].trainIdx].pt;
////		p2.at<float>(i, 0) = pt.x;
////		p2.at<float>(i, 1) = pt.y;
////	}
////
////	// 用RANSAC方法计算F
////	Mat m_Fundamental;
////	vector<uchar> m_RANSACStatus;       // 这个变量用于存储RANSAC后每个点的状态
////	findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);
////
////	// 计算野点个数
////
////	int OutlinerCount = 0;
////	for (int i=0; i<ptCount; i++)
////	{
////		if (m_RANSACStatus[i] == 0)    // 状态为0表示野点
////		{
////			OutlinerCount++;
////		}
////	}
////	int InlinerCount = ptCount - OutlinerCount;   // 计算内点
////	cout<<"内点数为："<<InlinerCount<<endl;
////
////
////	// 这三个变量用于保存内点和匹配关系
////	vector<Point2f> m_LeftInlier;
////	vector<Point2f> m_RightInlier;
////	vector<DMatch> m_InlierMatches;
////
////	m_InlierMatches.resize(InlinerCount);
////	m_LeftInlier.resize(InlinerCount);
////	m_RightInlier.resize(InlinerCount);
////	InlinerCount=0;
////	float inlier_minRx=img1.cols;        //用于存储内点中右图最小横坐标，以便后续融合
////
////	for (int i=0; i<ptCount; i++)
////	{
////		if (m_RANSACStatus[i] != 0)
////		{
////			m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
////			m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
////			m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
////			m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
////			m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
////			m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
////
////			if(m_RightInlier[InlinerCount].x<inlier_minRx) inlier_minRx=m_RightInlier[InlinerCount].x;   //存储内点中右图最小横坐标
////
////			InlinerCount++;
////		}
////	}
////
////	// 把内点转换为drawMatches可以使用的格式
////	vector<KeyPoint> key1(InlinerCount);
////	vector<KeyPoint> key2(InlinerCount);
////	KeyPoint::convert(m_LeftInlier, key1);
////	KeyPoint::convert(m_RightInlier, key2);
////
////	// 显示计算F过后的内点匹配
////	Mat OutImage;
////	drawMatches(img1, key1, img2, key2, m_InlierMatches, OutImage);
////	cvNamedWindow( "Match features", 1);
////	cvShowImage("Match features", &IplImage(OutImage));
////	waitKey(0);
////
////	cvDestroyAllWindows();
////
////	//矩阵H用以存储RANSAC得到的单应矩阵
////	Mat H = findHomography( m_LeftInlier, m_RightInlier, RANSAC );
////
////	//存储左图四角，及其变换到右图位置
////	std::vector<Point2f> obj_corners(4);
////	obj_corners[0] = Point(0,0); obj_corners[1] = Point( img1.cols, 0 );
////	obj_corners[2] = Point( img1.cols, img1.rows ); obj_corners[3] = Point( 0, img1.rows );
////	std::vector<Point2f> scene_corners(4);
////	perspectiveTransform( obj_corners, scene_corners, H);
////
////	//画出变换后图像位置
////	Point2f offset( (float)img1.cols, 0);  
////	line( OutImage, scene_corners[0]+offset, scene_corners[1]+offset, Scalar( 0, 255, 0), 4 );
////	line( OutImage, scene_corners[1]+offset, scene_corners[2]+offset, Scalar( 0, 255, 0), 4 );
////	line( OutImage, scene_corners[2]+offset, scene_corners[3]+offset, Scalar( 0, 255, 0), 4 );
////	line( OutImage, scene_corners[3]+offset, scene_corners[0]+offset, Scalar( 0, 255, 0), 4 );
////	imshow( "Good Matches & Object detection", OutImage );
////
////	waitKey(0);
////	imwrite("warp_position.jpg",OutImage);
////
//// 
//
//
//
//
//
//
//// 	int drift = scene_corners[1].x;                                                        //储存偏移量
//// 
//// 	//新建一个矩阵存储配准后四角的位置
//// 	int width = int(max(abs(scene_corners[1].x), abs(scene_corners[2].x)));
//// //	int height= img1.rows;                                                                  //或者：int height = int(max(abs(scene_corners[2].y), abs(scene_corners[3].y)));
//// 	int height= img2.rows;    
////	float origin_x=0,origin_y=0;
//// 	if(scene_corners[0].x<0) {
//// 		if (scene_corners[3].x<0) origin_x+=min(scene_corners[0].x,scene_corners[3].x);
//// 		else origin_x+=scene_corners[0].x;}
//// 	width-=int(origin_x);
//// 	if(scene_corners[0].y<0) {
//// 		if (scene_corners[1].y) origin_y+=min(scene_corners[0].y,scene_corners[1].y);
//// 		else origin_y+=scene_corners[0].y;}
//// 	//可选：height-=int(origin_y);
//// 	Mat imageturn=Mat::zeros(width,height,img1.type());
//// 
//// 	//获取新的变换矩阵，使图像完整显示
//// 	for (int i=0;i<4;i++) {scene_corners[i].x -= origin_x; } 	//可选：scene_corners[i].y -= (float)origin_y; }
//// 	Mat H1=getPerspectiveTransform(obj_corners, scene_corners);
//// 
//// 	//进行图像变换，显示效果
////	warpPerspective(img1,imageturn,H1,Size(width,height));	
//// 	imshow("image_Perspective", imageturn);
//// 	waitKey(0);
//// 
//// 
//// 	//图像融合
//// 	int width_ol=width-int(inlier_minRx-origin_x);
//// 	int start_x=int(inlier_minRx-origin_x);
//// 	cout<<"width: "<<width<<endl;
//// 	cout<<"img1.width: "<<img1.cols<<endl;
//// 	cout<<"start_x: "<<start_x<<endl;
//// 	cout<<"width_ol: "<<width_ol<<endl;
//// 
//// 	uchar* ptr=imageturn.data;
//// 	double alpha=0, beta=1;
//// 	for (int row=0;row<height;row++) {
//// 		ptr=imageturn.data+row*imageturn.step+(start_x)*imageturn.elemSize();
//// 		for(int col=0;col<width_ol;col++)
//// 		{
//// 			uchar* ptr_c1=ptr+imageturn.elemSize1();  uchar*  ptr_c2=ptr_c1+imageturn.elemSize1();
//// 			uchar* ptr2=img2.data+row*img2.step+(col+int(inlier_minRx))*img2.elemSize();
//// 			uchar* ptr2_c1=ptr2+img2.elemSize1();  uchar* ptr2_c2=ptr2_c1+img2.elemSize1();
//// 
//// 
//// 			alpha=double(col)/double(width_ol); beta=1-alpha;
//// 
//// 			if (*ptr==0&&*ptr_c1==0&&*ptr_c2==0) {
//// 				*ptr=(*ptr2);
//// 				*ptr_c1=(*ptr2_c1);
//// 				*ptr_c2=(*ptr2_c2);
//// 			}
//// 
//// 			*ptr=(*ptr)*beta+(*ptr2)*alpha;
//// 			*ptr_c1=(*ptr_c1)*beta+(*ptr2_c1)*alpha;
//// 			*ptr_c2=(*ptr_c2)*beta+(*ptr2_c2)*alpha;
//// 
//// 			ptr+=imageturn.elemSize();
//// 		}	}
//// 
//// 	//imshow("image_overlap", imageturn);
//// 	//waitKey(0);
//// 
//// 	Mat img_result=Mat::zeros(height,width+img2.cols-drift,img1.type());
//// 	uchar* ptr_r=imageturn.data;
//// 
//// 	for (int row=0;row<height;row++) {
//// 		ptr_r=img_result.data+row*img_result.step;
//// 
//// 		for(int col=0;col<imageturn.cols;col++)
//// 		{
//// 			uchar* ptr_rc1=ptr_r+imageturn.elemSize1();  uchar*  ptr_rc2=ptr_rc1+imageturn.elemSize1();
//// 
//// 			uchar* ptr=imageturn.data+row*imageturn.step+col*imageturn.elemSize();
//// 			uchar* ptr_c1=ptr+imageturn.elemSize1();  uchar*  ptr_c2=ptr_c1+imageturn.elemSize1();
//// 
//// 			*ptr_r=*ptr;
//// 			*ptr_rc1=*ptr_c1;
//// 			*ptr_rc2=*ptr_c2;
//// 
//// 			ptr_r+=img_result.elemSize();
//// 		}	
//// 
//// 		ptr_r=img_result.data+row*img_result.step+imageturn.cols*img_result.elemSize();
//// 		for(int col=imageturn.cols;col<img_result.cols;col++)
//// 		{
//// 			uchar* ptr_rc1=ptr_r+imageturn.elemSize1();  uchar*  ptr_rc2=ptr_rc1+imageturn.elemSize1();
//// 
//// 			uchar* ptr2=img2.data+row*img2.step+(col-imageturn.cols+drift)*img2.elemSize();
//// 			uchar* ptr2_c1=ptr2+img2.elemSize1();  uchar* ptr2_c2=ptr2_c1+img2.elemSize1();
//// 
//// 			*ptr_r=*ptr2;
//// 			*ptr_rc1=*ptr2_c1;
//// 			*ptr_rc2=*ptr2_c2;
//// 
//// 			ptr_r+=img_result.elemSize();
//// 		}	
//// 	}
//// 
//// 	imshow("image_result", img_result);
////	//imwrite("final_result.jpg",img_result);
////	waitKey(0);
////	return 0;
////}
//
//
//
//
//
//
//
//
////
////
////
//////****图像拼接算法：①直接法②按权重叠加图1和图2的重叠部分*****//
//////**********************************************//
////
////
//////计算原始图像点位在经过矩阵变换后在目标图像上对应位置
////Point2f getTransformPoint(const Point2f originalPoint,const Mat &transformMaxtri);
////
////int main()  
////{  
////	Mat image02=imread("E:\\test\\44.png");  
////	Mat image01=imread("E:\\test\\33.png");
////		imshow("拼接图像1",image01);
////	    imshow("拼接图像2",image02);
////
////
////	//灰度图转换
////	Mat image1,image2;  
////	cvtColor(image01,image1,CV_RGB2GRAY);
////	cvtColor(image02,image2,CV_RGB2GRAY);
////
////
////
////	//提取特征点  
////	SiftFeatureDetector siftDetector(800);  // 海塞矩阵阈值
////	vector<KeyPoint> keyPoint1,keyPoint2;  
////	siftDetector.detect(image1,keyPoint1);  
////	siftDetector.detect(image2,keyPoint2);	
////
////	//特征点描述，为下边的特征点匹配做准备  
////	SiftDescriptorExtractor siftDescriptor;  
////	Mat imageDesc1,imageDesc2;  
////	siftDescriptor.compute(image1,keyPoint1,imageDesc1);  
////	siftDescriptor.compute(image2,keyPoint2,imageDesc2);	
////
////	//获得匹配特征点，并提取最优配对  	
////	FlannBasedMatcher matcher;
////	vector<DMatch> matchePoints;  
////	matcher.match(imageDesc1,imageDesc2,matchePoints,Mat());
////	sort(matchePoints.begin(),matchePoints.end()); //特征点排序	
////	//获取排在前N个的最优匹配特征点
////	vector<Point2f> imagePoints1,imagePoints2;
////	for(int i=0;i<10;i++)
////	{		
////		imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);//matchePoints[i].queryIdx保存着第一张图片匹配点的序号；().pt表示点；pt.x位x坐标
////		imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);	//matchePoints[i].trainIdx保存第二张，训练集	
////	}
////
////	//获取图像1到图像2的投影映射矩阵，尺寸为3*3
////	Mat homo=findHomography(imagePoints1,imagePoints2,CV_RANSAC);	
////
////	/*Mat adjustMat;
////	if(homo.at<Vec3b>(1,2)[0] <0)
////	{
////		adjustMat=(Mat_<double>(3,3)<<1.0 ,0,image01.cols,0,1.0,0,0,0,1.0 );
////	}
////	else
////	{
////		adjustMat=(Mat_<double>(3,3)<<1.0 ,0,image01.cols,0,1.0,image01.rows,0,0,1.0 );
////	}
////*/
////	
////	//Mat adjustMat=(Mat_<double>(3,3)<<1.0 ,0,image01.cols,0,1.0,0,0,0,1.0 );
////	//下面一句程序是对homo矩阵进行修正
////	Mat adjustMat=(Mat_<double>(3,3)<<1.0 ,0,image01.cols,0,1.0,image01.rows,0,0,1.0 );//向后偏移image01.cols矩阵;向下移设置第二行第三个数。这样做是为了修正H矩阵的值为正，否则图像会因为越界缺失
////	////Mat adjustMat=(Mat_<double>(3,3)<<1.0,0,35,0,1.0,65,0,0,1.0); //修正矩阵时也可以直接使用数字
////	Mat adjustHomo= adjustMat*homo;
////
////	cout<<"变换矩阵为：\n"<<homo<<endl<<endl; //输出映射矩阵  
////	cout<<"变换矩阵为：\n"<<adjustHomo<<endl<<endl; //输出映射矩阵 
////
////
////	//获取最强配对点在原始图像、矩阵变换后图像上的对应位置，用于图像拼接点的定位
////	Point2f originalLinkPoint,targetLinkPoint,basedImagePoint;
////	originalLinkPoint=keyPoint1[matchePoints[0].queryIdx].pt;//匹配点在图1中的位置
////	targetLinkPoint=getTransformPoint(originalLinkPoint,adjustHomo);//匹配点在变换后在透视图的位置	//getTransformPoint：得到原始图像特征点按照adjustHomo矩阵变换之后，的结果点，赋给targetLinkPoint
////	basedImagePoint=keyPoint2[matchePoints[0].trainIdx].pt;//匹配点在图2中的位置
////
////	//图像配准
////	Mat imageTransform1;
////	warpPerspective(image01,imageTransform1,adjustMat*homo,Size(image02.cols+image01.cols+200,+image01.rows+image02.rows+200));
////	
////	imshow("经过透视矩阵变换后",imageTransform1);
////	
////
////	////第一种方法
////	////在最强匹配点的位置处衔接，最强匹配点左侧是图1，右侧是图2，这样直接替换图像衔接不好，光线有突变
////	//Mat ROIMat=image02(Rect(Point(basedImagePoint.x,0),Point(image02.cols,image02.rows)));	
////	//ROIMat.copyTo(Mat(imageTransform1,Rect(targetLinkPoint.x,0,image02.cols-basedImagePoint.x+1,image02.rows)));
////	
////
////	//第二种方法
////	//在最强匹配点左侧的重叠区域进行累加，是衔接稳定过渡，消除突变
////	Mat image1Overlap,image2Overlap; //定义图1和图2的重叠部分	
////	image1Overlap=imageTransform1(Rect(Point(targetLinkPoint.x-basedImagePoint.x,0),Point(targetLinkPoint.x,image02.rows)));  
////    image2Overlap=image02(Rect(0,0,image1Overlap.cols,image1Overlap.rows));  
////	
////
////	Mat image1ROICopy=image1Overlap.clone();  //复制一份图1的重叠部分
////	for(int i=0;i<image1Overlap.rows;i++)
////	{
////		for(int j=0;j<image1Overlap.cols;j++)
////		{
////			double weight;
////			weight=(double)j/image1Overlap.cols;  //随距离改变而改变的叠加系数
////			image1Overlap.at<Vec3b>(i,j)[0]=(1-weight)*image1ROICopy.at<Vec3b>(i,j)[0]+weight*image2Overlap.at<Vec3b>(i,j)[0];
////			image1Overlap.at<Vec3b>(i,j)[1]=(1-weight)*image1ROICopy.at<Vec3b>(i,j)[1]+weight*image2Overlap.at<Vec3b>(i,j)[1];
////			image1Overlap.at<Vec3b>(i,j)[2]=(1-weight)*image1ROICopy.at<Vec3b>(i,j)[2]+weight*image2Overlap.at<Vec3b>(i,j)[2];
////			//Vec3b是向量模板类。表示每一个Vec3b对象中,可以存储3个char，一般用RGB
////	
////	
////		}
////	}
////
////   Mat ROIMat=image02(Rect(Point(image1Overlap.cols,0),Point(image02.cols,image02.rows)));  //图2中不重合的部分  
//// //  ROIMat.copyTo(Mat(imageTransform1,Rect(targetLinkPoint.x,0, ROIMat.cols,image02.rows))); //不重合的部分直接衔接上去
//// ROIMat.copyTo(Mat(imageTransform1,Rect(targetLinkPoint.x,image01.rows,ROIMat.cols,ROIMat.rows)));//这句可以解决图像透视变换后缺失的情况
////
////
////	
////
////	//namedWindow("拼接结果",0);
////	imshow("拼接结果",imageTransform1);	
////	
////	waitKey();  
////	return 0;  
////}
////
//////计算原始图像点位在经过矩阵变换后在目标图像上对应位置
////Point2f getTransformPoint(const Point2f originalPoint,const Mat &transformMaxtri)
////{
////	Mat originelP,targetP;
////	originelP=(Mat_<double>(3,1)<<originalPoint.x,originalPoint.y,1.0);
////	targetP=transformMaxtri*originelP;
////	float x=targetP.at<double>(0,0)/targetP.at<double>(2,0);
////	float y=targetP.at<double>(1,0)/targetP.at<double>(2,0);
////	return Point2f(x,y);
////}
//
//
////#include <opencv2/core/core.hpp>
////#include <opencv2/features2d/features2d.hpp>
////#include <opencv2/highgui/highgui.hpp>
////#include <cv.h>
////#include <vector>
////
////using namespace cv;
////using namespace std;
////
//////FAST角点检测算法
////int main()
////{
////	Mat frame=imread("E:\\test\\kuailejiazu.jpg", 1);
////	double t = getTickCount();//当前滴答数
////	std::vector<KeyPoint> keyPoints;
////	FastFeatureDetector fast(100);	// 检测的阈值为50
////
////	fast.detect(frame, keyPoints);
////	drawKeypoints(frame, keyPoints, frame, Scalar(0,0,255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
////	
////	t = ((double)getTickCount() - t)/getTickFrequency();
////	cout<<"算法用时："<<t<<"秒"<<endl;
////
////	imshow("FAST特征点", frame);
////	cvWaitKey(0);
////
////	return 0;
////}
//
//
////#include<iostream>
////#include<opencv2/highgui/highgui.hpp>
////#include<opencv2/core/core.hpp>
////#include<opencv2/imgproc/imgproc.hpp>
////using namespace std;
////using namespace cv;
//
///*************轮廓检测**************/
///*********************************/
////int main()
////{
////	//读入和显示原图
////	string image_name="E:\\test\\2.jpg";
////	Mat src=imread(image_name);
////	imshow("src",src);
////	//转换成灰度图
////	Mat gray(src.size(),CV_8U);
////	cvtColor(src,gray,CV_BGR2GRAY);
////	imshow("gray",gray);
////	//设置阈值，转换成二值图像
////	threshold(gray,gray,128,255,THRESH_BINARY);
////	imshow("binary",gray);
////	//检测轮廓
////	std::vector<std::vector<cv::Point>>contours;
////	cv::findContours(gray,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
////	std::cout<<"Contours:"<<contours.size()<<std::endl;
////	std::vector<std::vector<cv::Point>>::const_iterator itContours=contours.begin();
////	for(;itContours!=contours.end();++itContours)
////	{
////		std::cout<<"Size:"<<itContours->size()<<std::endl;
////	}
////	cv::Mat result(gray.size(),CV_8U,cv::Scalar(255));
////	cv::drawContours(result,contours,-1,cv::Scalar(0),2);
////	cv::namedWindow("Contours");
////	cv::imshow("Contours",result);
////	cv::Mat original=cv::imread(image_name);
////	cv::drawContours(original,contours,-1,cv::Scalar(255,255,255),-1);
////	cv::namedWindow("Contours on image");
////	cv::imshow("Contours on image",original);
////	result.setTo(cv::Scalar(255));
////	cv::drawContours(result,contours,-1,cv::Scalar(0),-1);
////	waitKey(0);
////	return 0;
////}
//
///***********************************/
////
////Mat src,dst;
////int spatialRad=10,colorRad=10,maxPryLevel=1;
////void meanshift_seg()
////{
////	pyrMeanShiftFiltering(src,dst,spatialRad,colorRad,maxPryLevel);
////	RNG rng=theRNG();
////	Mat mask(dst.rows+2,dst.cols+2,CV_8UC1,Scalar::all(0));
////	for(int i=0;i<dst.rows;i++)
////		for(int j=0;j<dst.cols;j++)
////			if(mask.at<uchar>(i+1,j+1)==0)
////			{
////				Scalar newcolor(rng(256),rng(256),rng(256));
////				floodFill(dst,mask,Point(j,i),newcolor,0,Scalar::all(1),Scalar::all(1));
////			}
////			imshow("dst",dst);
////}
////void meanshift_seg_s(int i,void*)
////{
////	spatialRad=i;
////	meanshift_seg();
////}
////void meanshift_seg_c(int i,void*)
////{
////	spatialRad=i;
////	meanshift_seg();
////}
////void meanshift_seg_m(int i,void*)
////{
////	spatialRad=i;
////	meanshift_seg();
////}
////int main(int argc,uchar* argv[])
////{
////	namedWindow("src",WINDOW_AUTOSIZE);
////	namedWindow("dst",WINDOW_AUTOSIZE);
////	src=imread("E:\\test\\2.jpg");
////	CV_Assert(!src.empty());
////	spatialRad=10;
////	colorRad=10;
////	maxPryLevel=1;
////	createTrackbar("spatialRad","dst",&spatialRad,80,meanshift_seg_s);
////	createTrackbar("colorRad","dst",&colorRad,60,meanshift_seg_c);
////	createTrackbar("maxPryLevel","dst",&maxPryLevel,5,meanshift_seg_m);
////	imshow("src",src);
////	imshow("dst",src);
////	imshow("flood",src);
////	waitKey();
////	return 0;
////}
//
//
//
////int main()  
////{  
////	Mat src,dst_up,dst_down;  
////	src = imread("E:\\张英测试报告\\0303旋转缩放\\2.jpg", 1);  
////	if (src.empty())  
////	{  
////		printf("cannot load!");  
////		return -1;  
////	}  
////	namedWindow("原图");  
////	imshow("原图", src);  
////
////	//上采样  
////	pyrUp(src, dst_up, Size(src.cols * 2, src.rows * 2));  
////	imwrite("E:\\张英测试报告\\0303旋转缩放\\2_pyrUP.jpg",dst_up);
////	namedWindow("上采样");  
////	imshow("上采样", dst_up);  
////
////	//下采样  
////	pyrDown(src, dst_down, Size(src.cols / 2, src.rows / 2));  
////	namedWindow("下采样");  
////	imshow("下采样", dst_down);  
////
////	//高斯不同  
////	Mat g1, g2, dogImg;  
////	GaussianBlur(src, g1, Size(5, 5), 0, 0);  
////	GaussianBlur(src, g2, Size(5, 5), 11, 11);  
////	subtract(g1, g2, dogImg);  
////	//归一化显示  
////	normalize(dogImg, dogImg, 255, 0, NORM_MINMAX);//因为两个图像的差值肯定不大，看起来一片黑，所以需要把像素扩展到0~255的区间。  
////	namedWindow("高斯不同");  
////	imshow("高斯不同", dogImg);  
////
////	waitKey(0);  
////	return 0;  
////}  
//
//int main(int argc, char** argv) 
//{ 
//	Mat img_1 = imread("e:\\张英测试报告\\0130快速配准软件测试图\\1_4\\2.jpg"); 
//	Mat img_2 = imread("e:\\张英测试报告\\0130快速配准软件测试图\\1_4\\2_30.jpg");
//
//	// -- Step 1: Detect the keypoints using STAR Detector 
//	std::vector<KeyPoint> keypoints_1,keypoints_2; 
//	FastFeatureDetector detector(40); 
//	detector.detect(img_1, keypoints_1); 
//	detector.detect(img_2, keypoints_2);
//
//	// -- Stpe 2: Calculate descriptors (feature vectors) 
//	FREAK brief; 
//	Mat descriptors_1, descriptors_2; 
//	brief.compute(img_1, keypoints_1, descriptors_1); 
//	brief.compute(img_2, keypoints_2, descriptors_2);
//
//	//-- Step 3: Matching descriptor vectors with a brute force matcher 
//	BFMatcher matcher(NORM_HAMMING); 
//	std::vector<DMatch> mathces; 
//	matcher.match(descriptors_1, descriptors_2, mathces); 
//	// -- dwaw matches 
//	Mat img_mathes; 
//	drawMatches(img_1, keypoints_1, img_2, keypoints_2, mathces, img_mathes); 
//	// -- show 
//	imshow("Mathces", img_mathes);
//
//	waitKey(0); 
//	return 0; 
//}









/**
 * Object matching using function matchTemplate
 */



/// 全局变量 ///
Mat srcImg;			//原始图像
Mat templImg;		//模板图像
Mat resultImg;		//匹配结果图像
Mat img1_down;
Mat img2_down;
string dir,dir1,dir2,dir3,dir1_down,dir2_down,dir_result;

const char* imageWindow = "Source Image";		//原始图像显示窗口
const char* resultWindow = "Result Window";		//匹配结果图像显示窗口

int matchMethod;		//匹配方法index
int maxTrackbar = 5;	//滑动条范围（与匹配方法个数对应）

void MatchingMethod( int, void* );		//匹配函数
double scale;

////据说是具有尺度缩放性的模板匹配改进法。。。实验结果不佳
int app(vector<double> minV);  

//int main( )
//{
//	string dir = "e:\\张英测试报告\\0312SURF_BRISK\\1\\";
//	dir2 = dir + "2_4.jpg";
//	dir1 = dir + "1_4.jpg";
//	dir3 = dir;
//	dir_result =dir3 + "sf_result.jpg";
//	Mat img = imread(dir1);       //原图像
//	Mat temp = imread(dir2);     //模板
//	Mat result;
//
//	vector<double> minV;
//	vector<Point> minL;
//	double t = cvGetTickCount();
//
//
//	vector<Mat> down_temp;
//	down_temp.push_back(temp);
//
//
//	for (int i=0;i<6;i++)
//	{
//		Mat temp1;
//
//		//		cout<<down_temp[i].cols<<endl;
//		//		cout<<down_temp[i].rows<<endl;
//
//		int result_cols =  img.cols - down_temp[i].cols + 1;
//		int result_rows = img.rows - down_temp[i].rows + 1;
//		result.create( result_cols, result_rows, CV_32FC1 );
//		matchTemplate( img, down_temp[i], result, CV_TM_SQDIFF );
//
//		double minVal; 
//		double maxVal; 
//		Point  minLoc; 
//		Point  maxLoc;
//		Point  matchLoc;
//
//		minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc);
//
//		minVal=minVal/(down_temp[i].cols*down_temp[i].rows);
//		//	cout<<minVal<<endl;
//		minV.push_back(minVal);
//		minL.push_back(minLoc);
//		resize( down_temp[i], temp1, Size( down_temp[i].cols/1.3, down_temp[i].rows/1.3) );
//		down_temp.push_back(temp1);
//	}
//
//
//	int location;
//	location = app(minV); 
//	//	cout<<location<<endl;
//
//	rectangle( img, minL[location], Point( minL[location].x + down_temp[location].cols , minL[location].y + down_temp[location].rows ), Scalar::all(0), 2, 8, 0 );
//
//	t = (double)(cvGetTickCount() -t )/(cvGetTickFrequency()*1000.*1000.);
//	cout<<"尺度缩放模板匹配时间："<<t<<endl;
//
//	imwrite(dir_result,img);
//
//	imshow("结果",img);
//
//	waitKey();
//
//
//	return 0;
//}


//int app(vector<double> minV)           
//{
//	int t=0;
//
//
//	for (int i = 1; i < minV.size();i++)
//	{
//		if (minV[i] < minV[t]) t = i;
//	}
//
//
//	return t;
//}
//
//
//int main( int argc, char** argv )
//{
//	// 加载原始图像和模板图像
//	srcImg = imread( "e:\\张英测试报告\\0312SURF_BRISK\\2\\1_4.jpg", 1 );
//	templImg = imread("e:\\张英测试报告\\0312SURF_BRISK\\2\\2_4.jpg", 1 );
//
//	// 创建显示窗口
//	namedWindow( imageWindow, CV_WINDOW_AUTOSIZE );
//	namedWindow( resultWindow, CV_WINDOW_AUTOSIZE );
//
//	// 创建滑动条
//	char* trackbarLabel = 
//		"Method: \n \
//		0: SQDIFF \n \
//		1: SQDIFF NORMED \n \
//		2: TM CCORR \n \
//		3: TM CCORR NORMED \n \
//		4: TM COEFF \n \
//		5: TM COEFF NORMED";
//	//参数：滑动条名称 显示窗口名称 匹配方法index 滑动条范围 回调函数
//	createTrackbar( trackbarLabel, imageWindow, &matchMethod, maxTrackbar, MatchingMethod );
//
//	double t=cvGetTickCount();
//	MatchingMethod( 0, 0 );
//	t = (double)(cvGetTickCount() - t)/(cvGetTickFrequency()*1000.*1000.);
//	cout<<t<<endl;
//	waitKey(0);
//	return 0;
//}
//
// //函数定义 ///
//void MatchingMethod( int, void* )		//匹配函数
//{
//	// 深拷贝用于显示
//	Mat displayImg;
//	srcImg.copyTo( displayImg );
//
//	// 创建匹配结果图像，为每个模板位置存储匹配结果
//	// 匹配结果图像大小为：(W-w+1)*(H-h+1)
//	int result_cols =  srcImg.cols - templImg.cols + 1;
//	int result_rows = srcImg.rows - templImg.rows + 1;
//	resultImg.create( result_cols, result_rows, CV_32FC1 );
//
//	// 进行匹配并归一化
//	matchTemplate( srcImg, templImg, resultImg, matchMethod );
//	normalize( resultImg, resultImg, 0, 1, NORM_MINMAX, -1, Mat() );
//
//	// 使用minMaxLoc找出最佳匹配
//	double minVal, maxVal;
//	Point minLoc, maxLoc, matchLoc;
//	minMaxLoc( resultImg, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
//
//	// 对于CV_TM_SQDIFF和 CV_TM_SQDIFF_NORMED这两种方法，最小值为最佳匹配；对于别的方法最大值为最佳匹配
//	if( matchMethod  == CV_TM_SQDIFF || matchMethod == CV_TM_SQDIFF_NORMED )
//	{ matchLoc = minLoc; }
//	else  
//	{ matchLoc = maxLoc; }
//
//	// 在原始图像和匹配结果图像中以最佳匹配点为左上角标出最佳匹配框
//	rectangle( displayImg, matchLoc, Point( matchLoc.x + templImg.cols , matchLoc.y + templImg.rows ), Scalar::all(0), 2, 8, 0 ); 
//	rectangle( resultImg, matchLoc, Point( matchLoc.x + templImg.cols , matchLoc.y + templImg.rows ), Scalar::all(0), 2, 8, 0 ); 
//
//	imshow( imageWindow, displayImg );
//	imshow( resultWindow, resultImg );
//
//	return;
//}




//int main( int argc, char** argv )
//{
//  // 加载原始图像和模板图像
//	dir = "e:\\张英测试报告\\0312SURF_BRISK\\1\\";
//	dir1 = dir + "1.jpg";
//	dir2 = dir + "2_2000.jpg";
//	dir1_down = dir + "1_4.jpg";
//	dir2_down = dir + "2_4.jpg";
//	dir_result = dir + "result.jpg";
//
//	srcImg = imread( dir1, 1 );
//  templImg = imread( dir2, 1 );
//
//  scale = srcImg.rows/ templImg.rows ; 
//
////string dir1_down = "e:\\张英测试报告\\0312SURF_BRISK\\3\\1_4.jpg";
////string dir2_down = "e:\\张英测试报告\\0312SURF_BRISK\\3\\2_4.jpg";
//
//double  t_down = cvGetTickCount();
// pyrDown(srcImg, img1_down, Size(srcImg.cols / 2, srcImg.rows / 2));				//1/2降采样
// pyrDown(img1_down, img1_down, Size(img1_down.cols / 2, img1_down.rows / 2));	
//
// pyrDown(templImg, img2_down, Size(templImg.cols / 2, templImg.rows / 2));				//1/2降采样
// pyrDown(img2_down, img2_down, Size(img2_down.cols / 2, img2_down.rows / 2));	
// t_down = (double)(cvGetTickCount() - t_down)/(cvGetTickFrequency()*1000.*1000.);
// cout<<"降采样时间："<<t_down<<endl;
// /*double t1;
// t1=cvGetTickCount();*/
//
// imwrite(dir1_down,img1_down);
// imwrite(dir2_down,img2_down);
//
//  // 创建显示窗口
//  namedWindow( imageWindow, CV_WINDOW_AUTOSIZE );
//  namedWindow( resultWindow, CV_WINDOW_AUTOSIZE );
//  
//  // 创建滑动条
//  char* trackbarLabel = 
//	  "Method: \n \
//	  0: SQDIFF \n \
//	  1: SQDIFF NORMED \n \
//	  2: TM CCORR \n \
//	  3: TM CCORR NORMED \n \
//	  4: TM COEFF \n \
//	  5: TM COEFF NORMED";
//  //参数：滑动条名称 显示窗口名称 匹配方法index 滑动条范围 回调函数
//  createTrackbar( trackbarLabel, imageWindow, &matchMethod, maxTrackbar, MatchingMethod );
//
//  double t = cvGetTickCount();
//  
//  MatchingMethod( 0, 0 );
//  t = (double)(cvGetTickCount() - t)/(cvGetTickFrequency()*1000.*1000.);
//  cout<<"模板匹配时间"<<t<<endl;
//
//  waitKey(0);
//  return 0;
//}
//
///// 函数定义 ///
//void MatchingMethod( int, void* )		//匹配函数
//{
//  // 深拷贝用于显示
//  Mat displayImg;
//  srcImg.copyTo( displayImg );
// 
//  // 创建匹配结果图像，为每个模板位置存储匹配结果
//  // 匹配结果图像大小为：(W-w+1)*(H-h+1)
//  int result_cols =  srcImg.cols - templImg.cols + 1;
//  int result_rows = srcImg.rows - templImg.rows + 1;
//  resultImg.create( result_cols, result_rows, CV_32FC1 );
//
//  // 进行匹配并归一化
// // matchTemplate( srcImg, templImg, resultImg, matchMethod );
//   matchTemplate( img1_down, img2_down, resultImg, matchMethod );
//  normalize( resultImg, resultImg, 0, 1, NORM_MINMAX, -1, Mat() );
//
//  // 使用minMaxLoc找出最佳匹配
//  double minVal, maxVal;
//  Point minLoc, maxLoc, matchLoc;
//  minMaxLoc( resultImg, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
//
//  // 对于CV_TM_SQDIFF和 CV_TM_SQDIFF_NORMED这两种方法，最小值为最佳匹配；对于别的方法最大值为最佳匹配
//  if( matchMethod  == CV_TM_SQDIFF || matchMethod == CV_TM_SQDIFF_NORMED )
//    { matchLoc = minLoc; }
//  else  
//    { matchLoc = maxLoc; }
//
//  // 在原始图像和匹配结果图像中以最佳匹配点为左上角标出最佳匹配框
// /* rectangle( displayImg, matchLoc, Point( matchLoc.x + templImg.cols , matchLoc.y + templImg.rows ), Scalar(0,0,255), 2, 8, 0 ); 
//  rectangle( resultImg, matchLoc, Point( matchLoc.x + templImg.cols , matchLoc.y + templImg.rows ), Scalar::all(0), 2, 8, 0 ); 
//*/
//  matchLoc *= scale;	//因为是1/4降采样
//  cout<<"左上角坐标："<<matchLoc.x<<","<<matchLoc.y<<endl;
//  rectangle( displayImg, matchLoc, Point( matchLoc.x + templImg.cols , matchLoc.y + templImg.rows ),Scalar(0,0,255)  , 2, 8, 0 ); 
// 
//  imwrite(dir_result,displayImg);
//  rectangle( resultImg, matchLoc, Point( matchLoc.x + templImg.cols , matchLoc.y + templImg.rows ), Scalar::all(0), 2, 8, 0 ); 
// //  imwrite("e:\\张英测试报告\\0312SURF_BRISK\\1\\result_huidu.jpg",resultImg);
//
//  imshow( imageWindow, displayImg );
//  imshow( resultWindow, resultImg );
//
//  return;
//}

