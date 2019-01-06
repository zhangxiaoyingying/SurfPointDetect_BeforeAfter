////// sift.cpp : �������̨Ӧ�ó������ڵ㡣

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"//��Ϊ���������Ѿ�������opencv��Ŀ¼�����԰��䵱���˱���Ŀ¼һ��
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp" //��opencv��ʹ��sift������Ҫ��ͷ�ļ�"opencv2/nonfree/nonfree.hpp"��
									  //ע������Ƿ���ѵģ�Sift�㷨��ר��Ȩ���ڸ��ױ��Ǵ�ѧ���������ҵ�����ʹ�ã������з��ա�
#include "opencv2/legacy/legacy.hpp"


#include "opencv2/imgproc/imgproc.hpp"



using namespace cv;
using namespace std;
Mat di;


////
//////С����
//////int main()
//////{
//////	IplImage* img=cvLoadImage("E:\\test\\1.tif");//ͼ���СΪ940M��tif���򿪳���!!!!
//////	//IplImage* img=cvLoadImage("E:\\test\\zhang.tif");//ͼ���СΪ130K����ʽΪtif������ֱ�Ӵ�
//////	//IplImage* img=cvLoadImage("E:\\fusiontest\\2.bmp");//ͼ���СΪ769K����ʽΪbmp,����ֱ�Ӵ�
//////	cvNamedWindow("Example1",1);
//////	cvShowImage("Example1",img);
//////	cvWaitKey();
//////	cvReleaseImage(&img);
//////	cvDestroyWindow("Example1");
//////}
////
////
/////*
////////Harris�ǵ����㷨
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
////         cvCornerHarris(grayImage,pHarrisImg,block_size,7,0.04);//Harris�ǵ���
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
//////// sift_test.cpp : �������̨Ӧ�ó������ڵ㡣
//////int main(int argc,char* argv[])
//////{
//////   /* Mat image01=imread("e:\\test\\lena.bmp");  
//////    Mat image02=imread("e:\\test\\lena1.bmp");  
//////    Mat img_1,img_2;  
//////    GaussianBlur(image01,img_1,Size(3,3),0.5);  
//////    GaussianBlur(image02,img_2,Size(3,3),0.5); */
//////
//////   Mat img_1=imread("E:\\test\\Lena.bmp");//�궨��ʱCV_LOAD_IMAGE_GRAYSCALE=0��Ҳ���Ƕ�ȡ�Ҷ�ͼ��
//////   Mat img_2=imread("E:\\test\\Lena1.bmp");//һ��Ҫ�ǵ�����·����б�߷�������Matlab�������෴��
//////
//////
//////	
//////
//////if(!img_1.data || !img_2.data)//�������Ϊ��
//////    {
//////        cout<<"opencv error"<<endl;
//////		waitKey(100);
//////		system("pause");
//////return -1;
//////    }
//////    cout<<"open right"<<endl;
//////	
//////
////////��һ������SIFT���Ӽ��ؼ���
//////	//SurfFeatureDetector detector();
//////    SiftFeatureDetector detector(200);//���캯�������ڲ�Ĭ�ϵģ�����ԽС����Խ�٣���
//////    std::vector<KeyPoint> keypoints_1,keypoints_2;//����2��ר���ɵ���ɵĵ����������洢������
//////
//////    detector.detect(img_1,keypoints_1);//��img_1ͼ���м�⵽��������洢��������keypoints_1��
//////    detector.detect(img_2,keypoints_2);//ͬ��
//////
////////��ͼ���л���������
//////    Mat img_keypoints_1,img_keypoints_2;
//////
//////    drawKeypoints(img_1,keypoints_1,img_keypoints_1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);//���ڴ��л���������
//////    drawKeypoints(img_2,keypoints_2,img_keypoints_2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
//////
//////    imshow("sift_keypoints_1",img_keypoints_1);//��ʾ������
//////    imshow("sift_keypoints_2",img_keypoints_2);
//////
////////������������
//////    SiftDescriptorExtractor extractor;//���������Ӷ���
//////
//////    Mat descriptors_1,descriptors_2;//������������ľ���
//////
//////    extractor.compute(img_1,keypoints_1,descriptors_1);//������������
//////    extractor.compute(img_2,keypoints_2,descriptors_2);
//////
////////��burte force����ƥ����������
//////    BruteForceMatcher<L2<float>>matcher;//����һ��burte force matcher����
//////    vector<DMatch>matches;
//////    matcher.match(descriptors_1,descriptors_2,matches);
//////
////////����ƥ���߶�
//////    Mat img_matches;
//////    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_matches);//��ƥ������Ľ�������ڴ�img_matches��
//////
////////��ʾƥ���߶�
//////    imshow("sift_Matches",img_matches);//��ʾ�ı���ΪMatches
//////    waitKey(0);
//////return 0;
//////system("pause");
//////}
////
////
/////*
// surf_test.cpp : �������̨Ӧ�ó������ڵ㡣

#include <iostream>
#include <fstream>
int main(int argc,char* argv[])
{
    //Mat img_1=imread("E:\\test\\111.png");//�궨��ʱCV_LOAD_IMAGE_GRAYSCALE=0��Ҳ���Ƕ�ȡ�Ҷ�ͼ��
    //Mat img_2=imread("E:\\test\\222.png");//һ��Ҫ�ǵ�����·����б�߷�������Matlab�������෴��
////	Mat img_1=imread("e:\\��Ӣ���Ա���\\0303��ת����\\2_UP.jpg");
//	Mat img_2=imread("e:\\��Ӣ���Ա���\\0303��ת����\\1_1000.jpg");//ʹ��Photoshop����͸�Ӵ����ͼ����׼������





	


	Mat img_1=imread("e:\\test_t\\A2.jpg");	//��ͼ
	Mat img_2=imread("e:\\test_t\\B2.jpg");	//��ͼ

if(!img_1.data || !img_2.data)//�������Ϊ��
    {
        cout<<"opencv error"<<endl;
return -1;
    }
    cout<<"open right"<<endl;

//��һ������SURF���Ӽ��ؼ���

	  int minHessian=5000;//400��800��ʵ�����õĽ϶��ֵ
    SurfFeatureDetector detector(minHessian);//����Ϊͼ��Hessian�����б�ʽ����ֵ�����ֵ������Խ�󣬱�ʾ������Խ�٣�Խ�ȶ�

    std::vector<KeyPoint> keypoints_1,keypoints_2;//����2��ר���ɵ���ɵĵ����������洢������

    detector.detect(img_1,keypoints_1);//��img_1ͼ���м�⵽��������洢��������keypoints_1��
    detector.detect(img_2,keypoints_2);//ͬ��

//��ͼ���л���������
    Mat img_keypoints_1,img_keypoints_2;

    drawKeypoints(img_1,keypoints_1,img_keypoints_1,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(img_2,keypoints_2,img_keypoints_2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);

    imshow("surf_keypoints_1",img_keypoints_1);
    imshow("surf_keypoints_2",img_keypoints_2);

//������������
    SurfDescriptorExtractor extractor;//���������Ӷ���

    Mat descriptors_1,descriptors_2;//������������ľ���

    extractor.compute(img_1,keypoints_1,descriptors_1);
    extractor.compute(img_2,keypoints_2,descriptors_2);

//��burte force����ƥ����������
    BruteForceMatcher<L2<float>>matcher;//����һ��burte force matcher����
    vector<DMatch>matches;
    matcher.match(descriptors_1,descriptors_2,matches);

//����ƥ���߶�
    Mat img_matches;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_matches);//��ƥ������Ľ�������ڴ�img_matches��

	ofstream outFile1,outFile2;//��������
	outFile1.open("descriptor1.txt");//���ı�
	outFile2.open("descriptor2.txt");
	//������ͼ���ƥ���������
	for (int n = 0; n < 64; n++)
	{
		outFile1 << descriptors_1.at<int>(matches[0].queryIdx,n) << endl;
	}
	//����ұ�ͼ���ƥ���������
	for (int n = 0; n < 64; n++)
	{
		outFile2 << descriptors_2.at<int>(matches[0].trainIdx, n) << endl;
	}
	outFile1.close();//�ر��ı�
	outFile2.close();

	
	//��ʾƥ���߶�
    imshow("surf_Matches",img_matches);//��ʾ�ı���ΪMatches
    waitKey(0);
    return 0;
	system("pause");
}

////
////
/////*
//////͸�ӱ任���루opencv Դ���룩
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
////	//�ܼ�͸�ӱ任����3*3�����4���������
////	srcTri[0].x = 0;
////	srcTri[0].y = 0;
////	srcTri[1].x = src->width - 1;
////	srcTri[1].y = 0;
////	srcTri[2].x = 0;
////	srcTri[2].y = src->height - 1;
////	srcTri[3].x = src->width - 1;
////	srcTri[3].y = src->height - 1;
////
////	//�˵�ϵ��ȷ����Ӧ���λ�ã���˿����趨
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
////	//����ͼ��
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
////*********opencv stitch��ƴ��,�ٶ��ر���********//
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
//    // ʹ��stitch��������ƴ��
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
//    // ��ʾԴͼ�񣬺ͽ��ͼ��
//    imshow("ȫ��ͼ��", pano);
//    if (waitKey() == 27)
//        return 0;
//}
//*/
//
//
//
//
//////surf ��ͼ��ƴ��
////int main(int argc,char *argv[])    
////{    
////	Mat image01=imread("E:\\test\\xiaotu\\33.png");   
////    Mat image02=imread("E:\\test\\xioatu\\44.png");    //��ʹ��psʹ����ת�Ƕȹ����ͼ����׼�ͻ�����⣬�㷨�����ã��������б�ĽǶ�û��ȷ��
////                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
////    imshow("�ο�ͼ",image01);  
////   imshow("����ͼ",image02);  
////  
////    //�Ҷ�ͼת��  
////    Mat image1,image2;    
////    cvtColor(image01,image1,CV_RGB2GRAY);  
////   cvtColor(image02,image2,CV_RGB2GRAY);  
//// 
////  
////   //��ȡ������    
////    SurfFeatureDetector surfDetector(800);  // ����������ֵ  
////    vector<KeyPoint> keyPoint1,keyPoint2;    
////    surfDetector.detect(image1,keyPoint1);    
////   surfDetector.detect(image2,keyPoint2);    
////  
////    //������������Ϊ�±ߵ�������ƥ����׼��    
////    SurfDescriptorExtractor SurfDescriptor;    
////   Mat imageDesc1,imageDesc2;    
////    SurfDescriptor.compute(image1,keyPoint1,imageDesc1);    
////    SurfDescriptor.compute(image2,keyPoint2,imageDesc2);      
////  
////    //���ƥ�������㣬����ȡ�������     
////    FlannBasedMatcher matcher;  
////   vector<DMatch> matchePoints;    
////    matcher.match(imageDesc1,imageDesc2,matchePoints,Mat());  
////   sort(matchePoints.begin(),matchePoints.end()); //����������    
////  
////    //��ȡ����ǰN��������ƥ��������  
////    vector<Point2f> imagePoints1,imagePoints2;      
////    for(int i=0;i<10;i++)  
////    {         
////      imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);       
////        imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);       
////   }  
////    //��ȡͼ��1��ͼ��2��ͶӰӳ����� �ߴ�Ϊ3*3  
////    Mat homo=findHomography(imagePoints1,imagePoints2,CV_RANSAC);  
////   ////Ҳ����ʹ��getPerspectiveTransform�������͸�ӱ任���󣬲���Ҫ��ֻ����4���㣬Ч���Բ�  
////    //Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);     
////        cout<<"�任����Ϊ��\n"<<homo<<endl<<endl; //���ӳ�����  
////    //ͼ����׼  
////   Mat imageTransform1,imageTransform2;  
//// 
////  warpPerspective(image01,imageTransform1,homo,Size(image01.cols,image01.rows));
////    imshow("����͸�Ӿ���任��",imageTransform1);  
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
////********����������л���ƥ�����ͳ�ơ���ʾ�ȿ��Խ������****//
////**************************************************//
//
////int main()
////{
////	initModule_nonfree();//��ʼ��ģ�飬ʹ��SIFT��SURFʱ�õ� 
////	Ptr<FeatureDetector> detector = FeatureDetector::create( "SURF" );//����SIFT������������ɸĳ�SURF/ORB
////	Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SURF" );//���������������������ɸĳ�SURF/ORB
////	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );//��������ƥ����  
////	if( detector.empty() || descriptor_extractor.empty() )  
////		cout<<"fail to create detector!";  
////
////	//����ͼ��  
////	Mat img1 = imread("E:\\test\\xiaotu\\44.png");	//Сͼ������
////	Mat img2 = imread("E:\\test\\xiaotu\\1.tif");  //��ͼ���ο�ͼ
////	  
////	//��������  
////	double t = getTickCount();//��ǰ�δ���  
////	vector<KeyPoint> m_LeftKey,m_RightKey;  
////	detector->detect( img1, m_LeftKey );//���img1�е�SIFT�����㣬�洢��m_LeftKey��  
////	detector->detect( img2, m_RightKey );  
////	cout<<"ͼ��1���������:"<<m_LeftKey.size()<<endl;  
////	cout<<"ͼ��2���������:"<<m_RightKey.size()<<endl;  
////	
////	
////	//����������������������Ӿ��󣬼�������������  
////	Mat descriptors1,descriptors2;  
////	descriptor_extractor->compute( img1, m_LeftKey, descriptors1 );  
////	descriptor_extractor->compute( img2, m_RightKey, descriptors2 );  
////	
////	t = ((double)getTickCount() - t)/getTickFrequency();  
////	cout<<"SIFT�㷨��ʱt��"<<t<<"��"<<endl;  
////	//cout<<"ͼ��1�������������С��"<<descriptors1.size()  
////	//	<<"����������������"<<descriptors1.rows<<"��ά����"<<descriptors1.cols<<endl;  
////	//cout<<"ͼ��2�������������С��"<<descriptors2.size()  
////	//	<<"����������������"<<descriptors2.rows<<"��ά����"<<descriptors2.cols<<endl;  
////
////	//����������  
////	Mat img_m_LeftKey,img_m_RightKey;  
////	drawKeypoints(img1,m_LeftKey,img_m_LeftKey,Scalar::all(-1),0);  
////	drawKeypoints(img2,m_RightKey,img_m_RightKey,Scalar::all(-1),0);  
////	//imshow("Src1",img_m_LeftKey);  
////	//imshow("Src2",img_m_RightKey);  
////
////	//����ƥ��  
////	vector<DMatch> matches;//ƥ����  
////	descriptor_matcher->match( descriptors1, descriptors2, matches );//ƥ������ͼ�����������  
////	cout<<"Match������"<<matches.size()<<endl;  
////
////	//����ƥ�����о����������Сֵ  
////	//������ָ���������������ŷʽ���룬�������������Ĳ��죬ֵԽС��������������Խ�ӽ�  
////	double max_dist = 0;  
////	double min_dist = 100;  
////	for(int i=0; i<matches.size(); i++)  
////	{  
////		double dist = matches[i].distance;  
////		if(dist < min_dist) min_dist = dist;  
////		if(dist > max_dist) max_dist = dist;  
////	}  
////	cout<<"�����룺"<<max_dist<<endl;  
////	cout<<"��С���룺"<<min_dist<<endl;  
////
////	//ɸѡ���Ϻõ�ƥ���  
////	vector<DMatch> goodMatches;  
////	for(int i=0; i<matches.size(); i++)  
////	{  
////		if(matches[i].distance < 0.2 * max_dist)  
////		{  
////			goodMatches.push_back(matches[i]);  
////		}  
////	}  
////	cout<<"goodMatch������"<<goodMatches.size()<<endl;  
////
////	//����ƥ����  
////	Mat img_matches;  
////	//��ɫ���ӵ���ƥ���������ԣ���ɫ��δƥ���������  
////	drawMatches(img1,m_LeftKey,img2,m_RightKey,goodMatches,img_matches,  
////		Scalar::all(-1)/*CV_RGB(255,0,0)*/,CV_RGB(0,255,0),Mat(),2);  
////
////	imshow("MatchSIFT",img_matches);  
////	IplImage result=img_matches;
////
////	waitKey(0);
////
////
////	//RANSACƥ�����
////	vector<DMatch> m_Matches=goodMatches;
////	// ����ռ�
////	int ptCount = (int)m_Matches.size();
////	Mat p1(ptCount, 2, CV_32F);
////	Mat p2(ptCount, 2, CV_32F);
////
////	// ��Keypointת��ΪMat
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
////	// ��RANSAC��������F
////	Mat m_Fundamental;
////	vector<uchar> m_RANSACStatus;       // ����������ڴ洢RANSAC��ÿ�����״̬
////	findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);
////
////	// ����Ұ�����
////
////	int OutlinerCount = 0;
////	for (int i=0; i<ptCount; i++)
////	{
////		if (m_RANSACStatus[i] == 0)    // ״̬Ϊ0��ʾҰ��
////		{
////			OutlinerCount++;
////		}
////	}
////	int InlinerCount = ptCount - OutlinerCount;   // �����ڵ�
////	cout<<"�ڵ���Ϊ��"<<InlinerCount<<endl;
////
////
////	// �������������ڱ����ڵ��ƥ���ϵ
////	vector<Point2f> m_LeftInlier;
////	vector<Point2f> m_RightInlier;
////	vector<DMatch> m_InlierMatches;
////
////	m_InlierMatches.resize(InlinerCount);
////	m_LeftInlier.resize(InlinerCount);
////	m_RightInlier.resize(InlinerCount);
////	InlinerCount=0;
////	float inlier_minRx=img1.cols;        //���ڴ洢�ڵ�����ͼ��С�����꣬�Ա�����ں�
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
////			if(m_RightInlier[InlinerCount].x<inlier_minRx) inlier_minRx=m_RightInlier[InlinerCount].x;   //�洢�ڵ�����ͼ��С������
////
////			InlinerCount++;
////		}
////	}
////
////	// ���ڵ�ת��ΪdrawMatches����ʹ�õĸ�ʽ
////	vector<KeyPoint> key1(InlinerCount);
////	vector<KeyPoint> key2(InlinerCount);
////	KeyPoint::convert(m_LeftInlier, key1);
////	KeyPoint::convert(m_RightInlier, key2);
////
////	// ��ʾ����F������ڵ�ƥ��
////	Mat OutImage;
////	drawMatches(img1, key1, img2, key2, m_InlierMatches, OutImage);
////	cvNamedWindow( "Match features", 1);
////	cvShowImage("Match features", &IplImage(OutImage));
////	waitKey(0);
////
////	cvDestroyAllWindows();
////
////	//����H���Դ洢RANSAC�õ��ĵ�Ӧ����
////	Mat H = findHomography( m_LeftInlier, m_RightInlier, RANSAC );
////
////	//�洢��ͼ�Ľǣ�����任����ͼλ��
////	std::vector<Point2f> obj_corners(4);
////	obj_corners[0] = Point(0,0); obj_corners[1] = Point( img1.cols, 0 );
////	obj_corners[2] = Point( img1.cols, img1.rows ); obj_corners[3] = Point( 0, img1.rows );
////	std::vector<Point2f> scene_corners(4);
////	perspectiveTransform( obj_corners, scene_corners, H);
////
////	//�����任��ͼ��λ��
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
//// 	int drift = scene_corners[1].x;                                                        //����ƫ����
//// 
//// 	//�½�һ������洢��׼���Ľǵ�λ��
//// 	int width = int(max(abs(scene_corners[1].x), abs(scene_corners[2].x)));
//// //	int height= img1.rows;                                                                  //���ߣ�int height = int(max(abs(scene_corners[2].y), abs(scene_corners[3].y)));
//// 	int height= img2.rows;    
////	float origin_x=0,origin_y=0;
//// 	if(scene_corners[0].x<0) {
//// 		if (scene_corners[3].x<0) origin_x+=min(scene_corners[0].x,scene_corners[3].x);
//// 		else origin_x+=scene_corners[0].x;}
//// 	width-=int(origin_x);
//// 	if(scene_corners[0].y<0) {
//// 		if (scene_corners[1].y) origin_y+=min(scene_corners[0].y,scene_corners[1].y);
//// 		else origin_y+=scene_corners[0].y;}
//// 	//��ѡ��height-=int(origin_y);
//// 	Mat imageturn=Mat::zeros(width,height,img1.type());
//// 
//// 	//��ȡ�µı任����ʹͼ��������ʾ
//// 	for (int i=0;i<4;i++) {scene_corners[i].x -= origin_x; } 	//��ѡ��scene_corners[i].y -= (float)origin_y; }
//// 	Mat H1=getPerspectiveTransform(obj_corners, scene_corners);
//// 
//// 	//����ͼ��任����ʾЧ��
////	warpPerspective(img1,imageturn,H1,Size(width,height));	
//// 	imshow("image_Perspective", imageturn);
//// 	waitKey(0);
//// 
//// 
//// 	//ͼ���ں�
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
//////****ͼ��ƴ���㷨����ֱ�ӷ��ڰ�Ȩ�ص���ͼ1��ͼ2���ص�����*****//
//////**********************************************//
////
////
//////����ԭʼͼ���λ�ھ�������任����Ŀ��ͼ���϶�Ӧλ��
////Point2f getTransformPoint(const Point2f originalPoint,const Mat &transformMaxtri);
////
////int main()  
////{  
////	Mat image02=imread("E:\\test\\44.png");  
////	Mat image01=imread("E:\\test\\33.png");
////		imshow("ƴ��ͼ��1",image01);
////	    imshow("ƴ��ͼ��2",image02);
////
////
////	//�Ҷ�ͼת��
////	Mat image1,image2;  
////	cvtColor(image01,image1,CV_RGB2GRAY);
////	cvtColor(image02,image2,CV_RGB2GRAY);
////
////
////
////	//��ȡ������  
////	SiftFeatureDetector siftDetector(800);  // ����������ֵ
////	vector<KeyPoint> keyPoint1,keyPoint2;  
////	siftDetector.detect(image1,keyPoint1);  
////	siftDetector.detect(image2,keyPoint2);	
////
////	//������������Ϊ�±ߵ�������ƥ����׼��  
////	SiftDescriptorExtractor siftDescriptor;  
////	Mat imageDesc1,imageDesc2;  
////	siftDescriptor.compute(image1,keyPoint1,imageDesc1);  
////	siftDescriptor.compute(image2,keyPoint2,imageDesc2);	
////
////	//���ƥ�������㣬����ȡ�������  	
////	FlannBasedMatcher matcher;
////	vector<DMatch> matchePoints;  
////	matcher.match(imageDesc1,imageDesc2,matchePoints,Mat());
////	sort(matchePoints.begin(),matchePoints.end()); //����������	
////	//��ȡ����ǰN��������ƥ��������
////	vector<Point2f> imagePoints1,imagePoints2;
////	for(int i=0;i<10;i++)
////	{		
////		imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);//matchePoints[i].queryIdx�����ŵ�һ��ͼƬƥ������ţ�().pt��ʾ�㣻pt.xλx����
////		imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);	//matchePoints[i].trainIdx����ڶ��ţ�ѵ����	
////	}
////
////	//��ȡͼ��1��ͼ��2��ͶӰӳ����󣬳ߴ�Ϊ3*3
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
////	//����һ������Ƕ�homo�����������
////	Mat adjustMat=(Mat_<double>(3,3)<<1.0 ,0,image01.cols,0,1.0,image01.rows,0,0,1.0 );//���ƫ��image01.cols����;���������õڶ��е�����������������Ϊ������H�����ֵΪ��������ͼ�����ΪԽ��ȱʧ
////	////Mat adjustMat=(Mat_<double>(3,3)<<1.0,0,35,0,1.0,65,0,0,1.0); //��������ʱҲ����ֱ��ʹ������
////	Mat adjustHomo= adjustMat*homo;
////
////	cout<<"�任����Ϊ��\n"<<homo<<endl<<endl; //���ӳ�����  
////	cout<<"�任����Ϊ��\n"<<adjustHomo<<endl<<endl; //���ӳ����� 
////
////
////	//��ȡ��ǿ��Ե���ԭʼͼ�񡢾���任��ͼ���ϵĶ�Ӧλ�ã�����ͼ��ƴ�ӵ�Ķ�λ
////	Point2f originalLinkPoint,targetLinkPoint,basedImagePoint;
////	originalLinkPoint=keyPoint1[matchePoints[0].queryIdx].pt;//ƥ�����ͼ1�е�λ��
////	targetLinkPoint=getTransformPoint(originalLinkPoint,adjustHomo);//ƥ����ڱ任����͸��ͼ��λ��	//getTransformPoint���õ�ԭʼͼ�������㰴��adjustHomo����任֮�󣬵Ľ���㣬����targetLinkPoint
////	basedImagePoint=keyPoint2[matchePoints[0].trainIdx].pt;//ƥ�����ͼ2�е�λ��
////
////	//ͼ����׼
////	Mat imageTransform1;
////	warpPerspective(image01,imageTransform1,adjustMat*homo,Size(image02.cols+image01.cols+200,+image01.rows+image02.rows+200));
////	
////	imshow("����͸�Ӿ���任��",imageTransform1);
////	
////
////	////��һ�ַ���
////	////����ǿƥ����λ�ô��νӣ���ǿƥ��������ͼ1���Ҳ���ͼ2������ֱ���滻ͼ���νӲ��ã�������ͻ��
////	//Mat ROIMat=image02(Rect(Point(basedImagePoint.x,0),Point(image02.cols,image02.rows)));	
////	//ROIMat.copyTo(Mat(imageTransform1,Rect(targetLinkPoint.x,0,image02.cols-basedImagePoint.x+1,image02.rows)));
////	
////
////	//�ڶ��ַ���
////	//����ǿƥ��������ص���������ۼӣ����ν��ȶ����ɣ�����ͻ��
////	Mat image1Overlap,image2Overlap; //����ͼ1��ͼ2���ص�����	
////	image1Overlap=imageTransform1(Rect(Point(targetLinkPoint.x-basedImagePoint.x,0),Point(targetLinkPoint.x,image02.rows)));  
////    image2Overlap=image02(Rect(0,0,image1Overlap.cols,image1Overlap.rows));  
////	
////
////	Mat image1ROICopy=image1Overlap.clone();  //����һ��ͼ1���ص�����
////	for(int i=0;i<image1Overlap.rows;i++)
////	{
////		for(int j=0;j<image1Overlap.cols;j++)
////		{
////			double weight;
////			weight=(double)j/image1Overlap.cols;  //�����ı���ı�ĵ���ϵ��
////			image1Overlap.at<Vec3b>(i,j)[0]=(1-weight)*image1ROICopy.at<Vec3b>(i,j)[0]+weight*image2Overlap.at<Vec3b>(i,j)[0];
////			image1Overlap.at<Vec3b>(i,j)[1]=(1-weight)*image1ROICopy.at<Vec3b>(i,j)[1]+weight*image2Overlap.at<Vec3b>(i,j)[1];
////			image1Overlap.at<Vec3b>(i,j)[2]=(1-weight)*image1ROICopy.at<Vec3b>(i,j)[2]+weight*image2Overlap.at<Vec3b>(i,j)[2];
////			//Vec3b������ģ���ࡣ��ʾÿһ��Vec3b������,���Դ洢3��char��һ����RGB
////	
////	
////		}
////	}
////
////   Mat ROIMat=image02(Rect(Point(image1Overlap.cols,0),Point(image02.cols,image02.rows)));  //ͼ2�в��غϵĲ���  
//// //  ROIMat.copyTo(Mat(imageTransform1,Rect(targetLinkPoint.x,0, ROIMat.cols,image02.rows))); //���غϵĲ���ֱ���ν���ȥ
//// ROIMat.copyTo(Mat(imageTransform1,Rect(targetLinkPoint.x,image01.rows,ROIMat.cols,ROIMat.rows)));//�����Խ��ͼ��͸�ӱ任��ȱʧ�����
////
////
////	
////
////	//namedWindow("ƴ�ӽ��",0);
////	imshow("ƴ�ӽ��",imageTransform1);	
////	
////	waitKey();  
////	return 0;  
////}
////
//////����ԭʼͼ���λ�ھ�������任����Ŀ��ͼ���϶�Ӧλ��
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
//////FAST�ǵ����㷨
////int main()
////{
////	Mat frame=imread("E:\\test\\kuailejiazu.jpg", 1);
////	double t = getTickCount();//��ǰ�δ���
////	std::vector<KeyPoint> keyPoints;
////	FastFeatureDetector fast(100);	// ������ֵΪ50
////
////	fast.detect(frame, keyPoints);
////	drawKeypoints(frame, keyPoints, frame, Scalar(0,0,255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
////	
////	t = ((double)getTickCount() - t)/getTickFrequency();
////	cout<<"�㷨��ʱ��"<<t<<"��"<<endl;
////
////	imshow("FAST������", frame);
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
///*************�������**************/
///*********************************/
////int main()
////{
////	//�������ʾԭͼ
////	string image_name="E:\\test\\2.jpg";
////	Mat src=imread(image_name);
////	imshow("src",src);
////	//ת���ɻҶ�ͼ
////	Mat gray(src.size(),CV_8U);
////	cvtColor(src,gray,CV_BGR2GRAY);
////	imshow("gray",gray);
////	//������ֵ��ת���ɶ�ֵͼ��
////	threshold(gray,gray,128,255,THRESH_BINARY);
////	imshow("binary",gray);
////	//�������
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
////	src = imread("E:\\��Ӣ���Ա���\\0303��ת����\\2.jpg", 1);  
////	if (src.empty())  
////	{  
////		printf("cannot load!");  
////		return -1;  
////	}  
////	namedWindow("ԭͼ");  
////	imshow("ԭͼ", src);  
////
////	//�ϲ���  
////	pyrUp(src, dst_up, Size(src.cols * 2, src.rows * 2));  
////	imwrite("E:\\��Ӣ���Ա���\\0303��ת����\\2_pyrUP.jpg",dst_up);
////	namedWindow("�ϲ���");  
////	imshow("�ϲ���", dst_up);  
////
////	//�²���  
////	pyrDown(src, dst_down, Size(src.cols / 2, src.rows / 2));  
////	namedWindow("�²���");  
////	imshow("�²���", dst_down);  
////
////	//��˹��ͬ  
////	Mat g1, g2, dogImg;  
////	GaussianBlur(src, g1, Size(5, 5), 0, 0);  
////	GaussianBlur(src, g2, Size(5, 5), 11, 11);  
////	subtract(g1, g2, dogImg);  
////	//��һ����ʾ  
////	normalize(dogImg, dogImg, 255, 0, NORM_MINMAX);//��Ϊ����ͼ��Ĳ�ֵ�϶����󣬿�����һƬ�ڣ�������Ҫ��������չ��0~255�����䡣  
////	namedWindow("��˹��ͬ");  
////	imshow("��˹��ͬ", dogImg);  
////
////	waitKey(0);  
////	return 0;  
////}  
//
//int main(int argc, char** argv) 
//{ 
//	Mat img_1 = imread("e:\\��Ӣ���Ա���\\0130������׼�������ͼ\\1_4\\2.jpg"); 
//	Mat img_2 = imread("e:\\��Ӣ���Ա���\\0130������׼�������ͼ\\1_4\\2_30.jpg");
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



/// ȫ�ֱ��� ///
Mat srcImg;			//ԭʼͼ��
Mat templImg;		//ģ��ͼ��
Mat resultImg;		//ƥ����ͼ��
Mat img1_down;
Mat img2_down;
string dir,dir1,dir2,dir3,dir1_down,dir2_down,dir_result;

const char* imageWindow = "Source Image";		//ԭʼͼ����ʾ����
const char* resultWindow = "Result Window";		//ƥ����ͼ����ʾ����

int matchMethod;		//ƥ�䷽��index
int maxTrackbar = 5;	//��������Χ����ƥ�䷽��������Ӧ��

void MatchingMethod( int, void* );		//ƥ�亯��
double scale;

////��˵�Ǿ��г߶������Ե�ģ��ƥ��Ľ���������ʵ��������
int app(vector<double> minV);  

//int main( )
//{
//	string dir = "e:\\��Ӣ���Ա���\\0312SURF_BRISK\\1\\";
//	dir2 = dir + "2_4.jpg";
//	dir1 = dir + "1_4.jpg";
//	dir3 = dir;
//	dir_result =dir3 + "sf_result.jpg";
//	Mat img = imread(dir1);       //ԭͼ��
//	Mat temp = imread(dir2);     //ģ��
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
//	cout<<"�߶�����ģ��ƥ��ʱ�䣺"<<t<<endl;
//
//	imwrite(dir_result,img);
//
//	imshow("���",img);
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
//	// ����ԭʼͼ���ģ��ͼ��
//	srcImg = imread( "e:\\��Ӣ���Ա���\\0312SURF_BRISK\\2\\1_4.jpg", 1 );
//	templImg = imread("e:\\��Ӣ���Ա���\\0312SURF_BRISK\\2\\2_4.jpg", 1 );
//
//	// ������ʾ����
//	namedWindow( imageWindow, CV_WINDOW_AUTOSIZE );
//	namedWindow( resultWindow, CV_WINDOW_AUTOSIZE );
//
//	// ����������
//	char* trackbarLabel = 
//		"Method: \n \
//		0: SQDIFF \n \
//		1: SQDIFF NORMED \n \
//		2: TM CCORR \n \
//		3: TM CCORR NORMED \n \
//		4: TM COEFF \n \
//		5: TM COEFF NORMED";
//	//���������������� ��ʾ�������� ƥ�䷽��index ��������Χ �ص�����
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
// //�������� ///
//void MatchingMethod( int, void* )		//ƥ�亯��
//{
//	// ���������ʾ
//	Mat displayImg;
//	srcImg.copyTo( displayImg );
//
//	// ����ƥ����ͼ��Ϊÿ��ģ��λ�ô洢ƥ����
//	// ƥ����ͼ���СΪ��(W-w+1)*(H-h+1)
//	int result_cols =  srcImg.cols - templImg.cols + 1;
//	int result_rows = srcImg.rows - templImg.rows + 1;
//	resultImg.create( result_cols, result_rows, CV_32FC1 );
//
//	// ����ƥ�䲢��һ��
//	matchTemplate( srcImg, templImg, resultImg, matchMethod );
//	normalize( resultImg, resultImg, 0, 1, NORM_MINMAX, -1, Mat() );
//
//	// ʹ��minMaxLoc�ҳ����ƥ��
//	double minVal, maxVal;
//	Point minLoc, maxLoc, matchLoc;
//	minMaxLoc( resultImg, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
//
//	// ����CV_TM_SQDIFF�� CV_TM_SQDIFF_NORMED�����ַ�������СֵΪ���ƥ�䣻���ڱ�ķ������ֵΪ���ƥ��
//	if( matchMethod  == CV_TM_SQDIFF || matchMethod == CV_TM_SQDIFF_NORMED )
//	{ matchLoc = minLoc; }
//	else  
//	{ matchLoc = maxLoc; }
//
//	// ��ԭʼͼ���ƥ����ͼ���������ƥ���Ϊ���ϽǱ�����ƥ���
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
//  // ����ԭʼͼ���ģ��ͼ��
//	dir = "e:\\��Ӣ���Ա���\\0312SURF_BRISK\\1\\";
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
////string dir1_down = "e:\\��Ӣ���Ա���\\0312SURF_BRISK\\3\\1_4.jpg";
////string dir2_down = "e:\\��Ӣ���Ա���\\0312SURF_BRISK\\3\\2_4.jpg";
//
//double  t_down = cvGetTickCount();
// pyrDown(srcImg, img1_down, Size(srcImg.cols / 2, srcImg.rows / 2));				//1/2������
// pyrDown(img1_down, img1_down, Size(img1_down.cols / 2, img1_down.rows / 2));	
//
// pyrDown(templImg, img2_down, Size(templImg.cols / 2, templImg.rows / 2));				//1/2������
// pyrDown(img2_down, img2_down, Size(img2_down.cols / 2, img2_down.rows / 2));	
// t_down = (double)(cvGetTickCount() - t_down)/(cvGetTickFrequency()*1000.*1000.);
// cout<<"������ʱ�䣺"<<t_down<<endl;
// /*double t1;
// t1=cvGetTickCount();*/
//
// imwrite(dir1_down,img1_down);
// imwrite(dir2_down,img2_down);
//
//  // ������ʾ����
//  namedWindow( imageWindow, CV_WINDOW_AUTOSIZE );
//  namedWindow( resultWindow, CV_WINDOW_AUTOSIZE );
//  
//  // ����������
//  char* trackbarLabel = 
//	  "Method: \n \
//	  0: SQDIFF \n \
//	  1: SQDIFF NORMED \n \
//	  2: TM CCORR \n \
//	  3: TM CCORR NORMED \n \
//	  4: TM COEFF \n \
//	  5: TM COEFF NORMED";
//  //���������������� ��ʾ�������� ƥ�䷽��index ��������Χ �ص�����
//  createTrackbar( trackbarLabel, imageWindow, &matchMethod, maxTrackbar, MatchingMethod );
//
//  double t = cvGetTickCount();
//  
//  MatchingMethod( 0, 0 );
//  t = (double)(cvGetTickCount() - t)/(cvGetTickFrequency()*1000.*1000.);
//  cout<<"ģ��ƥ��ʱ��"<<t<<endl;
//
//  waitKey(0);
//  return 0;
//}
//
///// �������� ///
//void MatchingMethod( int, void* )		//ƥ�亯��
//{
//  // ���������ʾ
//  Mat displayImg;
//  srcImg.copyTo( displayImg );
// 
//  // ����ƥ����ͼ��Ϊÿ��ģ��λ�ô洢ƥ����
//  // ƥ����ͼ���СΪ��(W-w+1)*(H-h+1)
//  int result_cols =  srcImg.cols - templImg.cols + 1;
//  int result_rows = srcImg.rows - templImg.rows + 1;
//  resultImg.create( result_cols, result_rows, CV_32FC1 );
//
//  // ����ƥ�䲢��һ��
// // matchTemplate( srcImg, templImg, resultImg, matchMethod );
//   matchTemplate( img1_down, img2_down, resultImg, matchMethod );
//  normalize( resultImg, resultImg, 0, 1, NORM_MINMAX, -1, Mat() );
//
//  // ʹ��minMaxLoc�ҳ����ƥ��
//  double minVal, maxVal;
//  Point minLoc, maxLoc, matchLoc;
//  minMaxLoc( resultImg, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
//
//  // ����CV_TM_SQDIFF�� CV_TM_SQDIFF_NORMED�����ַ�������СֵΪ���ƥ�䣻���ڱ�ķ������ֵΪ���ƥ��
//  if( matchMethod  == CV_TM_SQDIFF || matchMethod == CV_TM_SQDIFF_NORMED )
//    { matchLoc = minLoc; }
//  else  
//    { matchLoc = maxLoc; }
//
//  // ��ԭʼͼ���ƥ����ͼ���������ƥ���Ϊ���ϽǱ�����ƥ���
// /* rectangle( displayImg, matchLoc, Point( matchLoc.x + templImg.cols , matchLoc.y + templImg.rows ), Scalar(0,0,255), 2, 8, 0 ); 
//  rectangle( resultImg, matchLoc, Point( matchLoc.x + templImg.cols , matchLoc.y + templImg.rows ), Scalar::all(0), 2, 8, 0 ); 
//*/
//  matchLoc *= scale;	//��Ϊ��1/4������
//  cout<<"���Ͻ����꣺"<<matchLoc.x<<","<<matchLoc.y<<endl;
//  rectangle( displayImg, matchLoc, Point( matchLoc.x + templImg.cols , matchLoc.y + templImg.rows ),Scalar(0,0,255)  , 2, 8, 0 ); 
// 
//  imwrite(dir_result,displayImg);
//  rectangle( resultImg, matchLoc, Point( matchLoc.x + templImg.cols , matchLoc.y + templImg.rows ), Scalar::all(0), 2, 8, 0 ); 
// //  imwrite("e:\\��Ӣ���Ա���\\0312SURF_BRISK\\1\\result_huidu.jpg",resultImg);
//
//  imshow( imageWindow, displayImg );
//  imshow( resultWindow, resultImg );
//
//  return;
//}

