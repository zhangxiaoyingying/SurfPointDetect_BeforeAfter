
#include "stdafx.h"

//////据说是具有尺度缩放性的模板匹配改进法。。。实验结果不佳
//int app(vector<double> minV);  
//
//int main( )
//{
//	dir2 = "e:\\张英测试报告\\0312SURF_BRISK\\1\\2_4.jpg";
//	dir1 = "e:\\张英测试报告\\0312SURF_BRISK\\1\\1_4.jpg";
//	dir3 = "e:\\张英测试报告\\0312SURF_BRISK\\1\\";
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
////		cout<<down_temp[i].cols<<endl;
////		cout<<down_temp[i].rows<<endl;
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
//	//	cout<<minVal<<endl;
//		minV.push_back(minVal);
//		minL.push_back(minLoc);
//		resize( down_temp[i], temp1, Size( down_temp[i].cols/1.3, down_temp[i].rows/1.3) );
//		down_temp.push_back(temp1);
//	}
//
//
//	int location;
//	location = app(minV); 
////	cout<<location<<endl;
//
//	rectangle( img, minL[location], Point( minL[location].x + down_temp[location].cols , minL[location].y + down_temp[location].rows ), Scalar::all(0), 2, 8, 0 );
//
//t = (double)(cvGetTickCount() -t )/(cvGetTickFrequency()*1000.*1000.);
//cout<<"尺度缩放模板匹配时间："<<t<<endl;
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
//
//
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