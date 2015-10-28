#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

#define AUTO_CORNER_DETECTION 1
#define CORNER_AREA  10

Scalar colorTab[] =
{
	Scalar(0, 0, 255),
	Scalar(0,255,0),
	Scalar(255,100,100),
	Scalar(255,0,255),
	Scalar(0,255,255)
};


vector<Point2i> mCorners;
vector<Point2i> mEdges;

int min_thresh = 75;
int max_thresh = 140;
int row, col;
RNG rng(12345);

void checkPoint(int a, int b);
void select4Corner(const Mat& img);
void otoCornerDetect(const Mat& src);
/**************/
static void onMouse(int event, int x, int y, int, void*);
Point2f roi4point[4];
int roiIndex = 0;
bool oksign = false;

Point2f MinDistFind(float x, float y, Point2f* inPoints);
void PointOrderbyConner(Point2f* inPoints, int w, int h);
/****************/


int main(int argc, char** argv){

	Mat src = imread("media/scan14.jpg");
	if (!src.data) {
		cout << "no input image\n";
		return 0;
	}
	cout << "cols:" << src.cols << " rows:" << src.rows << endl;
	if (src.rows > 900)
	{
		row = 900;
		col = 900 * src.cols / src.rows;
		Size dsize = Size(round(col), round(row));
		resize(src, src, dsize);
	}
	else {
		row = src.rows;
		col = src.cols;
	}
	
	if (AUTO_CORNER_DETECTION) {
		otoCornerDetect(src);
	}
	else {
		select4Corner(src);	
	}

	return(0);
}
void otoCornerDetect(const Mat& src) {
	Mat biggestContour = Mat::zeros(src.size(), CV_8UC3);

	cout << "cols:" << src.cols << " rows:" << src.rows << endl;
	Mat src_gray, src_blur, canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Convert image to gray and blur it
	cvtColor(src, src_gray, CV_RGB2GRAY);
	blur(src_gray, src_blur, Size(3, 3));

	/// Detect edges using canny	
	Canny(src_blur, canny_output, min_thresh, min_thresh*2.0, 3);

	/// Find contours
	findContours(canny_output, contours, noArray(), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point());

	/// Find surrounding contour
	int max = 0;
	int index = -1;
	for (int i = 0; i < contours.size(); i++) {
		int temp = contourArea(contours[i], false);
		if (temp > max) {
			max = temp;
			index = i;
		}
	}


	Mat raw_dist(src.size(), CV_32FC1);
	//conturun icimi diye bakar
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			raw_dist.at<float>(i, j) = pointPolygonTest(contours[index], Point2f(j, i), true);
		}
	}

	Mat nears = Mat::zeros(src.size(), CV_8UC3);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			//if (raw_dist.at<float>(i, j) < 0)//outside
			//{
			//	raw_dist.at<Vec3b>(i, j)[0] = 255 - (int)abs(raw_dist.at<float>(i, j)) * 255 / minVal;
			//}
			//else 
			if (raw_dist.at<float>(i, j) > 0)//inside
			{
				biggestContour.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				biggestContour.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				biggestContour.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
			}
			else //near
			{
				biggestContour.at<Vec3b>(i, j)[0] = 255;
				biggestContour.at<Vec3b>(i, j)[1] = 255;
				biggestContour.at<Vec3b>(i, j)[2] = 255;

				nears.at<Vec3b>(i, j)[0] = 255;
				nears.at<Vec3b>(i, j)[1] = 255;
				nears.at<Vec3b>(i, j)[2] = 255;

				mEdges.push_back(Point(i, j));
			}
		}
	}

	cout << "max area is " << max << endl;

	//draw largest contour
	drawContours(biggestContour, contours, index, Scalar(255), 2, 8, noArray(), INT_MAX, Point());
	//Rect boundRect = boundingRect(contours[index]);

	Mat dst, dst_norm, dst_norm_scaled;
	cvtColor(nears, src_gray, CV_BGR2GRAY);

	/// Detector parameters
	int blockSize = 7;
	int apertureSize = 5;
	double k = 0.05;

	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	for (int i = 0; i < dst_norm.rows; i++) {
		for (int j = 0; j < dst_norm.cols; j++) {
			if ((int)dst_norm.at<float>(i, j) > min_thresh*2.0) {
				circle(biggestContour, Point(j, i), 10, colorTab[i % 5], 2, 8, 0);
				//cout << "unchecked point of corner " << i << "_" << j << endl;
				checkPoint(j, i);
			}
		}
	}
	//select4Corner(biggestContour);
	for (int i = 0; i < mCorners.size(); i++) {
		circle(biggestContour, mCorners[i], 20, colorTab[0], 2, 8, 0);
	}
	cout << mCorners.size() << " corners found\n";
	//vector<vector<Point>> hull;
	//convexHull(mCorners, hull, false, true);
	//drawContours(biggestContour, hull, 0, Scalar(55, 55, 55), 1, 8, noArray(), INT_MAX, Point());
	//rectangle(biggestContour, boundRect, colorTab[3], 3, 8, 0);

	/// Showing the result
	namedWindow("corners", CV_WINDOW_AUTOSIZE);
	imshow("corners", biggestContour);
	waitKey(0);
	/**************************************/
	vector<Point2i> dstCorners;
	dstCorners.push_back(Point(0, 0));
	dstCorners.push_back(Point(800, 0));
	dstCorners.push_back(Point(800, 800));
	dstCorners.push_back(Point(0, 800));


	Mat H = findHomography(mCorners, dstCorners, noArray(), 0, 3.0);
	//imshow("homografy", dstCorners);

	Mat outputImage;
	warpPerspective(src, outputImage, H, Size(biggestContour.cols, biggestContour.rows));
	imshow("output", outputImage);
	waitKey();
}
void checkPoint(int a, int b) {
	bool isNear = false;
	int indexNear = 0;

	if (col - a < CORNER_AREA || row - b < CORNER_AREA ||
		a < CORNER_AREA || b < CORNER_AREA) {
		//cout << "out of paper " << i << "(col:"<<col<<")_(row:"<<row<<")" << j << endl;
		return;
		
	}
	for (int k = 0; k < mCorners.size(); k++) {
		if(sqrt(pow(mCorners[k].x - a,2)+pow(mCorners[k].y - b,2)) < CORNER_AREA){//already corner
			//cout << "corner exist " << i << "_" << j << endl;
			isNear = true;
			break;
		}

	}
	if (!isNear) {//for new corner
		cout << "----->checked point of corner " << a << "_" << b << endl;
		mCorners.push_back(Point(a,b));
	}
}

void select4Corner(const Mat& img){
	Mat RoiImg;
	//window  
	namedWindow("set roi by 4 points", 0);

	//mouse callback  
	setMouseCallback("set roi by 4 points", onMouse, 0);


	while (1)
	{
		if (oksign == true) //right button click  
			break;

		//draw point  
		RoiImg = img.clone();
		for (int i = 0; i< roiIndex; ++i)
			circle(RoiImg, roi4point[i], 5, CV_RGB(255, 0, 255), 5);
		imshow("set roi by 4 points", RoiImg);

		waitKey(10);
	}
	printf("points ordered by LT, RT, RB, LB \n");
	PointOrderbyConner(roi4point, img.size().width, img.size().height);
	for (int i = 0; i< 4; ++i)
	{
		printf("[%d] (%.2lf, %.2lf) \n", i, roi4point[i].x, roi4point[i].y);
	}
	int xMin = INT_MAX, xMax = INT_MIN, 
		yMin = INT_MAX, yMax = INT_MIN;
	for (int i = 0; i < 4; i++)
	{
		if (roi4point[i].x > xMax)
			xMax = roi4point[i].x;
		if (roi4point[i].x < xMin)
			xMin = roi4point[i].x;
		if (roi4point[i].y > yMax)
			yMax = roi4point[i].y;
		if (roi4point[i].y < yMin)
			yMin = roi4point[i].y;
	}
	int Weight = xMax - xMin;
	int Height = yMax - yMin;
	cout << "min_x:" << xMin << " max_x:" << xMax << " min_y:" << yMin << " max_y:" << yMax << " x:" << Weight << " y:" << Height << endl;


	//drwaring  
	RoiImg = img.clone();
	string TestStr[4] = { "LT","RT","RB","LB" };
	putText(RoiImg, TestStr[0].c_str(), roi4point[0], CV_FONT_NORMAL, 1, Scalar(0, 0, 255), 3);
	circle(RoiImg, roi4point[0], 3, CV_RGB(0, 0, 255));

	for (int i = 1; i< roiIndex; ++i)
	{
		line(RoiImg, roi4point[i - 1], roi4point[i], CV_RGB(255, 0, 0), 1);
		circle(RoiImg, roi4point[i], 1, CV_RGB(0, 0, 255), 3);
		putText(RoiImg, TestStr[i].c_str(), roi4point[i], CV_FONT_NORMAL, 1, Scalar(0, 0, 255), 3);
	}

	line(RoiImg, roi4point[0], roi4point[roiIndex - 1], CV_RGB(255, 0, 0), 1);
	imshow("set roi by 4 points2", RoiImg);

	//prepare to get homography matrix  
	vector< Point2f> P1; //clicked positions  
	vector< Point2f> P2(4); //user setting positions  
	for (int i = 0; i< 4; ++i)
		P1.push_back(roi4point[i]);

	//user setting position  
	P2[0].x = 0; P2[0].y = 0;
	P2[1].x = Weight; P2[1].y = 0;
	P2[2].x = Weight; P2[2].y = Height;
	P2[3].x = 0; P2[3].y = Height;

	//get homography  
	Mat H = findHomography(P1, P2);

	//warping  
	Mat warped_image;
	warpPerspective(img, warped_image, H, Size(img.cols, img.rows));
	rectangle(warped_image, Point(0, 0), Point(Weight, Height), CV_RGB(255, 0, 0));
	imshow("warped_image", warped_image);


	///////////////////////////  
	//calculation confirm  
	cout << "h:" << endl << H << endl;
	cout << "size rows and cols " << H.rows << " " << H.cols << endl;

	Mat A(3, 4, CV_64F); //3xN, P1  
	Mat B(3, 4, CV_64F); //3xN, P2  
						 //B = H*A  (P2 = h(P1))  


	for (int i = 0; i< 4; ++i)
	{
		A.at< double>(0, i) = P1[i].x;
		A.at< double>(1, i) = P1[i].y;
		A.at< double>(2, i) = 1;

		B.at< double>(0, i) = P2[i].x;
		B.at< double>(1, i) = P2[i].y;
		B.at< double>(2, i) = 1;
	}

	cout << "a" << endl << A << endl;
	cout << "b" << endl << B << endl;
	Mat HA = H*A;

	for (int i = 0; i< 4; ++i)
	{
		HA.at< double>(0, i) /= HA.at< double>(2, i);
		HA.at< double>(1, i) /= HA.at< double>(2, i);
		HA.at< double>(2, i) /= HA.at< double>(2, i);
	}

	cout << "HA" << endl << HA << endl;
	
	waitKey(0);

}
void PointOrderbyConner(Point2f* inPoints, int w, int h)
{

	vector< pair< float, float> > s_point;
	for (int i = 0; i< 4; ++i)
		s_point.push_back(make_pair(inPoints[i].x, inPoints[i].y));

	//sort  
	sort(s_point.begin(), s_point.end(), [](const pair< float, float>& A, const pair< float, float>& B) { return A.second < B.second; });

	if (s_point[0].first < s_point[1].first)
	{
		inPoints[0].x = s_point[0].first;
		inPoints[0].y = s_point[0].second;

		inPoints[1].x = s_point[1].first;
		inPoints[1].y = s_point[1].second;

	}
	else {
		inPoints[0].x = s_point[1].first;
		inPoints[0].y = s_point[1].second;

		inPoints[1].x = s_point[0].first;
		inPoints[1].y = s_point[0].second;
	}

	if (s_point[2].first > s_point[3].first)
	{
		inPoints[2].x = s_point[2].first;
		inPoints[2].y = s_point[2].second;

		inPoints[3].x = s_point[3].first;
		inPoints[3].y = s_point[3].second;

	}
	else {
		inPoints[2].x = s_point[3].first;
		inPoints[2].y = s_point[3].second;

		inPoints[3].x = s_point[2].first;
		inPoints[3].y = s_point[2].second;
	}
}
static void onMouse(int event, int x, int y, int, void*)
{
	if (event == CV_EVENT_LBUTTONDOWN && oksign == false)
	{
		//4 point select  
		if (roiIndex >= 4)
		{
			roiIndex = 0;
			for (int i = 0; i< 4; ++i)
				roi4point[i].x = roi4point[i].y = 0;
		}

		roi4point[roiIndex].x = x;
		roi4point[roiIndex].y = y;

		//point coordinate print  
		printf("1:(%.2lf,%.2lf), 2:(%.2lf,%.2lf), 3:(%.2lf,%.2lf), 4:(%.2lf,%.2lf)\n",
			roi4point[0].x, roi4point[0].y, roi4point[1].x, roi4point[1].y, roi4point[2].x, roi4point[2].y, roi4point[3].x, roi4point[3].y);

		roiIndex++;
	}

	if (event == CV_EVENT_RBUTTONDOWN)
	{
		//set point.  
		if (roiIndex == 4)
		{
			oksign = true;
			printf("Warping Start!!!\n");
		}
	}
}