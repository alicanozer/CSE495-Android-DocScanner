#ifndef _HEADER_
#define _HEADER_
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

#define AUTO_CORNER_DETECTION 0
#define CORNER_AREA  10

Scalar colorTab[] =
{
	Scalar(0, 0, 255),
	Scalar(0,255,0),
	Scalar(255,100,100),
	Scalar(255,0,255),
	Scalar(0,255,255)
};

vector<Point2f> mCorners;
vector<Point2f> mEdges;

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

//Point2f MinDistFind(float x, float y, Point2f* inPoints);

void otoCornerDetect(const Mat& src);
void checkPoint(int a, int b);
void select4Corner(const Mat& img);
void PointOrderbyConner(Point2f* inPoints, int w, int h);
static void onMouse(int event, int x, int y, int, void*);



Mat p2d_interpolateM(Mat Image, Mat P, int Method);
Mat compute_homographyM(Mat m, Mat M, Mat& Hnorm, Mat& inv_Hnorm);
Mat p2d_rectangularM(Mat InputImage, Mat Vertices, int Interpolation);

#endif // !_HEADER_



