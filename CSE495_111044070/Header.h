#ifndef _HEADER_
#define _HEADER_
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <list>

#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace xfeatures2d;



Scalar colorTab[] =
{
	Scalar(0, 0, 255),
	Scalar(0,255,0),
	Scalar(255,100,100),
	Scalar(255,0,255),
	Scalar(0,255,255),
	Scalar(0,0,0)
};

vector<Point> mCorners;
vector<Point2i> mEdges;


int row, col;
RNG rng(12345);

void checkPoint(int a, int b);
void select4Corner(const Mat& img);
void otoCornerDetect(const Mat& src);
/**************/
static void onMouse(int event, int x, int y, int, void*);
Point2i roi4point[4];
int roiIndex = 0;
bool oksign = false;

//Point2f MinDistFind(float x, float y, Point2f* inPoints);
void drawPoly(Mat img, vector<Point> vp);
void drawPoly(Mat img, Point lt, Point rt, Point rb, Point lb);
vector<Point> fillPoints(vector<Point> src);
void otoCornerDetect(const Mat& src);
void checkPoint(int a, int b);
void select4Corner(const Mat& img);
void PointOrderbyCorner(Point2i* inPoints, int w, int h);
static void onMouse(int event, int x, int y, int, void*);
Mat divideAndProject(const Mat img, vector<Point> contour, vector<Point> vertices, const int r, const int c);


vector<Point> DouglasPeucker(vector<Point> &points, double epsilon);
double distance_to_Line(cv::Point line_start, cv::Point line_end, cv::Point point);
list<Point> DouglasPeuckerWrapper(vector<Point> &points, vector<Point>::iterator it1, vector<Point>::iterator it2, double epsilon);


Mat p2d_interpolateM(Mat Image, Mat P, int Method);
Mat compute_homographyM(Mat m, Mat M, Mat& Hnorm, Mat& inv_Hnorm);
Mat p2d_rectangularM(Mat InputImage, Mat Vertices, int Interpolation);

#endif // !_HEADER_



