#ifndef _HEADER_
#include "Header.h"
#include <fstream>
#include <time.h> // to calculate time needed
#define MAX_HEIGHT 1080 //max height of input image
#define AUTO_CORNER_DETECTION 1
#define DIV_IMG 36

void drawPoly(Mat img, vector<Point> vp) {
	if (vp.size() != 4) {
		cout << "drawPoly error\n";
		return;
	}
	line(img, vp[0], vp[1], Scalar(0, 0, 0));
	line(img, vp[1], vp[2], Scalar(0, 0, 0));
	line(img, vp[2], vp[3], Scalar(0, 0, 0));
	line(img, vp[3], vp[0], Scalar(0, 0, 0));
}
void drawPoly(Mat img, Point lt, Point rt, Point rb, Point lb) {

	line(img, lt, rt, Scalar(0, 0, 0));
	line(img, rt, rb, Scalar(0, 0, 0));
	line(img, rb, lb, Scalar(0, 0, 0));
	line(img, lb, lt, Scalar(0, 0, 0));
}
vector<Point> fillPoints(vector<Point> src);
int main(int argc, char** argv){

	Mat src = imread("C:/Users/Can/Desktop/Data/img_0184.jpg");//19
	if (!src.data) {
		cout << "no input image\n";
		return 0;
	}
	row = src.rows;
	col = src.cols;
	cout << "cols:" << src.cols << " rows:" << src.rows << endl;
	if (src.rows > MAX_HEIGHT)
	{
		row = MAX_HEIGHT;
		col = MAX_HEIGHT * src.cols / src.rows;
		Size dsize = Size(round(col), round(row));
		resize(src, src, dsize);
	}
	else {
		row = src.rows;
		col = src.cols;
	}
	int sayi = min(row, col);
	int ebob=1;

	for (int i = sayi; i > 0; i--){
		if ((row % i) == 0 && (col % i) == 0){
			ebob = i;
			break;
		}
	}
	cout << "ebob:" << ebob << endl;

	/*
	float dummy_query_data[8] = { 172, 287, 1031, 256, 1249, 773, 40, 873 };
	Mat Vertices = cv::Mat(4, 2, CV_32FC1, dummy_query_data);
	Mat output = p2d_rectangularM(src, Vertices, 1);
	cout << "returned:"<<output.dims << endl;
	cout << output.rows << endl;
	cvtColor(output, output, CV_RGB2GRAY);
	imshow("result1", output);
	waitKey();
	*/
	

	if (AUTO_CORNER_DETECTION) {
		otoCornerDetect(src);
		//Mat result = divideAndProject(src, 1, 2);
		//imshow("dividexx", result);
		//waitKey();
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
	int biggest_index = -1;
	for (int i = 0; i < contours.size(); i++) {
		int temp = contourArea(contours[i], false);
		if (temp > max) {
			max = temp;
			biggest_index = i;
		}
	}


	Mat raw_dist(src.size(), CV_32FC1);
	//conturun icimi diye bakar
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			raw_dist.at<float>(i, j) = pointPolygonTest(contours[biggest_index], Point2f(j, i), true);
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
	drawContours(biggestContour, contours, biggest_index, colorTab[5], 2, 8, noArray(), INT_MAX, Point());
	//Rect boundRect = boundingRect(contours[biggest_index]);
	
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
				//circle(biggestContour, Point(j, i), 10, colorTab[i % 5], 2, 8, 0);
				//cout << "unchecked point of corner " << i << "_" << j << endl;
				checkPoint(j, i);
			}
		}
	}
	//select4Corner(biggestContour);

	for (int i = 0; i < mCorners.size(); i++) {
		circle(biggestContour, mCorners[i], 20, colorTab[5], 2, 8, 0);
		char text[20];
		sprintf(text, "%d,%d", mCorners[i].x, mCorners[i].y);
		putText(biggestContour, text, mCorners[i], CV_FONT_NORMAL, 1, colorTab[5]);
	}
	cout << mCorners.size() << " corners found\n";
	/// Showing the result
	namedWindow("corners", CV_WINDOW_AUTOSIZE);
	imshow("corners", biggestContour);

	/**************************************/
	vector<Point> dstCorners;
	dstCorners.push_back(Point(0, 0));
	dstCorners.push_back(Point(900, 0));
	dstCorners.push_back(Point(900, 900));
	dstCorners.push_back(Point(0, 900));
	Point2i newCorners[4];
	for (int i = 0; i < 4; i++){
		newCorners[i] = mCorners[i];
	}
	PointOrderbyConner(newCorners, src.size().width, src.size().height);
	for (int i = 0; i < 4; i++) {
		mCorners[i] = newCorners[i];
	}
	/***************/
	Mat result = divideAndProject(src, contours[biggest_index], mCorners, DIV_IMG, DIV_IMG);
	imshow("Result", result);
	waitKey();

	//Mat H = findHomography(mCorners, dstCorners, noArray(), 0, 3.0);
	//Mat outputImage;
	//warpPerspective(src, outputImage, H, Size(biggestContour.cols, biggestContour.rows));
	//imshow("output", outputImage);
	//waitKey();
}
double distancePoints(Point a, Point b) {
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}
Mat divideAndProject(const Mat img, vector<Point> contour, vector<Point> vertices, const int r, const int c) {
	int Height = img.size().height;
	int Width = img.size().width;	
	Mat segments = img.clone();// = Mat::zeros(img.size(), CV_8UC3);
	Mat OutputImage, warped;
	int rt_i=-1, lt_i=-1, rb_i=-1, lb_i=-1;	

	contour = fillPoints(contour);

	//finding indexes of corners with given contour and points
	vector<Point> leftEdge, rightEdge, topEdge, buttomEdge;
	for (int i = 0; i < contour.size(); i++)
	{
		if (distancePoints(contour[i], vertices[0])<5) lt_i = i;
		if (distancePoints(contour[i], vertices[1])<5) rt_i = i;
		if (distancePoints(contour[i], vertices[2])<5) rb_i = i;
		if (distancePoints(contour[i], vertices[3])<5) lb_i = i;
	}
	//draw contour start-end points
	cout << "st"<<contour[0] << endl;
	cout << "fn"<<contour[contour.size() - 1] << endl;
	if (lt_i == -1 || rt_i == -1 || rb_i == -1 || lb_i == -1) {
		cout << "finding corner error\n";
		exit(-1);
	}

	//cout << "lt_i:"<< lt_i << ">" << contour[lt_i] << endl;
	//cout << "rt_i:"<< rt_i << ">" << contour[rt_i] << endl;
	//cout << "rb_i:"<< rb_i << ">" << contour[rb_i] << endl;
	//cout << "lb_i:"<< lb_i << ">" << contour[lb_i] << endl;
	//cout << "contours.size():" << contour.size() << endl;

	/////KONTORUN HANGI EDGE DEN BASLADIGINI BULUP EDGELERI AYIRIR
	enum WHICH{LEFT=0,BUTTOM,RIGHT,TOP};
	WHICH EDGE;
	int startingContour= -1;
	if (lt_i<rt_i && lt_i<rb_i && lt_i<lb_i) EDGE = TOP;
	if (rt_i<lt_i && rt_i<rb_i && rt_i<lb_i) EDGE = RIGHT;
	if (lb_i<rt_i && lb_i<rb_i && lb_i<lt_i) EDGE = LEFT;
	if (rb_i < rt_i && rb_i < lt_i && rb_i < lb_i) EDGE = BUTTOM;
	
	
	if (EDGE == TOP) {
		cout << "contour starting TOP\n";
		for (int i = lt_i; i <= lb_i; i++)
			leftEdge.push_back(contour[i]);
		for (int i = lb_i; i <= rb_i; i++) 
			buttomEdge.push_back(contour[i]);
		for (int i = rt_i; i >= rb_i; i--) 
			rightEdge.push_back(contour[i]);

		for (int i = lt_i; i >= 0; i--) 
			topEdge.push_back(contour[i]);
		for (int i = contour.size(); i >= rt_i; i--)
			topEdge.push_back(contour[i]);
	}
	else if (EDGE == LEFT) {
		cout << "contour starting LEFT\n";
		for (int i = lb_i; i <= rb_i; i++) 
			buttomEdge.push_back(contour[i]);
		for (int i = rt_i ; i >= rb_i; i--) 
			rightEdge.push_back(contour[i]);
		for (int i = lt_i ; i >= rt_i; i--) 
			topEdge.push_back(contour[i]);

		for (int i = lt_i; i < contour.size(); i++) 
			leftEdge.push_back(contour[i]);
		for (int i = 0; i <= lb_i; i++) 
			leftEdge.push_back(contour[i]);
	}
	else if (EDGE == BUTTOM) {
		cout << "contour starting BUTTOM\n";
		for (int i = lt_i; i <= lb_i; i++) 
			leftEdge.push_back(contour[i]);
		for (int i = rt_i; i >= rb_i; i--)
			rightEdge.push_back(contour[i]);
		for (int i = lt_i ; i >= rt_i; i--)
			topEdge.push_back(contour[i]);

		for (int i = lb_i; i < contour.size(); i++)
			buttomEdge.push_back(contour[i]);
		for (int i = 0; i < rb_i; i++) 
			buttomEdge.push_back(contour[i]);		
	}
	else {//RIGHT
		cout << "contour starting RIGHT\n";
		for (int i = lt_i; i <= lb_i; i++)
			leftEdge.push_back(contour[i]);
		for (int i = lt_i; i >= rt_i; i--) 
			topEdge.push_back(contour[i]);
		for (int i = lb_i; i <= rb_i; i++) 
			buttomEdge.push_back(contour[i]);

		for (int i = rt_i; i >= 0; i--) 
			rightEdge.push_back(contour[i]);
		for (int i = contour.size()-1; i >= rb_i; i--)
			rightEdge.push_back(contour[i]);
	}

	
	vector<vector<Point>> coordinates;
	coordinates.resize(r + 1);
	for (int i = 0; i <= r; i++){
		coordinates[i].resize(c + 1);
	}
	cout << "leftEdge  :" << leftEdge.size()<<" "<< leftEdge[0] << " " << leftEdge[leftEdge.size() - 1] << endl;
	cout << "buttomEdge:" << buttomEdge.size() << " "<<buttomEdge[0] << " " << buttomEdge[buttomEdge.size() - 1] << endl;
	cout << "rightEdge :" << rightEdge.size() << " " <<rightEdge[0] << " " << rightEdge[rightEdge.size() - 1] << endl;
	cout << "topEdge   :" << topEdge.size() << " "<<topEdge[0] << " " << topEdge[topEdge.size() - 1] << endl;
	//draw edge pixels
	//for (int i = 0; i<leftEdge.size(); i++)
	//	circle(segments, cvPoint(leftEdge[i].x, leftEdge[i].y), 2, colorTab[0], -1, 8, 0);
	//for (int i = 0; i<buttomEdge.size(); i++)
	//	circle(segments, cvPoint(buttomEdge[i].x, buttomEdge[i].y), 2, colorTab[1], -1, 8, 0);
	//for (int i = 0; i<rightEdge.size(); i++)
	//	circle(segments, cvPoint(rightEdge[i].x, rightEdge[i].y), 2, colorTab[2], -1, 8, 0);
	//for (int i = 0; i<topEdge.size(); i++)
	//	circle(segments, cvPoint(topEdge[i].x, topEdge[i].y), 2, colorTab[3], -1, 8, 0);

	//edge noktalarý bul
	vector<int> l_index, r_index, t_index, b_index;
	double l_to = (double)(leftEdge.size() / r);
	double r_to = (double)(rightEdge.size() / r);
	double t_to = (double)(topEdge.size() / c);
	double b_to = (double)(buttomEdge.size() / c); 
	for (int i = 0; i <= r; i++)
	{
		l_index.push_back(round(l_to*i));
		r_index.push_back(round(r_to*i));
		circle(segments, leftEdge[l_index[i]], 5, colorTab[0]);
		circle(segments, rightEdge[r_index[i]], 10, colorTab[1]);
	}
	for (int i = 0; i <= c; i++)
	{
		t_index.push_back(round(t_to*i));
		b_index.push_back(round(b_to*i));
		circle(segments, topEdge[t_index[i]], 15, colorTab[2]);
		circle(segments, buttomEdge[b_index[i]], 20, colorTab[3]);
	}
	
	for (int i = 0; i <= r; i++){
		for (int j = 0; j <= c; j++){
			coordinates[i][j].x = leftEdge[l_index[i]].x + round((((double)rightEdge[r_index[i]].x - leftEdge[l_index[i]].x) / c) * j);
			coordinates[i][j].y = topEdge[t_index[j]].y  + round((((double)buttomEdge[b_index[j]].y - topEdge[t_index[j]].y) / r) * i);
	//		cout << "coordinates[" << i << "][" << j << "]:" << coordinates[i][j] << endl;
			circle(segments, coordinates[i][j], 2, colorTab[5],2);
		}
	}	
	
	//cout << "-------------------------------" << endl;
	//cout << coordinates[0][0] << "--" << coordinates[0][c] << endl;
	//cout << coordinates[r][0] << "--" << coordinates[r][c] << endl;
	//cout << "-------------------------------" << endl;
	//calculate target coordinate acc. to selected points
	int xMin = INT_MAX, xMax = INT_MIN,
		yMin = INT_MAX, yMax = INT_MIN;
	for (int i = 0; i < 4; i++) {
		if (vertices[i].x > xMax) xMax = vertices[i].x;
		if (vertices[i].x < xMin) xMin = vertices[i].x;
		if (vertices[i].y > yMax) yMax = vertices[i].y;
		if (vertices[i].y < yMin) yMin = vertices[i].y;
	}
	for (int i = 0; i < 4; i++)
	{
		cout << "Vertices[" << i << "]:" << vertices[i] << endl;
	}
	int tWidth = (xMax - xMin)*1.2;
	int tHeight = (yMax - yMin)*1.2;
	cout << "tWidth:" << tWidth << endl;
	cout << "tHeight:" << tHeight << endl;
	

	//parcalari bilestirmek icin gereken koordinatlar
	vector<vector<Point>> Targets;
	Targets.resize(r+1);
	for (int i = 0; i < (r+1); i++) {
		Targets[i].resize(c+1);
	}
	for (int i = 0; i <= r; i++) {
		for (int j = 0; j <= c; j++) {
			Targets[i][j].x = round((double)(Width / c) * j);
			Targets[i][j].y = round((double)(Height / r) * i);
			//cout << "Targets" << i << "," << j << ":" << Targets[i][j] << endl;
		}
	}
	//cout << "-------------------------------" << endl;
	//cout << Targets[0][0] << "--" << Targets[0][c] << endl;
	//cout << Targets[r][0] << "--" << Targets[r][c] << endl;
	//cout << "-------------------------------" << endl;

	//warping  
	Mat output_image = Mat::zeros(Size(Width/c, Height/r), CV_8UC3);
	warped = Mat::zeros(img.size(), CV_8UC3);
	Mat imgRect = img.clone();

	int scale_x = Width / c;
	int scale_y = Height / r;
	for (int i = 0; i < (r); i++) {
		for (int j = 0; j < (c); j++){
			vector<Point> P1(4);
			P1.clear();
			P1.push_back(coordinates[i][j]);
			P1.push_back(coordinates[i][j+1]);
			P1.push_back(coordinates[i+1][j+1]);
			P1.push_back(coordinates[i+1][j]);
			vector<Point> P2(4);
			P2.clear();
			P2.push_back(Point(0, 0));
			P2.push_back(Point(scale_x, 0));
			P2.push_back(Point(scale_x, scale_y));
			P2.push_back(Point(0, scale_y));

			Mat H = Mat::zeros(Size(3, 3), CV_8UC3); 
			drawPoly(segments, P1);
			//drawPoly(imgRect, Targets[i][j], Targets[i][j+1], Targets[ +1][j+1], Targets[i+1][j]);
			//rectangle(rects, P1[1],P1[3], Scalar(0, 0, 0), 1, 8, 0);
						
			H=findHomography(P1, P2, noArray(), 0, 3.0);

			warpPerspective(img, output_image, H, Size(output_image.cols, output_image.rows));
			drawPoly(output_image, P2);

			output_image.copyTo(warped.rowRange(Targets[i][j].y,Targets[i+1][j].y).colRange(Targets[i][j].x, Targets[i][j+1].x));
			//cout << Targets[i][j].y << "," << Targets[i + 1][j].y << "-" << Targets[i][j].x << "," << Targets[i][j + 1].x << endl;
		}
	}
	imshow("segments", segments);
	//imshow("imgRect", imgRect);


	return warped;
}
vector<Point> fillPoints(vector<Point> src) {
	assert(src.size() < 2);
	vector<Point> dst;
	int vSize = src.size();
	int i=0, j=0;
	cout << "size:" <<vSize << endl;
	for (int i = 0; i < (vSize -1); i++){
		int j = i + 1;

		int x1 = src[i].x;
		int x2 = src[j].x;
		int y1 = src[i].y;
		int y2 = src[j].y;

		dst.push_back(src[i]);
		double m = (double)(y2 - y1) / (x2 - x1);
		for (int j = 1; j <= abs(x2 - x1); j++)
		{
			if (x2 >= x1) {
				int x = x1 + j;
				dst.push_back(Point(x, m*x - m*x1 + y1));
			}
			else {
				int x = x1 - j;
				dst.push_back(Point(x, m*x - m*x1 + y1));
			}			
		}		
	}
	int x1 = src[vSize-1].x;
	int x2 = src[0].x;
	int y1 = src[vSize-1].y;
	int y2 = src[0].y;

	dst.push_back(src[vSize-1]);
	double m = (double)(y2 - y1) / (x2 - x1);
	for (int j = 1; j <= abs(x2 - x1); j++)
	{
		if (x2 >= x1) {
			int x = x1 + j;
			dst.push_back(Point(x, m*x - m*x1 + y1));
		}
		else {
			int x = x1 - j;
			dst.push_back(Point(x, m*x - m*x1 + y1));
		}
	}	

	return dst;
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
	namedWindow("set roi by 4 points", 1);

	//mouse callback  
	setMouseCallback("set roi by 4 points", onMouse, 0);

	while (1){
		if (oksign == true) //right button click  
			break;

		//draw point  
		RoiImg = img.clone();
		for (int i = 0; i< roiIndex; ++i)
			circle(RoiImg, roi4point[i], 5, CV_RGB(255, 0, 255), 5);
		imshow("set roi by 4 points", RoiImg);

		waitKey(10);
		
	}
	destroyWindow("set roi by 4 points");
	printf("points ordered by LT, RT, RB, LB \n");
	PointOrderbyConner(roi4point, img.size().width, img.size().height);
	for (int i = 0; i< 4; ++i){
		printf("[%d] (%.2lf, %.2lf) \n", i, roi4point[i].x, roi4point[i].y);
	}

	//drwaring  
	RoiImg = img.clone();
	string TestStr[4] = { "LT","RT","RB","LB" };
	putText(RoiImg, TestStr[0].c_str(), roi4point[0], CV_FONT_NORMAL, 1, Scalar(255, 0, 0), 3);
	circle(RoiImg, roi4point[0], 1, CV_RGB(255, 255, 255),3);

	for (int i = 1; i< roiIndex; ++i){
		line(RoiImg, roi4point[i - 1], roi4point[i], CV_RGB(0, 255, 0), 3);
		circle(RoiImg, roi4point[i], 1, CV_RGB(255, 255, 255), 3);
		putText(RoiImg, TestStr[i].c_str(), roi4point[i], CV_FONT_NORMAL, 1, Scalar(255, 0, 0), 3);
	}

	line(RoiImg, roi4point[0], roi4point[roiIndex - 1], CV_RGB(0, 255, 0), 3);
	imshow("set roi by 4 points2", RoiImg);

	//prepare to get homography matrix  
	vector< Point2f> P1; //clicked positions   
	for (int i = 0; i< 4; ++i)
		P1.push_back(roi4point[i]);

	//calculate target coordinate acc. to selected points
	int xMin = INT_MAX, xMax = INT_MIN,
		yMin = INT_MAX, yMax = INT_MIN;
	for (int i = 0; i < 4; i++){
		if (roi4point[i].x > xMax)
			xMax = roi4point[i].x;
		if (roi4point[i].x < xMin)
			xMin = roi4point[i].x;
		if (roi4point[i].y > yMax)
			yMax = roi4point[i].y;
		if (roi4point[i].y < yMin)
			yMin = roi4point[i].y;
	}
	int tWidth = xMax - xMin;
	int tHeight = yMax - yMin;
	cout << "min_x:" << xMin << " max_x:" << xMax <<
		"\nmin_y:" << yMin << " max_y:" << yMax <<
		"\ntargetWidth:" << tWidth << " targetHeight:" << tHeight << endl;

	//user setting position  
	vector<Point> P2(4);
	P2[0].x = 0; P2[0].y = 0;
	P2[1].x = tWidth; P2[1].y = 0;
	P2[2].x = tWidth; P2[2].y = tHeight;
	P2[3].x = 0; P2[3].y = tHeight;

	//get homography  
	Mat trP1, trP2;
	transpose(P1, trP1);
	transpose(P2, trP2);
	Mat H = findHomography(P1, P2);
	cout << "\nHOMOGRAPHY MATRIX2:" << H.rows << "x" << H.cols << endl << H << endl << endl;

	//warping  
	Mat output_image = Mat::zeros(Size(tWidth, tHeight), CV_8UC3);
	warpPerspective(img, output_image, H, Size(output_image.cols, output_image.rows));
	rectangle(output_image, Point(0, 0), Point(tWidth, tHeight), CV_RGB(255, 0, 0));
	imshow("output_image", output_image);

	
	waitKey(0);
	destroyAllWindows();
	exit(EXIT_SUCCESS);

}
void PointOrderbyConner(Point2i* inPoints, int w, int h){
	vector< pair< float, float> > s_point;
	for (int i = 0; i< 4; ++i)
		s_point.push_back(make_pair(inPoints[i].x, inPoints[i].y));

	//sort  
	sort(s_point.begin(), s_point.end(), [](const pair< float, float>& A, const pair< float, float>& B) { return A.second < B.second; });

	if (s_point[0].first < s_point[1].first){
		inPoints[0].x = s_point[0].first;
		inPoints[0].y = s_point[0].second;

		inPoints[1].x = s_point[1].first;
		inPoints[1].y = s_point[1].second;

	}else {
		inPoints[0].x = s_point[1].first;
		inPoints[0].y = s_point[1].second;

		inPoints[1].x = s_point[0].first;
		inPoints[1].y = s_point[0].second;

	}if (s_point[2].first > s_point[3].first){
		inPoints[2].x = s_point[2].first;
		inPoints[2].y = s_point[2].second;

		inPoints[3].x = s_point[3].first;
		inPoints[3].y = s_point[3].second;

	}else {
		inPoints[2].x = s_point[3].first;
		inPoints[2].y = s_point[3].second;

		inPoints[3].x = s_point[2].first;
		inPoints[3].y = s_point[2].second;
	}
}













static void onMouse(int event, int x, int y, int, void*) {
	if (event == CV_EVENT_LBUTTONDOWN && oksign == false) {
		//4 point select  
		if (roiIndex >= 4) {
			roiIndex = 0;
			for (int i = 0; i < 4; ++i)
				roi4point[i].x = roi4point[i].y = 0;
		}

		roi4point[roiIndex].x = x;
		roi4point[roiIndex].y = y;

		//point coordinate print  
		printf("1:(%.2lf,%.2lf), 2:(%.2lf,%.2lf), 3:(%.2lf,%.2lf), 4:(%.2lf,%.2lf)\n",
			roi4point[0].x, roi4point[0].y, roi4point[1].x, roi4point[1].y, roi4point[2].x, roi4point[2].y, roi4point[3].x, roi4point[3].y);

		roiIndex++;
	}

	if (event == CV_EVENT_RBUTTONDOWN) {
		//set point.  
		if (roiIndex == 4) {
			oksign = true;
			printf("Warping \n");
		}
	}
}
vector<Point> DouglasPeucker(vector<Point> &points, double epsilon) {
	if (points.size() <= 2) {
		return points;
	}

	list<Point> l = DouglasPeuckerWrapper(points, points.begin(), --(points.end()), epsilon);
	vector<Point> result{ begin(l), end(l) };
	return result;
}
double distance_to_Line(cv::Point line_start, cv::Point line_end, cv::Point point)
{
	double normalLength = _hypot(line_end.x - line_start.x, line_end.y - line_start.y);
	double distance = (double)((point.x - line_start.x) * (line_end.y - line_start.y) -
		(point.y - line_start.y) * (line_end.x - line_start.x)) / normalLength;
	return distance < 0 ? distance * -1 : distance;
}
list<Point> DouglasPeuckerWrapper(vector<Point> &points, vector<Point>::iterator it1, vector<Point>::iterator it2, double epsilon) {
	// Find the point with the maximum distance
	double dmax = 0;
	vector<Point>::iterator indexIt;

	for (vector<Point>::iterator tempIt = it1 + 1; tempIt != it2; ++tempIt) {
		double d = distance_to_Line(*it1, *it2, *tempIt);
		if (d > dmax) {
			dmax = d;
			indexIt = tempIt;
		}
	}

	// If max distance is greater than epsilon, recursively simplify
	if (dmax > epsilon) {
		// Recursive call
		list<Point> l1, l2;
		l1 = DouglasPeuckerWrapper(points, it1, indexIt, epsilon);
		l2 = DouglasPeuckerWrapper(points, indexIt, it2, epsilon);
		l1.pop_back();
		for (list<Point>::iterator itr = l2.begin(); itr != l2.end(); ++itr) {
			l1.push_back(*itr);
		}
		return l1;
	}
	else {
		// Erase all the points between it1 and it2
		list<Point> l;
		l.push_back(*it1);
		l.push_back(*it2);
		return l;
	}
}
Mat compute_homographyM(Mat m, Mat M, Mat& Hnorm, Mat& inv_Hnorm) {
	return findHomography(m, M);
}
	/*
//	compute_homography
//
//		[H, Hnorm, inv_Hnorm] = compute_homography(m, M)
//
//		Computes the planar homography between the point coordinates on the plane(M) and the image
//		point coordinates(m).
//
//	INPUT: m : homogeneous coordinates in the image plane(3xN matrix)
//	   M : homogeneous coordinates in the plane in 3D (3xN matrix)
//
//	   OUTPUT : H : Homography matrix(3x3 homogeneous matrix)
//			Hnorm : Normalization matrix used on the points before homography computation
//					(useful for numerical stability is points in pixel coordinates)
//			inv_Hnorm : The inverse of Hnorm
//
//		Definition : m ~H*M where "~" means equal up to a non zero scalar factor.
//
//		Method : First computes an initial guess for the homography through quasi - linear method.
//				Then, if the total number of points is larger than 4, optimize the solution by minimizing
//				the reprojection error(in the least squares sense).
//
//
//		Important functions called within that program :
//
//comp_distortion_oulu : Undistorts pixel coordinates.
//
//	compute_homography.m : Computes the planar homography between points on the grid in 3D, and the image plane.
//
//	project_points.m : Computes the 2D image projections of a set of 3D points, and also returns te Jacobian
//	matrix(derivative with respect to the intrinsic and extrinsic parameters).
//	This function is called within the minimization loop.
	
	int Np = m.cols;
	if (m.rows < 3){
		vector<Mat> arrays;
		arrays.push_back(m);
		arrays.push_back(Mat::ones(1, Np, CV_8U));
		vconcat(arrays, m);
	}

	if (M.rows < 3){
		vector<Mat> arrays;
		arrays.push_back(M);
		arrays.push_back(Mat::ones(1, Np, CV_8U));
		vconcat(arrays, M);
	}

	m = m / (Mat::ones(3, 1, CV_8U)*m.at<Mat>(2));
	M = M / (Mat::ones(3, 1, CV_8U)*M.at<Mat>(2));

	// Prenormalization of point coordinates(very important) :
	// (Affine normalization)

	Mat ax = m.at<Mat>(0);
	Mat ay = m.at<Mat>(1);

	Mat mxx = (Mat)mean(ax, noArray());
	Mat myy = (Mat)mean(ay, noArray());

	ax = ax - mxx;
	ay = ay - myy;

	Mat scxx = (Mat)mean(abs(ax));
	Mat scyy = (Mat)mean(abs(ay));

	Hnorm.at<Mat>(0, 0) = scxx.inv;
	Hnorm.at<Mat>(0, 1) = 0;
	Hnorm.at<Mat>(0, 2) = -1 * mxx*scxx.inv;
	Hnorm.at<Mat>(1, 0) = 0;
	Hnorm.at<Mat>(1, 1) = scyy.inv;
	Hnorm.at<Mat>(1, 2) = -1 * myy*scyy.inv;
	Hnorm.at<Mat>(2, 0) = 0;
	Hnorm.at<Mat>(2, 1) = 0;
	Hnorm.at<Mat>(2, 2) = 1;

	inv_Hnorm.at<Mat>(0, 0) = scxx;
	inv_Hnorm.at<Mat>(0, 1) = 0;
	inv_Hnorm.at<Mat>(0, 2) = mxx;
	inv_Hnorm.at<Mat>(1, 0) = 0;
	inv_Hnorm.at<Mat>(1, 1) = scyy;
	inv_Hnorm.at<Mat>(1, 2) = myy;
	inv_Hnorm.at<Mat>(2, 0) = 0;
	inv_Hnorm.at<Mat>(2, 1) = 0;
	inv_Hnorm.at<Mat>(2, 2) = 1;
	
	Mat mn = Hnorm*m;

	// Compute the homography between m and mn :
	// Build the matrix :

	Mat L = Mat::zeros(2*Np,9,CV_8U);
	Mat trM;
	transpose(M, trM);

	int ii = 0, jj = 0;
	for (int i = 0; i < 2*Np; i+=2,++ii)		
		for (int j = 0; j < 3; j++,++jj)
			L.at<float>(i, j) = trM.at<float>(ii, jj);

	ii = 0, jj = 0;
	for (int i = 1; i < 2*Np; i+=2,++ii)
		for (int j = 3; j < 6; j++,++jj)
			L.at<float>(i, j) = trM.at<float>(ii, jj);

	ii = 0, jj = 0;
	Mat trK1;
	transpose(((Mat::ones(3, 1, CV_8U)*mn.row(0).dot(M)), trK1);
	for (int i = 0; i < 2 * Np; i += 2, ++ii)
		for (int j = 6; j < 9; j++, ++jj){			
			L.at<float>(i, j) = trK1.at<float>(ii,jj);
		}

	ii = 0, jj = 0;
	Mat trK2;
	transpose(((Mat::ones(3, 1, CV_8U)*mn.at<Mat>(1)).dot(M)), trK2);
	for (int i = 1; i < 2 * Np; i += 2, ++ii)
		for (int j = 6; j < 8; j++, ++jj){			
			L.at<float>(i, j) = trK2.at<float>(ii, jj);
		}

	if (Np>4){
		Mat trL;
		transpose(L, trL);
		L = trL*L;
	}
	Mat U, S, V;
	//SVD::compute(L, S, U, V, 0);
	//todo SVD nin parametreerini kontrol et!!! 
	SVDecomp(L, S, U, V, 0);
	Mat hh = V.col(8);
	hh = hh / hh.col(8);

	//todo reshape paramtre kontrol et!!!
	Mat Hrem = hh.reshape(0, 9);

	//Final homography
	Mat H = inv_Hnorm*Hrem;
	//if 0,
	//	m2 = H*M;
	//m2 = [m2(1, :). / m2(3, :); m2(2, :). / m2(3, :)];
	//merr = m(1:2, : ) - m2;
	//end;

	//keyboard;
	//Homography refinement if there are more than 4 points:

	if (Np > 4){

		//final refinement
		Mat trH;
		transpose(H, trH);
		Mat hhv = trH.reshape(0, 9*1);
		hhv = hhv.at<Mat>(0,7);

		
		for (int i = 0; i < 10; i++)
		{
			Mat mrep = H*M;
			Mat J = Mat::zeros(2 * Np, 8, CV_8U);
			Mat MMM = M / Mat::ones(3, 1, CV_8U)*mrep.row(2);


			ii = 0, jj = 0;
			Mat invTrMMM;
			transpose(MMM,invTrMMM);
			invTrMMM = invTrMMM.inv;
			for (int i = 0; i < 2 * Np; i += 2, ++ii)
				for (int j = 0; j < 3; j++, ++jj) {
					L.at<float>(i, j) = invTrMMM.at<float>(ii, jj);
				}
			ii = 0, jj = 0;
			for (int i = 1; i < 2 * Np; i += 2, ++ii)
				for (int j = 3; j < 6; j++, ++jj) {
					L.at<float>(i, j) = invTrMMM.at<float>(ii, jj);
				}

			mrep = mrep / Mat::ones(3, 1, CV_8U) * mrep.row(2);
			Mat m_err;
			m_err = m.row(0) - mrep.row(0);
			m_err = m.row(1) - mrep.row(1);
			//m_err = m_err(:); ne demek

			Mat MMM2 = (Mat::ones(3, 1, CV_8U)*mrep.row(0).dot(MMM));
			Mat MMM3 = (Mat::ones(3, 1, CV_8U)*mrep.row(1).dot(MMM));

			Mat trMMM2, trMMM3;
			transpose(MMM2, trMMM2);
			transpose(MMM3, trMMM3);
			ii = 0, jj = 0;
			for (int i = 0; i < 2 * Np; i += 2, ++ii)
				for (int j = 6; j < 8; j++, ++jj) {
					L.at<float>(i, j) = trMMM2.at<float>(ii, jj);
				}
			ii = 0, jj = 0;
			for (int i = 1; i < 2 * Np; i += 2, ++ii)
				for (int j = 6; j < 8; j++, ++jj) {
					L.at<float>(i, j) = trMMM3.at<float>(ii, jj);
				}
			transpose(M / (Mat::ones(3, 1, CV_8U)*mrep.row(2)), MMM);
			Mat trJ;
			transpose(J, trJ);
			Mat hh_innov = (trJ*J).inv*m_err;
			Mat hhv_up = hhv - hh_innov;

			//Mat H_up = hhv_up.reshape(0, 9);
			Mat H_up = hhv_up / Hrem(3, 3);

		}
	}








	return H;
	
	return findHomography(m, M);
	
}
*/
Mat p2d_rectangularM(Mat InputImage,Mat Vertices, int Interpolation) {
	//Mat A(1, 3, CV_64F);
	Mat P1 = Vertices.row(0);//top-left corner
	Mat P2 = Vertices.row(1);//top-right corner
	Mat P3 = Vertices.row(2);//bottom-right corner
	Mat P4 = Vertices.row(3);//bottom-left corner

	int ImageW = ceil(max(norm(P1 - P2), norm(P3 - P4)));
	int ImageH = ceil(max(norm(P2 - P3), norm(P1 - P4)));
	cout << ImageH << "xxx" << ImageW << endl;

	//int dummy_query_data[12] = { 0, 0,1,
	//							ImageW - 1, 0,1,
	//							ImageW - 1, ImageH - 1,1,
	//							0, ImageH - 1,1 };

	//Mat TargetVertices = Mat(4, 3, CV_32FC1, dummy_query_data);

	//Mat trVertices, trTargetVertices;
	//transpose(Vertices, trVertices);
	//transpose(TargetVertices, trTargetVertices);
	//Mat Hnorm, inv_Hnorm;

	//cout << trVertices << endl;
	//cout << trTargetVertices << endl;
	//todo transposes go
	//Mat H = compute_homographyM(Vertices, TargetVertices, Hnorm, inv_Hnorm);
	float homo_data[9] = { 0.7718, -0.2402, 172,
						  -0.0098,  0.5358, 287,
						   0,       0,      1 };

	Mat H = Mat(3, 3, CV_32FC1,homo_data);

	cout << "HOMOGRAPHY\n";
	cout << H << endl;
	//return H;
	int plain = 3;
	int sizes[3] = { ImageH, ImageW, plain };//H*I*D
	Mat OutputImage=Mat::zeros(3,sizes, CV_32FC1);	
	//cout << OutputImage.rows << "x" << OutputImage.cols << "dims:" << OutputImage.dims << endl;
	
	Mat P_o;
	Mat P_i;
	const int dims[3] = { 1,1,3 };
	Mat Pixel = Mat::zeros(3, dims, CV_32FC1);
	for (int i = 0; i < ImageW-2; i++){
		for (int j = 0; j < ImageH-2; j++){	
		//	P_o.at<float>(0,0) = i;
		//	P_o.at<float>(1) = j;
			//P_o.at<float>(2) = 1;
			transpose(Matx13f(i,j,1), P_o);
			P_i = H*P_o;//3x3 x 3x1 = 3x1
			
			P_i = P_i.rowRange(0,2).col(0) / P_i.at<float>(Point(2,0));//???
			//cout << P_i << endl;
			//Mat interpolated = p2d_interpolateM(InputImage, P_i, Interpolation);
			//cout << "interpolated" << interpolated<<endl;
			//cout << i1 << "-" << i2 <<endl;
			OutputImage.row(j).col(i) =  p2d_interpolateM(InputImage, P_i, Interpolation);
		}
	}
	OutputImage = OutputImage / 255.0;
	return OutputImage;
}
Mat p2d_interpolateM(Mat Image, Mat P, int Method) {
	const int dims[3] = { 1,1,3 };
	Mat Pixel = Mat::zeros(3, dims, CV_32FC1);
	switch (Method){
		case 1:
		{
			int x = P.at<float>(Point(0, 0));
			int y = P.at<float>(Point(1, 0));
			if (x >= 0 && x< Image.cols && y>= 0 && y<Image.rows)
				Pixel= Image.col(x).row(y);
			break;
		}

		case 2:
		{
			float P_0 = P.at<float>(Point(0, 0));
			float P_1 = P.at<float>(Point(1, 0));
			float dx = P_1 - floor(P_1);
			float dy = P_0 - floor(P_0);
			cout << P_0 << endl;
			cout << P_1 << endl;
			cout << dx << endl;
			cout << dy << endl;			
			Mat P00 = Image.col(floor(P_1)).row(floor(P_0));
			Mat P01 = Image.col(floor(P_1)).row(ceil(P_0));
			Mat P10 = Image.col(ceil(P_1)).row(floor(P_0));
			Mat P11 = Image.col(ceil(P_1)).row(ceil(P_0));
			Mat Pixel1 = (1 - dx)*P00.col(0).row(0) + dx*P10.col(0).row(0);
			Mat Pixel2 = (1 - dx)*P01.col(0).row(0) + dx*P11.col(0).row(0);
			Pixel = (1 - dy)*Pixel1.col(0).row(0) + dy*Pixel2.col(0).row(0);

			break;
		}

		case 3:
		{
			Pixel = Image.col(ceil(P.at<float>(1))).row(ceil(P.at<float>(0)));
			break;
		}

		default:
		{
			cout << "TYPE SELECTION ERROR" << endl;
			break;
		}
	}
	return Pixel;
}

#endif // !_HEADER_