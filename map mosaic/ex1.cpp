
#include "stdafx.h" 

#include <stdio.h>
#include <stdlib.h> 
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <iostream>
#include <ctime>
//#include <cmath>
#include <math.h> 
#include "string.h"


using namespace std;
using namespace cv;

Mat img_matches, ransac;
Size size1;


// Canny
Mat src, src_gray;
Mat dst, detected_edges;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
//
Mat white_out, white_out1, common;
//
Mat src2, dst2;
int morph_elem = 1;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;
char* window_name = "Morphology";

vector<Point> contoursConvexHull( vector<vector<Point> > contours )
{
    vector<Point> result;
    vector<Point> pts;
    for ( size_t i = 0; i< contours.size(); i++)
        for ( size_t j = 0; j< contours[i].size(); j++)
            pts.push_back(contours[i][j]);
    convexHull( pts, result );
    return result;
}

Mat change_color(Mat img, int blue1, int green1, int red1, int blue2, int green2, int red2) {

	int r = 0, c, i, j;
	int cols = img.cols;
	int rows = img.rows;

	Mat output(img.size(), img.type());
	img.copyTo(output);
	
	for (i = 0; i < rows; i++)
    {
        c = 0;
        for (j = 0; j < cols; j++)
        {	
			if(output.at<Vec3b>(Point(c, r))[0] == blue1 && output.at<Vec3b>(Point(c, r))[1] == green1 && output.at<Vec3b>(Point(c, r))[2] == red1) {
				
				output.at<Vec3b>(Point(c, r))[0] = blue2;
				output.at<Vec3b>(Point(c, r))[1] = green2;
				output.at<Vec3b>(Point(c, r))[2] = red2;
				
			}
            c++;
        }
        r++;
    }

	return output;
}

void calc_r(Mat input_img1, Mat input_img2, int minHessian) {
	Mat img1, img2;
	cvtColor(input_img1, img1, CV_RGB2GRAY);
	cvtColor(input_img2, img2, CV_RGB2GRAY);
	vector<KeyPoint> keypoints1, keypoints2;
	SurfFeatureDetector detector(minHessian);
	detector.detect(img1, keypoints1);
	detector.detect(img2, keypoints2);
	printf("[2] Keypoints1 amount: %d\nKeypoints2 amount: %d\n", (int)keypoints1.size(), (int)keypoints2.size());
	Mat descriptors1, descriptors2;
	SurfDescriptorExtractor extractor;
	extractor.compute(img1, keypoints1, descriptors1);
	extractor.compute(img2, keypoints2, descriptors2);
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( descriptors1, descriptors2, matches );
	double max_dist = 0;
	double min_dist = 1000;
	double r = 0;

	for( int i = 0; i < descriptors1.rows; i++ ) { 
		double dist = matches[i].distance;
		r += dist;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	double R = r/(double)descriptors1.rows;

	printf("[2] Total matches amount: %d\n", (int)matches.size());
	printf("[2] Max dist : %f \n", max_dist );
	printf("[2] Min dist : %f \n", min_dist );
	printf("[2] Total r: %f , m: %d, R: %f\n", r, descriptors1.rows, R );

	keypoints1.clear();
	keypoints2.clear();
	matches.clear();
}

void h_compare(Mat img1, Mat img2, string name, bool flag) {
	Mat h(Size(img1.rows+img2.rows, max(img1.cols, img2.cols)), img1.type());
	hconcat(img1, img2, h);
	if (flag)
		h = change_color(h, 0, 0, 0, 255, 255, 255);
	imshow(name, h);
	imwrite(name, h);
}

Mat blending(Mat img1, Mat img2, Mat mask1, double alpha) {

	//imshow("input1", input_img1);
	Mat blend(img1.size(), img1.type()); //
	blend = img1 + img2;
	//blend = Scalar::all(255);


	int cols = mask1.cols;
	int rows = mask1.rows;
	int r = 0, c, i, j;

    int h, w; 

	Vec3b color_mask1, color_1, color_2, color_b; // Цветовые маски

    double beta = ( 1.0 - alpha );

	// Проходим по всему изображению
	for (i = 0; i < rows; i++) {
        c = 0;
        for (j = 0; j < cols; j++) {	
			color_mask1 = mask1.at<Vec3b>(Point(c, r));
			color_1 = img1.at<Vec3b>(Point(c, r));
			color_2 = img2.at<Vec3b>(Point(c, r));
			color_b = blend.at<Vec3b>(Point(c, r));

			// Если пиксель находится в общей части
			if(mask1.at<Vec3b>(Point(c, r))[0] == 255 && mask1.at<Vec3b>(Point(c, r))[1] == 255 && mask1.at<Vec3b>(Point(c, r))[2] == 255) {
				
				addWeighted( img1.at<Vec3b>(Point(c, r)), alpha, img2.at<Vec3b>(Point(c, r)), beta, 0.0, blend.at<Vec3b>(Point(c, r)));
			} 
			
            c++;
        }
        r++;
    }

	return blend;
}

Mat common_bright(Mat img, Mat mask, int shift) {

	Mat output(img.size(), img.type());
	img.copyTo(output);

	int cols = mask.cols;
	int rows = mask.rows;
	int r = 0, c, i, j;

	int blue, green, red = 0;

	Vec3b color_mask, color_out; // Цветовые маски

	// Проходим по всему изображению
	for (i = 0; i < rows; i++) {
        c = 0;
        for (j = 0; j < cols; j++) {	
			color_mask = mask.at<Vec3b>(Point(c, r));
			// Если пиксель находится в общей части
			if(color_mask[0] == 255 && color_mask[1] == 255 && color_mask[2] == 255) {
			//printf("color mask: %d %d %d\n", color_mask[0], color_mask[1], color_mask[2]);
				
				blue = output.at<Vec3b>(Point(c, r))[0];
				green = output.at<Vec3b>(Point(c, r))[1];
				red = output.at<Vec3b>(Point(c, r))[2];

				blue += shift;
				if (blue > 255) blue = 255;
				green += shift;
				if (blue > 255) green = 255;
				red += shift;
				if (red > 255) red = 255;

				output.at<Vec3b>(Point(c, r))[0] = blue;
				output.at<Vec3b>(Point(c, r))[1] = green;
				output.at<Vec3b>(Point(c, r))[2] = red;

			}
            c++;
        }
        r++;
    }

	return output;
}

void Morphology_Operations( int, void* )
{
  // Since MORPH_X : 2,3,4,5 and 6
  int operation = morph_operator + 2;

  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

  /// Apply the specified morphology operation
  //morphologyEx( src2, dst2, operation, element );
  dilate( src2, dst2, element );
  imshow( window_name, dst2 );
  }

Mat cannyth(Mat img, double min_th, double max_th, Scalar color) {

  Mat src(img.size(), img.type());
  img.copyTo(src);

  /// Convert the image to grayscale
  cvtColor(src, src_gray, CV_BGR2GRAY );

  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(11, 11) );
  //imshow("blur", detected_edges);

  /// Canny detector
  Canny(detected_edges, detected_edges, min_th, max_th, kernel_size);

  /// Using Canny's output as a mask, we display our result
  Mat canny(src.size(), src.type());
  Mat pen(src.size(), src.type());

  canny = Scalar::all(0);
  pen = color;
  pen.copyTo( canny, detected_edges);

  // find the contours
  vector< vector<Point> > contours;
  findContours(detected_edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); //
  Mat draw_cont1(src.size(), src.type());
  draw_cont1 = Scalar::all(0);
  Mat draw_cont2(src.size(), src.type());
  draw_cont2 = Scalar::all(0);


  // CV_FILLED fills the connected components found
  drawContours(draw_cont1, contours, -1, color, 1);//, CV_FILLED);

  vector< vector<Point> > contours2;
  double area, length;
  double min_area = contourArea(contours[0]);
  double max_area = 0;

  double min_l = arcLength(contours[0], true);
  double max_l = 0;

  int min_index = 0, max_index = 0;
  Point2f center;
  float radius;

  for(int i = 0; i < contours.size(); ++i) {
        
		length = arcLength(contours[i], true);
        area = contourArea(contours[i]);
		if (length < min_l) {
			min_l = length;
			min_index = i;
		}
		if (length > max_l) {
			max_l = length;
			max_index = i;
		}
		if (area < min_area) {
			min_area = area;
			min_index = i;
		}
		if (area > max_area) {
			max_area = area;
			max_index = i;
		}
        //minEnclosingCircle(contours[i], center, radius);
 
        //draw contour property value at the contour center.
        char buffer[64] = {0};      
        //sprintf(buffer, "Area: %.2lf", area);
        //putText(draw_cont, buffer, center, FONT_HERSHEY_SIMPLEX, .4, Scalar(255), 1);
 
        //sprintf(buffer, "Length: %.2lf", length);
        //putText(draw_cont, buffer, Point(center.x,center.y+20), FONT_HERSHEY_SIMPLEX, .4, Scalar(255), 1);
    }
	
	// vol 1
	for(int i = 0; i < contours.size(); ++i) {
		if (arcLength(contours[i], true) > max_l/50.0) {
			contours2.push_back(contours[i]);
		}

		/*if (contourArea(contours[i]) > max_area/20.0) {
			contours2.push_back(contours[i]);
		}*/
			
	}

	

	// vol 2
	/*contours2.resize(contours.size());
    for( int k = 0; k < contours2.size(); k++ )
        approxPolyDP(Mat(contours[k]), contours2[k], 2, true);*/

	//draw_cont1 = Scalar::all(0);
	draw_cont1.copyTo(src2);
	drawContours(draw_cont2, contours2, -1, color, 1);//, CV_FILLED);
	//imshow("draw_cont1", draw_cont1);


	// vol 3 morph
	/*morph_size = 1;
	Mat element = getStructuringElement( morph_elem, Size( 1*morph_size + 1, 1*morph_size+1 ), Point( morph_size, morph_size ) );
	draw_cont2.copyTo(src2);
	dilate( src2, draw_cont2, element );*/
	//erode( draw_cont2, draw_cont2, element );

	// approx
	/*vector<Point> ConvexHullPoints =  contoursConvexHull(contours2);
	polylines( draw_cont2, ConvexHullPoints, true, Scalar(0,0,255), 1 );
    imshow("contoursConvexHull", draw_cont1);*/

	//drawContours(draw_cont2, contours2, -1, color, 1);

	//imshow("cont2", draw_cont2);

	
	//double min_length = arcLength(contours[min_index], true);
 //   minEnclosingCircle(contours[min_index], center, radius);
	//char buffer1[64] = {0};      
 //   sprintf(buffer1, "min area: %.2lf", min_area);
	//printf("min area: %.2lf\n", min_area);
 //   //putText(draw_cont, buffer1, center, FONT_HERSHEY_SIMPLEX, .4, Scalar(0,0,255), 1);
	//sprintf(buffer1, "Length: %.2lf", min_length);
	//printf("Min length: %.2lf\n", min_l);
 //   //putText(draw_cont, buffer1, Point(center.x,center.y+20), FONT_HERSHEY_SIMPLEX, .4, Scalar(0,0,255), 1);

	//double max_length = arcLength(contours[max_index], true);
	//minEnclosingCircle(contours[max_index], center, radius);
	//char buffer2[64] = {0};      
 //   sprintf(buffer2, "Max area: %.2lf", max_area);
	//printf("Max area: %.2lf\n", max_area);
 //   //putText(draw_cont, buffer2, center, FONT_HERSHEY_SIMPLEX, .4, Scalar(0,0,255), 1);
	//sprintf(buffer2, "Length: %.2lf", max_length);
	//printf("Max length: %.2lf\n", max_l);
    //putText(draw_cont, buffer2, Point(center.x,center.y+20), FONT_HERSHEY_SIMPLEX, .4, Scalar(0,0,255), 1);


	

  /*Mat comp_cont(Size(draw_cont1.cols + draw_cont2.cols, 2*max(draw_cont1.rows,draw_cont2.rows)), CV_8UC3, Scalar(0));
  hconcat(draw_cont1, draw_cont2, comp_cont);*/

  h_compare(draw_cont1, draw_cont2, "output/compare contours.jpg", false);


  //imshow("draw cont1", draw_cont1);

//  // morph -----------------------
//  /// Create window
//  
// draw_cont1.copyTo(src2);
// namedWindow( window_name, CV_WINDOW_AUTOSIZE );
// 
///// Create Trackbar to select Morphology operation
// createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, Morphology_Operations );
//
// /// Create Trackbar to select kernel type
// createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
//                 &morph_elem, max_elem,
//                 Morphology_Operations );
//
// /// Create Trackbar to choose kernel size
// createTrackbar( "Kernel size:\n 2n +1", window_name,
//                 &morph_size, max_kernel_size,
//                 Morphology_Operations );
//
// /// Default start
// Morphology_Operations( 0, 0 );
//
// // ------------------------------------

  //imshow("draw cont2", draw_cont2);
  //imshow("compare contours", comp_cont);
  //imwrite("output/morph.jpg", comp_cont);

  contours.clear();
  contours2.clear();
  return draw_cont2;
}

Mat common_area(Mat im1, Mat im2) {

	Mat img1(im1.size(), im1.type());
	Mat img2(im2.size(), im2.type());
	im1.copyTo(img1);
	im2.copyTo(img2);
	int cols = img1.cols;
	int rows = img1.rows;

	int r1 = 0, r2 = 0, c1, c2, i, j;
	for (i = 0; i < rows; i++)
    {
        c1 = 0;
        for (j = 0; j < cols; j++)
        {	
			Vec3b color = img1.at<Vec3b>(Point(c1, r1));
			//printf("COLOR of POINT [%d:%d]: %d %d %d\n\n", r,c, color[0], color[1], color[2]);
			if ((int)color[0] != 0 && (int)color[1] != 0 && (int)color[2] != 0) {
				img1.at<Vec3b>(Point(j, i))[0] = 255;
				img1.at<Vec3b>(Point(j, i))[1] = 255;
				img1.at<Vec3b>(Point(j, i))[2] = 0;
				
			}		
            c1++;
        }
        r1++;
    }

	for (i = 0; i < rows; i++)
    {
        c2 = 0;
        for (j = 0; j < cols; j++)
        {	
			Vec3b color = img2.at<Vec3b>(Point(c2, r2));
			//printf("COLOR of POINT [%d:%d]: %d %d %d\n\n", r,c, color[0], color[1], color[2]);
			if ((int)color[0] != 0 && (int)color[1] != 0 && (int)color[2] != 0) {
				img2.at<Vec3b>(Point(j, i))[0] = 0;
				img2.at<Vec3b>(Point(j, i))[1] = 0;
				img2.at<Vec3b>(Point(j, i))[2] = 255;
				
			}		
            c2++;
        }
        r2++;
    }

	//imshow("img1", img1);
	//imshow("img2", img2);

	return img1 + img2;
}

void qa(Mat im1, Mat im2, Mat mask, Mat pan, vector<Vec4i> li1, vector<Vec4i> li2) {

	//imshow("input1", input_img1);
	Mat common_pix(pan.size(), pan.type()); //
	Mat canny_sum(im1.size(), im1.type()); // Для суммы двух контуров

	// Преобразуем цвета контуров
	Mat img1 = change_color(im1, 255, 255, 255, 255, 255 ,0);
	Mat img2 = change_color(im2, 255, 255, 255, 0, 0, 255);

	// Сумма контуров
	canny_sum = img1 + img2;

	Mat bw_canny_sum;
	canny_sum.copyTo(bw_canny_sum);
	bw_canny_sum = change_color(bw_canny_sum, 255, 255, 255, 55, 55 ,55);
	bw_canny_sum = change_color(bw_canny_sum, 0, 0, 0, 255, 255 ,255);
	bw_canny_sum = change_color(bw_canny_sum, 255, 255, 0, 0, 0 ,0);
	bw_canny_sum = change_color(bw_canny_sum, 0, 0, 255, 150, 150 ,150);
	//imshow("bw canny sum", bw_canny_sum);

	// Изображения для отображения оценки качества
	Mat qa1(canny_sum.size(), canny_sum.type());
	Mat qa2(canny_sum.size(), canny_sum.type());
	canny_sum.copyTo(qa1);
	canny_sum.copyTo(qa2);
	
	pan.copyTo(common_pix); // Копируем панораму
	int cols = mask.cols;
	int rows = mask.rows;
	int r = 0, c, i, j;

	int total = 0; // Общее количество пикселей в общей части
	int good = 0; // Количество всех совпавших пикселей в общей части
	int not_black_pix = 0, white_pix = 0; // Значения для подсчета совпадения пикселей
	int x0 = cols, x1 = 0, y0 = rows, y1 = 0;
    int h, w; 

	Vec3b color_mask, color_1, color_2, sum_mask; // Цветовые маски

	// Проходим по всему изображению
	for (i = 0; i < rows; i++) {
        c = 0;
        for (j = 0; j < cols; j++) {	
			color_mask = mask.at<Vec3b>(Point(c, r));
			color_1 = im1.at<Vec3b>(Point(c, r));
			color_2 = im2.at<Vec3b>(Point(c, r));
			//sum_mask = canny_sum.at<Vec3b>(Point(c, r));
			
			// Если пиксель находится в общей части
			if(color_mask[0] == 255 && color_mask[1] == 255 && color_mask[2] == 255) {
			//printf("color mask: %d %d %d\n", color_mask[0], color_mask[1], color_mask[2]);
				
				// Суммируем все совпавшие пиксели
				//if (canny_sum.at<Vec3b>(Point(c, r))[0] == 255 && canny_sum.at<Vec3b>(Point(c, r))[1] == 255 && canny_sum.at<Vec3b>(Point(c, r))[2] == 255)
				//	white_pix++;

				// Суммируем все не черные (не совпавшие пиксели и не фон)
				//if (canny_sum.at<Vec3b>(Point(c, r))[0] != 0)
				//	not_black_pix++;

				//if(color_1[0] != 0 && color_1[1] != 0 && color_1[2] != 0 && color_2[0] != 0 && color_2[1] != 0 && color_2[2] != 0) {
				//	common_pix.at<Vec3b>(Point(c, r))[2] = 255;
				//}

				// Находим границы общей части
				if(c < x0) x0 = c;
				if(c > x1) x1 = c;
				if(r < y0) y0 = r;
				if(r > y1) y1 = r;
			
			}
            c++;
        }
        r++;
    }

	// Размеры общей части
	w = x1 - x0;
	h = y1 - y0;

	// Размер окна для оценки качества
	int w_size = 45; 

	// Расширим область для разбиения на окна
	int w_add = w % w_size;
	int h_add = h % w_size;
	w += (w_size - w_add);
	h += (w_size - h_add);

	// Чтобы не выйти за пределы изображения при расширении
	if (w > cols) w -= w_size;
	if (h > rows) h -= w_size;
	
	printf("common part - width: %d, height: %d\n", w, h);
	// Добавим внешнюю рамку
	//rectangle(qa1, Point2f(x0, y0), Point2f(x0 + w, y0 + h), Scalar(0,0,255), 2);
	//rectangle(qa2, Point2f(x0, y0), Point2f(x0 + w, y0 + h), Scalar(0,0,255), 2);

	int pix1, pix2, black_pix, empty_boxes = 0;
	double ai, a = 0.0, pi, p = 0.0;
	int c1 = 0, r1 = 0;
	int c_roi = 0, r_roi = 0;
	int white_pix1, not_black_pix1;

	
	// Проходим по всей области для оценки
	for (int wi = x0; wi < x0+w; wi += w_size) {
		for (int hi = y0; hi < y0+h; hi += w_size) {
			pix1 = 0; // количество пикселей первого изображения в окне
			pix2 = 0; // количество пикселей второго изображения в окне
			black_pix = 0; 
			ai = 0.0; // локальная оценка качества
			pi = 0.0;

			// Нарисуем границы окон
			rectangle(qa1, Point2f(wi, hi), Point2f(wi+w_size, hi + w_size), Scalar(0,255,255), 0.5);
			rectangle(qa2, Point2f(wi, hi), Point2f(wi+w_size, hi + w_size), Scalar(0,255,255), 0.5);

			white_pix1 = 0; // Количество локальных белых пикселей в окне
			not_black_pix1 = 0; // Количество не черных пикселей в окне

			for (int ri = 0; ri < w_size; ri++) {
				//c1 = wi + ci;
				for (int ci = 0; ci < w_size; ci++) {
					c1 = wi + ci;
					r1 = hi + ri;

					color_1 = im1.at<Vec3b>(Point(c1, r1));
					color_2 = im2.at<Vec3b>(Point(c1, r1));
					sum_mask = canny_sum.at<Vec3b>(Point(c1, r1));

					// Оценка на основе межпиксельного расстояния
					if (color_1[0] == 255 && color_1[1] == 255 && color_1[2] == 255) pix1++;
					if (color_2[0] == 255 && color_2[1] == 255 && color_2[2] == 255) pix2++;
					
					// Оценка по совпадению пикселей в окне
					if (canny_sum.at<Vec3b>(Point(c1, r1))[0] == 255 && canny_sum.at<Vec3b>(Point(c1, r1))[1] == 255 && canny_sum.at<Vec3b>(Point(c1, r1))[2] == 255)
					{	
						//printf("color mask: %d %d %d\n", canny_sum.at<Vec3b>(Point(c1, r1))[0], canny_sum.at<Vec3b>(Point(c1, r1))[1], canny_sum.at<Vec3b>(Point(c1, r1))[2]);
						white_pix1++;
					}
					if ((canny_sum.at<Vec3b>(Point(c1, r1))[0] > 0) && (canny_sum.at<Vec3b>(Point(c1, r1))[1] > 0) && (canny_sum.at<Vec3b>(Point(c1, r1))[2] > 0))
					{
						not_black_pix1++;
						//printf("sum: %d - %d - %d\n", sum_mask[0], sum_mask[1], sum_mask[2]);
						//printf("sum: %d", sum_mask[0]);
					}

				}

			}

			// Считаем окна без контуров
			if (pix1 == 0 && pix2 == 0) empty_boxes++;

			// Вычисляем локальную оценку на основе межпиксельного расстояния
			if ((pix1 == 0) || (pix2 == 0)) ai = 0.0;
			else { //ai = double(pix2)/double(pix1);	
				if (pix2 > pix1)  
					ai = double(pix1)/double(pix2);
				else 
					ai = double(pix2)/double(pix1);
			}

			//printf("pix1 = %d, pix2 = %d, ai = %.2f\n", pix1, pix2 , ai);
			
			// Суммируем локальные оценки
			if (ai < 0.4) empty_boxes++;
			else a += ai;

			

			if (ai >= 0.3) {
				//if (ai > 0.8) ai -= 0.1;
				char buffer1[64] = {0};      
				sprintf(buffer1, "%.2f", ai);
				//printf("ai: %.2f\n", ai);
				//putText(qa1, buffer1, Point(wi+1,hi+8), FONT_ITALIC, 0.3, Scalar(0,255,255), 1);
				putText(qa1, buffer1, Point(wi+5,hi+15), FONT_ITALIC, 0.5, Scalar(0,255,255), 2);

				char buffer2[64] = {0};
				sprintf(buffer2, "%d:%d", pix2,pix1);
				//printf("%d/%d\n", pix2,pix1);
				//putText(qa1, buffer2, Point(wi+1,hi+18), FONT_ITALIC, 0.27, Scalar(0,255,255), 1);

				c_roi = wi;
				r_roi = hi;

			}

			// Вычисляем локальную оценку по пикселям
			if ((not_black_pix1) == 0 || white_pix1 == 0) pi = 0.0;
			//else pi = double(2 * white_pix1)/(white_pix1 + not_black_pix1);
			else pi = double(white_pix1)/double(not_black_pix1/2.0);
			//printf("white = %d, not black = %d, pi = %.2f\n", white_pix1, not_black_pix1 , pi);
			p += pi;

			if (pi > 0) {
				char buffer3[64] = {0};      
				sprintf(buffer3, "%.2f", pi);
				//printf("pi: %.2f\n", pi);
				//putText(qa2, buffer3, Point(wi+2,hi+15), FONT_ITALIC, 0.3, Scalar(0,255,255), 1);
			}		

		}

	}

	// поиск прямых линий
	Mat canny_line1(im1.size(), im1.type());
	Mat canny_line2(im2.size(), im2.type());
	
	canny_line1 = im1.clone();
	canny_line2 = im2.clone();
	canny_line1 = Scalar::all(255) - canny_line1;
	canny_line2 = Scalar::all(255) - canny_line2;

	//------------------
	int dc = 30; // Порог по расстоянию
	double da = 10.0; // Порог по углу
	Vec4i l1, l2;
	double angle1, angle2;
	double Kangle = 0.0, Kan = 0.0, dangle;
	int Kn = 0;
	for( int i = 0; i < li1.size(); i++ )	{
		for( int j = 0; j < li2.size(); j++ )	{
			l1 = li1[i];
			l2 = li2[j];
			// Проверяем, что концы отрезков лежат в общей части двух изображений
			if (l1[0] >= x0 && l1[0] <= (x0 + w) && l1[2] >= x0 && l1[2] <= (x0 + w) && l1[1] >= y0 && l1[1] <= (y0 + h) && l1[3] >= y0 && l1[3] <= (y0 + h)
				&& l2[0] >= x0 && l2[0] <= (x0 + w) && l2[2] >= x0 && l2[2] <= (x0 + w) && l2[1] >= y0 && l2[1] <= (y0 + h) && l2[3] >= y0 && l2[3] <= (y0 + h)) {
					// Проверяем, что координаты концов отрезков лежат друг от друга не далее чем значение dc
					if (abs(l1[0]-l2[0])<dc && abs(l1[1]-l2[1])<dc && abs(l1[2]-l2[2])<dc && abs(l1[3]-l2[3])<dc) {
						angle1 = atan2((double)l1[3] - (double)l1[1], (double)l1[2] - (double)l1[0]) * 180 / CV_PI;
						angle2 = atan2((double)l2[3] - (double)l2[1], (double)l2[2] - (double)l2[0]) * 180 / CV_PI;
						if (angle1 < 0) angle1 = -angle1;
						if (angle2 < 0) angle2 = -angle2;
						// Проверяем, что разность угол между отрезками не превышает значение da
						Point2f diff1 = Point(l1[0], l1[1]) - Point(l1[2], l1[3]);
						double linelength = sqrt(diff1.x*diff1.x + diff1.y*diff1.y);
						if (abs(angle1 - angle2) < da ) {
							line( canny_line1, Point(l1[0], l1[1]), Point(l1[2], l1[3]), Scalar(0,0,255), 2, CV_AA);
							line( canny_line2, Point(l2[0], l2[1]), Point(l2[2], l2[3]), Scalar(0,0,255), 2, CV_AA);
							line( bw_canny_sum, Point(l1[0], l1[1]), Point(l1[2], l1[3]), Scalar(0,0,255), 2, CV_AA);
							line( bw_canny_sum, Point(l2[0], l2[1]), Point(l2[2], l2[3]), Scalar(255,0,0), 2, CV_AA);
							char ang1[64] = {0};      
							sprintf(ang1, "%.2f", angle1);
							putText(canny_line1, ang1, Point(l1[0]-10, l1[1]+15), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255), 2);
							char ang2[64] = {0};      
							sprintf(ang2, "%.2f", angle2);
							putText(canny_line2, ang2, Point(l2[0]-10, l2[1]+15), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255), 2);
							
							Point2f pt1(l2[0], l2[1]);
							Point2f pt2(l2[2], l2[3]);
							LineIterator it2(canny_line2, pt1, pt2, 8);
							Point2f ptc((l1[0]+l1[2])/2, (l1[1]+l1[3])/2); // центр первого отрезка
							//printf("pt1.x = %f, pt1.y = %f\n", pt1.x, pt1.y);
							//rintf("pt2.x = %f, pt2.y = %f\n", pt2.x, pt2.y);

							double d = 0.0, di = 0.0;
							for (int i=0; i < it2.count; i++, ++it2) {
								Point2f it2p = it2.pos(); // текущая точка отрезка
								Point2f diff = it2p - ptc;
								//if (i == 0 || i == it2.count-1) line( canny_line2, it2p, ptc, Scalar(0,255,0), 1, CV_AA);
								//line( canny_line2, it2p, ptc, Scalar(0,255,0), 1, CV_AA);
								//circle(canny_line2, it2p, 5, Scalar(0,255,255), 1);
								di = sqrt(diff.x*diff.x + diff.y*diff.y);
								//printf("[%d] di = %f\n", i, di);
								//printf("ptc.x = %f, ptc.y = %f\n", ptc.x, ptc.y);
								//di = norm(it2.pos(), ptc);
								d += di;

							}
							//printf("total d = %f\n", d);
							d = d/it2.count;
							//printf("avg d = %f\n", d);
							dangle = abs(angle1-angle2);

							Kan = d * cos(dangle * CV_PI / 180);
							Kangle += Kan;
							Kn += 1;
							//printf("angle1 = %f, angle2 = %f, dangle = %f, cos(angle) = %f\n\n", angle1, angle2, dangle, cos(dangle));
							char ang3[64] = {0};      
							sprintf(ang3, "%.4f", dangle);
							putText(bw_canny_sum, ang3, Point(l1[0]-10, l1[1]+15), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0), 2);
						}
					}
			}
		}
	}
	Kangle = Kangle/(double)Kn;
	//h_compare(canny_line1, canny_line2, "output/lines.jpg", false);
	//imshow("bw canny sum", bw_canny_sum);
	imwrite("output/bw_canny_sum.jpg", bw_canny_sum);

	//------------------

	int total_boxes = w * h / (w_size * w_size);
	int full_boxes = total_boxes - empty_boxes;
	double K1 = double(a)/double(full_boxes);
	double K2 = double(p)/double(full_boxes);

	printf("Empty boxes: %d, total_boxes: %d, full_boxes: %d\n", empty_boxes, total_boxes, full_boxes);
	printf("Total a: %f\n\nFinal K: %f\n\n", a, K1);

	char buffer4[64] = {0};      
	sprintf(buffer4, "K = %.3f", K1);
	putText(qa1, buffer4, Point(cols/4-5,rows/8), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,255), 2);

	char buffer5[64] = {0};      
	sprintf(buffer5, "K = %.3f", K2);
	//putText(qa2, buffer5, Point(cols/4-5,rows/8), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,255), 2);

	// Рассчитываем оценку по пикселям во всей общей части
	/*good = 2 * white_pix;
	total = white_pix + not_black_pix;
	double alpha_pix = (double)good/(double)total;
	printf("[NO INTEGRAL] Total pixels: %d, good pixels: %d\nalpha pix: %f\n", total, good, alpha_pix);*/

	imshow("Оценка на основе межпиксельного расстояния", qa1);
	//imshow("Оценка по пикселям", qa2);
	imwrite("output/qa1.jpg", qa1);
	//imwrite("output/qa2.jpg", qa2);
	Mat bw_qa;
	qa1.copyTo(bw_qa);
	bw_qa = change_color(bw_qa, 255, 255, 255, 55, 55 ,55);
	bw_qa = change_color(bw_qa, 0, 0, 0, 255, 255 ,255);
	bw_qa = change_color(bw_qa, 0, 255, 255, 255, 0 , 0); // цифры 0 
	bw_qa = change_color(bw_qa, 255, 255, 0, 0, 0 ,0);
	bw_qa = change_color(bw_qa, 0, 0, 255, 150, 150 ,150);
	bw_qa = change_color(bw_qa, 255, 0, 0, 100, 100 ,255); // цифры 1
	//imshow("bw qa1", bw_qa);
	imwrite("output/bw_qa1.jpg", bw_qa);
	//h_compare(change_color(pan, 0, 0, 0, 255, 255 ,255), bw_qa, "output/qa good.jpg", false);

	//h_compare(qa1, qa2, "output/qas.jpg", false);

	Mat one_roi(Size(w_size, w_size), qa1.type());
	Rect rect1(c_roi, r_roi, w_size, w_size);
	one_roi = Mat(canny_sum, rect1);
	resize(one_roi, one_roi, cv::Size(), 5.0, 5.0);
	//imshow("one roi", one_roi);
}

Mat calc_h(Mat input_img1, Mat input_img2, string type, int Hes, int method, double th=0) {

	Mat img1, img2;

	// Convert to Grayscale
	cvtColor(input_img1, img1, CV_RGB2GRAY);
	cvtColor(input_img2, img2, CV_RGB2GRAY);

	// detecting keypoints
	int minHessian = Hes;
	
	vector<KeyPoint> keypoints1, keypoints2;

	unsigned int start_time =  clock();

	if (type == "SIFT") {
		SiftFeatureDetector detector(0,3,0.04,2.3,1.6);
		detector.detect(img1, keypoints1);
		detector.detect(img2, keypoints2);

	} else if (type == "SURF") {
		SurfFeatureDetector detector(minHessian);
		detector.detect(img1, keypoints1);
		detector.detect(img2, keypoints2);
	}

	printf("Keypoints1 amount: %d\nKeypoints2 amount: %d\n\n", (int)keypoints1.size(), (int)keypoints2.size());

	// computing descriptors
	Mat descriptors1, descriptors2;
	if (type == "SIFT") {
		SiftDescriptorExtractor extractor;
		extractor.compute(img1, keypoints1, descriptors1);
		extractor.compute(img2, keypoints2, descriptors2);
	} else if (type == "SURF") {
		SurfDescriptorExtractor extractor;
		extractor.compute(img1, keypoints1, descriptors1);
		extractor.compute(img2, keypoints2, descriptors2);	
	}

	unsigned int end_time = clock();
	unsigned int search_time = end_time - start_time;
	printf("-------------- Time: %d\n\n", search_time);

	// matching descriptors
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( descriptors1, descriptors2, matches );

	// кооринаты всех сматченных точек
	/*for( int i = 0; i < (int)matches.size(); i++ )	{
		printf( "Match [%d] -- [x] %f [y] %f \n", i+1, keypoints1[matches[i].queryIdx ].pt.x, keypoints1[matches[i].queryIdx ].pt.y );
	}*/

	double max_dist = 0;
	double min_dist = 1000;

	for( int i = 0; i < descriptors1.rows; i++ ) { 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	printf("Total matches amount: %d\n\n", (int)matches.size());
	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	vector< DMatch > good_matches;
	//Mat img_matches;
	
	double k = 1.0;
	double k_max = 6.0;

	// берем все ключевые точки
	/*for( int i = 0; i < descriptors1.rows; i++ )
		good_matches.push_back( matches[i]);*/

	// берем точки по порогу евклидового расстояния
	while(k < k_max) {
			
		printf("current k: %f \n\n", k);
		for( int i = 0; i < descriptors1.rows; i++ ) { 
			if( matches[i].distance <= max(k * min_dist, 0.01) ) { 
				good_matches.push_back( matches[i]); 
			}
		}
		if (good_matches.size() < 4) {

			if(k == k_max) {
				printf("k_max = %f is reached\n");
				break;
			}
			good_matches.clear();
			k += 0.5;
		} else break;

	}

	// берем точки с прореживанием по координатам

	/*for( int i = 0; i < (int)good_matches.size(); i++ )	{
		printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i+1, good_matches[i].queryIdx, good_matches[i].trainIdx );
	}*/

	//-- Локализация объектов
	vector<Point2f> obj1;
	vector<Point2f> obj2;

	//-- Get the keypoints from the good matches
	for( int i = 0; i < good_matches.size(); i++ ) {
		obj1.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
		obj2.push_back( keypoints2[ good_matches[i].trainIdx ].pt ); 
	}

	// Find the Homography Matrix
	Mat H(3,3,CV_64F);
	Mat ransac_mask;

	printf("Good matches: %d\n\n",obj1.size());
	if(obj1.size() >=4) {
		H = findHomography( obj2, obj1, method, th, ransac_mask);

		cout << "\n\nH:\n"<< H << "\n\n";
	} else {
		printf("There must be 4 good matches at least\n\n");
		//return -1;
	}

	// RANSAC draw
	vector< DMatch > good_matches2;
	int numGood = 0;
	for (int i = 0; i< good_matches.size(); i ++)
    {

        if (ransac_mask.at<uchar>(i) != 0)  // RANSAC selection
        {
   
            good_matches2.push_back(good_matches[i]);

            numGood = numGood + 1;

        }

    }

	//// draw all matches black - white
	//drawMatches(img1, keypoints1, img2, keypoints2, 
	//good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), 
	//vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);// DRAW_RICH_KEYPOINTS NOT_DRAW_SINGLE_POINTS

	// draw all matches color
	drawMatches(input_img1, keypoints1, input_img2, keypoints2, 
	good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), 
	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);// DRAW_RICH_KEYPOINTS NOT_DRAW_SINGLE_POINTS

	// draw ransac matches
	drawMatches(input_img1, keypoints1, input_img2, keypoints2, 
	good_matches2, ransac, Scalar(255,0,0), Scalar(0,0,255),//Scalar::all(-1), Scalar::all(-1), 
	vector<char>(), DrawMatchesFlags::DEFAULT);// DRAW_RICH_KEYPOINTS NOT_DRAW_SINGLE_POINTS

	keypoints1.clear();
	keypoints2.clear();
	obj1.clear();
	obj2.clear();
	good_matches.clear();

	return H;
}

Mat mosaic(Mat img1, Mat img2, Mat mask, Mat H) {

  //Coordinates of the 4 corners of the image
  std::vector<cv::Point2f> corners(4);
  corners[0] = cv::Point2f(0, 0);
  corners[1] = cv::Point2f(0, img2.rows);
  corners[2] = cv::Point2f(img2.cols, 0);
  corners[3] = cv::Point2f(img2.cols, img2.rows);

  vector<Point2f> cornersTransform(4);
  perspectiveTransform(corners, cornersTransform, H);

  double offsetX = 0.0;
  double offsetY = 0.0;

  //Get max offset outside of the image
  for(size_t i = 0; i < 4; i++) {
    //std::cout << "cornersTransform[" << i << "]=" << cornersTransform[i] << std::endl;
    if(cornersTransform[i].x < offsetX) {
      offsetX = cornersTransform[i].x;
    }

    if(cornersTransform[i].y < offsetY) {
      offsetY = cornersTransform[i].y;
    }
  }

  offsetX = -offsetX;
  offsetY = -offsetY;
  std::cout << "offsetX=" << offsetX << " ; offsetY=" << offsetY << std::endl;

  //Get max width and height for the new size of the panorama
  double maxX = std::max((double) img1.cols+offsetX, (double) std::max(cornersTransform[2].x, cornersTransform[3].x)+offsetX);
  double maxY = std::max((double) img1.rows+offsetY, (double) std::max(cornersTransform[1].y, cornersTransform[3].y)+offsetY);
  std::cout << "maxX=" << maxX << " ; maxY=" << maxY << std::endl;

  Size size_warp(maxX, maxY);
  size1 = size_warp;
  Mat panorama(size_warp, CV_8UC3);
  Mat first(size_warp, CV_8UC3);
  first = Scalar::all(0);
  Mat second(size_warp, CV_8UC3);

  //Create the transformation matrix to be able to have all the pixels
  Mat H2 = Mat::eye(3, 3, CV_64F);
  H2.at<double>(0,2) = offsetX;
  H2.at<double>(1,2) = offsetY;

  //H2.at<double>(1,1) += 0.3;
  //H2.at<double>(0,1) += 0.1;

  warpPerspective(img2, panorama, H2*H, size_warp, CV_WARP_FILL_OUTLIERS);
  panorama.copyTo(second);
  //imshow("sec", second);

  //ROI for img1
  Rect img1_rect(offsetX, offsetY, img1.cols, img1.rows);
  Mat half;


  //First iteration
  if(mask.empty()) {

    //Copy img1 in the panorama using the ROI
    Mat half = Mat(panorama, img1_rect);
	Mat half1 = Mat(first, img1_rect);
    //img1.copyTo(half);
	
	// copy pixels from img1 to half if pixel is not black
	int r = 0, c, i, j;
	for (i = 0; i < half.rows; i++)
    {
        c = 0;
        for (j = 0; j < half.cols; j++)
        {	
			Vec3b color = img1.at<Vec3b>(Point(c, r));
			//printf("COLOR of POINT [%d:%d]: %d %d %d\n\n", r,c, color[0], color[1], color[2]);
			if ((int)color[0] != 0 && (int)color[1] != 0 && (int)color[2] != 0) {
				half.at<Vec3b>(Point(j, i)) = img1.at<Vec3b>(Point(c, r));
				half1.at<Vec3b>(Point(j, i)) = img1.at<Vec3b>(Point(c, r));
			}		
            c++;
        }
        r++;
    }

	// 359 - 50 150 | 369 - 40 140 |  401 - 30 130 | 422 - 20 120 | 455 - 10 110

	int min_canny = 10; //10
	int max_canny = 60; //40

	// rho – Distance resolution of the accumulator in pixels.
	// theta – Angle resolution of the accumulator in radians.
	int threshold = 40; // Accumulator threshold parameter. Only those lines are returned that get enough votes
	int minLineLength = 40; // Minimum line length. Line segments shorter than that are rejected.
	int maxLineGap = 20; // Maximum allowed gap between points on the same line to link them.
	// 20 70 10
	// kernel pre canny 11x11
	// def canny length/10
	// canny 10 40

	vector<Vec4i> lines1, lines2;
	Mat canny1 = cannyth(first, min_canny, max_canny, Scalar(255, 255, 255));
	HoughLinesP(detected_edges, lines1, 1, 1*CV_PI/180, threshold, minLineLength, maxLineGap );
	Mat canny2 = cannyth(second, min_canny, max_canny, Scalar(255, 255, 255));
	HoughLinesP(detected_edges, lines2, 1, 1*CV_PI/180, threshold, minLineLength, maxLineGap );


	//imshow("canny1", canny1);
	//imshow("canny2", canny2);
	//h_compare(canny1, canny2, "output/cannys.jpg", false);

	//imshow("canny sum", canny1 | canny2);
	//imwrite("output/canny sum.jpg", canny1 | canny2);

	//Mat canny1_2 = cannyth(panorama, min_canny, max_canny, Scalar(255, 255, 255));
	//imshow("canny1_2", canny1_2);

	//imshow("first", first);
	//imshow("second", second);
	//imwrite("output/second.jpg", second);
	

	common = common_area(first, second);
	//imshow("common part", common);


	qa(canny1, canny2, common, panorama, lines1, lines2);

	//-------------


    //Create the new mask matrix for the panorama
    mask = Mat::ones(img2.size(), CV_8U)*255;
    warpPerspective(mask, mask, H2*H, size_warp, CV_WARP_FILL_OUTLIERS);
    rectangle(mask, img1_rect, Scalar(255), -1);
	//imshow("mask", mask);

  } else {

    //Create an image with the final size to paste img1
    Mat maskTmp = Mat::zeros(size_warp, img1.type());
    half = Mat(maskTmp, img1_rect);
    img1.copyTo(half);

    //Copy img1 into panorama using a mask
    Mat maskTmp2 = Mat::zeros(size_warp, CV_8U);
    half = Mat(maskTmp2, img1_rect);
    mask.copyTo(half);
    maskTmp.copyTo(panorama, maskTmp2);

    //Create a mask for the warped part
    maskTmp = cv::Mat::ones(img2.size(), CV_8U)*255;
    warpPerspective(maskTmp, maskTmp, H2*H, size_warp, CV_WARP_FILL_OUTLIERS);

    maskTmp2 = Mat::zeros(size_warp, CV_8U);
    half = Mat(maskTmp2, img1_rect);
    //Copy the old mask in maskTmp2
    mask.copyTo(half);
    //Merge the old mask with the new one
    maskTmp += maskTmp2;
    maskTmp.copyTo(mask);
  }

  corners.clear();
  cornersTransform.clear();

  //Mat blend = blending(first, second, common, 0.5);
  //h_compare(blend, panorama, "output/two ways.jpg", true);

  calc_r(first, second, 400);


  //h_compare(first, second, "output/first and second.jpg", true);
  //h_compare(panorama, common, "output/common1.jpg", true);

  //return white_out1;
  return panorama;
}

Mat mosaic2(Mat img1, Mat img2, Mat mask, Mat H) {

	Mat panorama(size1, CV_8UC3);
	warpPerspective(img2,panorama,H, size1);//cv::Size(input_img1.cols+input_img2.cols,input_img1.rows+input_img2.rows));
	//imshow("WARP1", result);
	cv::Mat half(panorama,cv::Rect(0,0,img1.cols,img1.rows));
	img1.copyTo(half);
	return panorama;

}

Mat add_border(Mat img, Scalar color, int thinkness) {
	rectangle(img, Point2f(0, 0), Point2f(img.cols, img.rows), color, thinkness);
	return img;
}


int main(int argc, char** argv) {


    char* filename1 = "input/map1.jpg";
	char* filename2 = "input/map2.jpg";
	char* filename3 = "input/map3.png";


    Mat img1 = imread(filename1,1); //CV_LOAD_IMAGE_GRAYSCALE
    Mat img2 = imread(filename2,1);
	Mat img3 = imread(filename3,1);

	
	if(img1.empty() || img2.empty() || img3.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }


	double fx = 1.0;
	double fy = 1.0;

	resize(img1, img1, cv::Size(), fx, fy);
	resize(img2, img2, cv::Size(), fx, fy);
	resize(img3, img3, cv::Size(), fx, fy);

	//Mat all1(Size(img1.cols + img2.cols, max(img1.rows,img2.rows)), CV_8UC3, Scalar(0));
	//Mat all1(Size(max(img1.cols, img2.cols), img1.rows + img2.rows), CV_8UC3, Scalar(0));
	//vconcat(img1, img2, all1);
	//Mat all2(Size(all1.cols + img3.cols, max(all1.rows,img3.rows)), CV_8UC3, Scalar(0));
	//hconcat(all1, img3, all2);
	//imshow("all images", all1);
	//Mat dst3(Size(img1_surf.cols + img2_surf.cols, 2*max(img1_surf.rows,img2_surf.rows)), CV_8UC3, Scalar(0));

	Mat mask;
	Mat H_1_to_2;


	H_1_to_2 = calc_h(img1,img2, "SURF", 400, CV_RANSAC, 5.0);
	//Mat matH_1_to_2_1 = calc_h(img1,img2,400, CV_LMEDS);
	//
	imshow("matches1", img_matches);
	imshow("ransac1", ransac);
	imwrite("output/matches1.jpg", img_matches);
	imwrite("output/ransac1.jpg", ransac);

	Mat panorama1 = mosaic(img1, img2, mask, H_1_to_2);
	panorama1 = change_color(panorama1, 0, 0, 0, 255, 255, 255);
	//Mat panorama11 = mosaic2(img1, img2, mask, H_1_to_2);
	imshow("panorama1", panorama1);
	//imshow("panorama11", panorama11);
	imwrite("output/panorama1.jpg", panorama1);
	//imwrite("output/panorama11.jpg", panorama1);

	//Mat com1 = common;

	//h_compare(panorama11, panorama1, "output/cutting.jpg", true);


	//Mat H_2_to_3 = calc_h(panorama1, img3, "SURF", 400, CV_RANSAC, 3.0);
	//imshow("matches2", img_matches);
	//imwrite("output/matches2.jpg", img_matches);
	//panorama1 = mosaic(panorama1, img3, mask, H_2_to_3);
	////panorama1 = change_color(panorama1, 0, 0, 0, 255, 255, 255);
	//imshow("panorama2", panorama1);
	//imwrite("output/panorama2.jpg", panorama1);

	/*Mat com2 = common;

	white_out = change_color(panorama1, 0, 0, 0, 255, 255, 255);
	imshow("white out", white_out);
	imwrite("output/white_out.jpg", white_out);*/

	/*white_out1 = common_bright(white_out, com1, -25);
	Mat white_out2 = common_bright(white_out1, com2, -25);
	imshow("white out1", white_out2);*/

	img1.release();
	img2.release();
	img3.release();
	mask.release();
	H_1_to_2.release();
	//H_2_to_3.release();

	waitKey(0);
	return 0;
}

	



