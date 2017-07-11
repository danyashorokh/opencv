#include "stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <cv.h>
#include <cxcore.h>
#include <iostream>
#include <stdio.h>
#include <highgui.h>
#include <stdlib.h>
//#include <vector>

using namespace std;
using namespace cv;

int Hmin = 0;
int Hmax = 256;
int Smin = 0;
int Smax = 256;
int Vmin = 0;
int Vmax = 256;

int Hmin1 = 0;
int Hmax1 = 256;
int Smin1 = 0;
int Smax1 = 256;
int Vmin1 = 0;
int Vmax1 = 256;

int Hmin2 = 0;
int Hmax2 = 256;
int Smin2 = 0;
int Smax2 = 256;
int Vmin2 = 0;
int Vmax2 = 256;

int HSVmax = 256;

int minsize = 50; //0
int maxsize = 50;
int maxs = 50;


int rectw = 35;
int recth = 45;

// Для морфологических преобразований
int dilation_elem = 0;
int kernel_size1 = 0;  // dilate 1
int kernel_size2 = 0;  // erode 1
int kernel_size3 = 0;  // close 1
int kernel_size4 = 0;  // dilate 1
int kernel_size5 = 0;  // erode 1
int kernel_size6 = 0;  // close 1
int const max_elem = 1;
int const max_kernel_size = 11;

// Для оператора Кэнни
Mat detected_edges;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

int counter = 0; //счетчик для рисования зоны roi


Mat input_image, team1, team2, hsv1, hsv2, team1_hsv, team2_hsv, team1_morph, team2_morph;
Mat input_hsv, pitch_hsv, pitch_canny, pitch_rgb, pitch_hough, pitch_morph, ball;
Mat two_teams;
Mat cont;
Mat element1, element2, element3, element4, element5, element6;

// ROI
IplImage* roi = 0;
Point roi_1,roi_2;

IplImage* addplayer = 0;

//
// функции-обработчики ползунков
//
//void myTrackbarHmin(int pos) {
//        Hmin = pos;
//}

// рисуем целеуказатель
void drawTarget(/*Mat input*/IplImage* input, int x, int y, int counter) {
	//Point p3,p4;
	if((counter!=0)&&(counter%2==1)) {
		roi_1.x = x;
		roi_1.y = y;
		
	}

	if((counter!=0)&&(counter%2==0)) {
		roi_2.x = x;
		roi_2.y = y;
		//rectangle(input, p3,p4, Scalar(0, 255, 0, 4), 2);
		cvRectangle(input, roi_1,roi_2, Scalar(0, 255, 0, 4), 2);
	}

        
}

// обработчик событий от мышки
void myMouseCallback( int event, int x, int y, int flags, void* param )
{
	    Point center;
		center.x = x;
		center.y = y;

		IplImage* input = (IplImage*) param;

        switch( event ){
                case CV_EVENT_MOUSEMOVE: 
                        break;

                case CV_EVENT_LBUTTONDOWN:
                        printf("%d x %d\n", x, y);
						counter++;
                        drawTarget(input, x, y, counter);
                        break;

                case CV_EVENT_LBUTTONUP:
                        break;

				case CV_EVENT_RBUTTONDOWN:
                        printf("%d x %d\n", x, y);
						circle( team1, center, 7, Scalar(0,255,255), 2, 8, 0 );
						circle( team2, center, 7, Scalar(0,255,255), 2, 8, 0 );
						circle( two_teams, center, 7, Scalar(0,255,255), 2, 8, 0 );
						cvCircle( roi,center, 7, Scalar(0,255,255), 2, 8, 0 );
                        break;

                case CV_EVENT_RBUTTONUP:
                        break;
        }
}

void addPlayer( int event, int x, int y, int flags, void* param )
{
	    Point p3,p4;
		p3.x = x - rectw/2;
		p3.y = y - recth/4;
		p4.x = x + rectw/2;
		p4.y = y + recth;

		IplImage* input = (IplImage*) param;

        switch( event ){
                case CV_EVENT_MOUSEMOVE: 
                        break;

                case CV_EVENT_LBUTTONDOWN:
                        rectangle(team1, p3,p4, Scalar(255, 0, 0, 4), 2);
						rectangle(two_teams, p3,p4, Scalar(255, 0, 0, 4), 2);
                        break;

                case CV_EVENT_LBUTTONUP:
                        break;

				case CV_EVENT_RBUTTONDOWN:
                        rectangle(team2, p3,p4, Scalar(0, 0, 255, 4), 2);
						rectangle(two_teams, p3,p4, Scalar(0, 0, 255, 4), 2);
                        break;

                case CV_EVENT_RBUTTONUP:
                        break;
        }
}

void detectTeam(Mat input, Mat output, int b, int g, int r){

	for (int y = roi_1.y; y < roi_2.y; y++) { //for (int y = 0; y < input.rows; y++)
						for (int x = roi_1.x; x < roi_2.x; x++) {  //for (int x = 0; x < input.cols; x++)
							int value = input.at<uchar>(y, x);
							if (value == 255) {
								Rect rect;
								Point p1,p2;
								p1.x = x - rectw/2;
								p1.y = y - recth/4;
								p2.x = x + rectw/2;
								p2.y = y + recth;
								int count = floodFill(input, Point(x, y), Scalar(200), &rect);
								if((rect.width >= minsize && rect.width <= maxsize)
									|| (rect.height >= minsize && rect.height <= maxsize)){
									//rectangle(output, p1,p2, Scalar(b, g, r, 4), 2);
									rectangle(output, rect, Scalar(b, g, r, 4), 2);
								}
							}
						}
					}
}

void detectBall(Mat input, Mat output) {
	
	Mat gray;
	/// Convert it to gray
  cvtColor( input, gray, CV_BGR2GRAY );
  imshow("gray", gray);

 // for (int y = roi_1.y; y < roi_2.y; y++) { //for (int y = 0; y < input.rows; y++)
	//for (int x = roi_1.x; x < roi_2.x; x++) {  //for (int x = 0; x < input.cols; x++)
	//						int value = input.at<uchar>(y, x);
	//						if (value == 0) {
	//							Rect rect;
	//							Point center;
	//							center.x = x;
	//							center.y = y;
	//							int count = floodFill(input, Point(x, y), Scalar(200), &rect);
	//							if((rect.width >= minsize/3 && rect.width <= maxsize/3)
	//								|| (rect.height >= minsize/3 && rect.height <= maxsize/3)){
	//								circle( output, center, 7, Scalar(0,255,255), 2, 8, 0 );
	//							}
	//						}
	//	}
	//}


  //////Reduce the noise so we avoid false circle detection
  //GaussianBlur( gray, gray, Size(9, 9), 2, 2 );
  vector<Vec3f> circles;
  /// Apply the Hough Transform to find the circles
  HoughCircles( input, circles, CV_HOUGH_GRADIENT, 2, gray.rows/16, 40, 100, 1, 7 ); //200 100 1 10
  // Draw the circles detected
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
	  circle( output, center, radius, Scalar(0,255,255), 2, 8, 0 );
	  //printf("centerx = %d, centery = %d, radius = %d\n", center.x, center.y, radius);
   }
}

void dilation1( int, void* ){
  int dilation_type = MORPH_ELLIPSE;
  //if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  //else if( dilation_elem == 1 ){ dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*kernel_size1 + 1, 2*kernel_size1+1 ),
                                       Point( kernel_size1, kernel_size1 ) );
  //dilate(team1_hsv, team1_morph, element);
  //imshow("team1", team1_morph);
  
}

void hsv_test1() {
	// экспер
		Hmin1 = 54;//74;//72;//54;
		Hmax1 = 139;//124;//191;//139;
		Smin1 = 60;//44;//44;//60;
		Smax1 = 207;//251;//247;//207;
		Vmin1 = 74;//63;//63;//74;
		Vmax1 = 208;//240;//208;//208;
		//inRange(hsv1, Scalar(Hmin1, Smin1, Vmin1), Scalar(Hmax1, Smax1, Vmax1), team1_hsv);
}

void hsv_test2() {
	// экспер
		Hmin2 = 0;//0;//33;
		Hmax2 = 170;//139;//142;
		Smin2 = 0;//0;//0;
		Smax2 = 16;//23;//35;
		Vmin2 = 174;//153;//153;
		Vmax2 = 210;//236;//256;
		//inRange(hsv2, Scalar(Hmin2, Smin2, Vmin2), Scalar(Hmax2, Smax2, Vmax2), team2_hsv);
}

void morph_test() {
	kernel_size1 = 1;
	kernel_size3 = 5;
	kernel_size4 = 1;
	kernel_size6 = 5;
}


void trackbar() {
		namedWindow("input_hsv", 0); 
		
		cvCreateTrackbar("Hmin", "input_hsv", &Hmin, HSVmax);
        cvCreateTrackbar("Hmax", "input_hsv", &Hmax, HSVmax);
        cvCreateTrackbar("Smin", "input_hsv", &Smin, HSVmax);
        cvCreateTrackbar("Smax", "input_hsv", &Smax, HSVmax);
        cvCreateTrackbar("Vmin", "input_hsv", &Vmin, HSVmax);
        cvCreateTrackbar("Vmax", "input_hsv", &Vmax, HSVmax);
}

void trackbar1() {
		namedWindow("Control1", 0); 
		
		cvCreateTrackbar("Hmin", "Control1", &Hmin1, HSVmax);
        cvCreateTrackbar("Hmax", "Control1", &Hmax1, HSVmax);
        cvCreateTrackbar("Smin", "Control1", &Smin1, HSVmax);
        cvCreateTrackbar("Smax", "Control1", &Smax1, HSVmax);
        cvCreateTrackbar("Vmin", "Control1", &Vmin1, HSVmax);
        cvCreateTrackbar("Vmax", "Control1", &Vmax1, HSVmax);
		cvCreateTrackbar( "Dilate:\n 2n + 1", "Control1",
                  &kernel_size1, max_kernel_size);
		cvCreateTrackbar( "Erode:\n 2n + 1", "Control1",
                  &kernel_size2, max_kernel_size);
		cvCreateTrackbar( "Close:\n 2n + 1", "Control1",
                  &kernel_size3, max_kernel_size);
}

void trackbar2() {
		namedWindow("Control2", 0); 
		
		cvCreateTrackbar("Hmin", "Control2", &Hmin2, HSVmax);
        cvCreateTrackbar("Hmax", "Control2", &Hmax2, HSVmax);
        cvCreateTrackbar("Smin", "Control2", &Smin2, HSVmax);
        cvCreateTrackbar("Smax", "Control2", &Smax2, HSVmax);
        cvCreateTrackbar("Vmin", "Control2", &Vmin2, HSVmax);
        cvCreateTrackbar("Vmax", "Control2", &Vmax2, HSVmax);
		cvCreateTrackbar( "Dilate:\n 2n + 1", "Control2",
                  &kernel_size4, max_kernel_size);
		cvCreateTrackbar( "Erode:\n 2n + 1", "Control2",
                  &kernel_size5, max_kernel_size);
		cvCreateTrackbar( "Close:\n 2n + 1", "Control2",
                  &kernel_size6, max_kernel_size);
}

void Hough1() {
	/*vector<Vec2f> lines;
		  HoughLines(pitch_canny, lines, 1, CV_PI/180, 100, 0, 0 );
		
		  for( size_t i = 0; i < lines.size(); i++ )
		  {
			 float rho = lines[i][0], theta = lines[i][1];
			 Point pt1, pt2;
			 double a = cos(theta), b = sin(theta);
			 double x0 = a*rho, y0 = b*rho;
			 pt1.x = cvRound(x0 + 1000*(-b));
			 pt1.y = cvRound(y0 + 1000*(a));
			 pt2.x = cvRound(x0 - 1000*(-b));
			 pt2.y = cvRound(y0 - 1000*(a));
			 line( pitch_hough, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
		  }*/
		
		vector<Vec4i> lines;
		HoughLinesP(pitch_canny, lines, 1, CV_PI/180, 50, 50, 70 ); //20
		for( size_t i = 0; i < lines.size(); i++ )
		{
		Vec4i l = lines[i];
		line( pitch_hough, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, CV_AA);
		}
}

void findCont() {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	/// Find contours
	findContours( pitch_canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	cont = Mat::zeros( input_image.size(), CV_8UC3 );
  /// Draw contours
  for( int i = 0; i< contours.size(); i++ )
     {
       drawContours( cont, contours, i, Scalar(0,0,255), 2, 8, hierarchy, 0, Point() );
     }
}

void CannyThreshold(int, void*) {
  /// Reduce noise with a kernel 3x3
  blur( pitch_hsv, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, pitch_canny, lowThreshold, lowThreshold*ratio, kernel_size ); //detected_edges 2x

  /// Using Canny's output as a mask, we display our result
  //pitch = Scalar::all(0);

  //input_image.copyTo( pitch, detected_edges);

  Hough1();
  
 }

void test_canny() {
	/// Reduce noise with a kernel 3x3
  blur( pitch_hsv, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, pitch_canny, 80, lowThreshold*ratio, kernel_size ); //detected_edges 2x
}



int main(int argc, char* argv[])
{
        // имя картинки
        char* filename = "rain1.jpg";//"test4new.jpeg";
        // получаем картинку
		input_image = imread(filename);
		team1 = imread(filename);
		team2 = imread(filename);
		two_teams = imread(filename);
		pitch_hough = imread(filename);
		ball = imread(filename);

		roi = cvLoadImage(filename,1);
		addplayer = cvLoadImage(filename,1);

		if(!input_image.data) return -1;
		if(!team1.data) return -1;
		if(!team2.data) return -1;
		if(!pitch_hough.data) return -1;
		
        printf("[i] image: %s\n", filename);

		// ползунки
		//trackbar();
		trackbar1();
		trackbar2();
        
		//namedWindow("Morph", 0);
		cvNamedWindow("size",CV_WINDOW_AUTOSIZE);
		//cvNamedWindow("Morph1",CV_WINDOW_AUTOSIZE);
		//cvNamedWindow("Morph2",CV_WINDOW_AUTOSIZE);
		cvCreateTrackbar("Min size", "size", &minsize, maxs);
        cvCreateTrackbar("Max size", "size", &maxsize, maxs);
		//createTrackbar( "Element:\n 0: Rect \n 1: Ellipse", "Control1", &dilation_elem, max_elem, dilation1 );
		
		//---------Canny---------------------------------------------------------

		/// Create a matrix of the same type and size as src (for dst)
		pitch_hsv.create( input_image.size(), input_image.type() );
		pitch_canny.create( input_image.size(), input_image.type() );
		ball.create( input_image.size(), input_image.type() );

		cvtColor( input_image, input_hsv, CV_BGR2HSV);
		int Hmin = 34; 
		int Hmax = 66;
		int Smin = 107;
		int Smax = 256;
		int Vmin = 103;
		int Vmax = 189;
		inRange(input_hsv, Scalar(Hmin, Smin, Vmin), Scalar(Hmax, Smax, Vmax), pitch_hsv);

		namedWindow( "Canny", CV_WINDOW_AUTOSIZE );
		
		
		// окно для отображения ROI
        cvNamedWindow("roi",CV_WINDOW_AUTOSIZE);
		cvNamedWindow("add",CV_WINDOW_AUTOSIZE);

		createTrackbar( "Min Threshold:", "Canny", &lowThreshold, max_lowThreshold, CannyThreshold );

		//--------------------------------------------------------------

		//---------------------Поиск контуров---------------------------
		test_canny();
		findCont();

		// вешаем обработчик мышки
        cvSetMouseCallback( "roi", myMouseCallback, (void*) roi);
		cvSetMouseCallback( "add", addPlayer, (void*) roi);
	
		
		//--------------------------------------------------------------

		hsv_test1();
		hsv_test2();
		morph_test();
		

		cvtColor(input_image, hsv1, COLOR_BGR2HSV);
		cvtColor(input_image, hsv2, COLOR_BGR2HSV);
		//medianBlur(HSV, blurred, 21);
		//cvGetImage(hsv_and,thresh);

        while(true) {

				inRange(hsv1, Scalar(Hmin1, Smin1, Vmin1), Scalar(Hmax1, Smax1, Vmax1), team1_hsv);
				inRange(hsv2, Scalar(Hmin2, Smin2, Vmin2), Scalar(Hmax2, Smax2, Vmax2), team2_hsv);
				//inRange(input_hsv, Scalar(Hmin, Smin, Vmin), Scalar(Hmax, Smax, Vmax), pitch_hsv);
				
				Mat element1 = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 2*kernel_size1 + 1, 2*kernel_size1+1 ),
                                       Point( kernel_size1, kernel_size1 ) );
				Mat element2 = getStructuringElement( MORPH_ELLIPSE,
												   Size( 2*kernel_size2 + 1, 2*kernel_size2+1 ),
												   Point( kernel_size2, kernel_size2 ) );
				Mat element3 = getStructuringElement( MORPH_ELLIPSE,
												   Size( 2*kernel_size3 + 1, 2*kernel_size3+1 ),
												   Point( kernel_size3, kernel_size3 ) );
				Mat element4 = getStructuringElement( MORPH_ELLIPSE,
												   Size( 2*kernel_size4 + 1, 2*kernel_size4+1 ),
												   Point( kernel_size4, kernel_size4 ) );
				Mat element5 = getStructuringElement( MORPH_ELLIPSE,
												   Size( 2*kernel_size5 + 1, 2*kernel_size5+1 ),
												   Point( kernel_size5, kernel_size5 ) );
				Mat element6 = getStructuringElement( MORPH_ELLIPSE,
												   Size( 2*kernel_size6 + 1, 2*kernel_size6+1 ),
												   Point( kernel_size6, kernel_size6 ) );
				
                // выполняем преобразования
				dilate(team1_hsv, team1_hsv, element1); //team1_morph
                dilate(team2_hsv, team2_hsv, element4);
				erode(team1_hsv, team1_hsv, element2);
				erode(team2_hsv, team2_hsv, element5);
				morphologyEx(team1_hsv, team1_hsv, MORPH_CLOSE, element3);
				morphologyEx(team2_hsv, team2_hsv, MORPH_CLOSE, element6);
				//--------------
					
				//detectTeam(team1_morph, team1, 255, 0, 0);
				//detectTeam(team2_morph, team2, 0, 0, 255);

				detectTeam(team1_hsv, two_teams, 255, 0, 0);
				detectTeam(team2_hsv, two_teams, 0, 0, 255);
				//detectBall(input_image, ball);

				//imshow("input", input_image);
				imshow("team1_hsv", team1_hsv);
				imshow("team2_hsv", team2_hsv);
				//imshow("morph1", team1_morph);
				//imshow("morph2", team2_morph);
				//imshow("pitch_morph", pitch_morph);
				//imshow("team1", team1);
			    //imshow("team2", team2);
				imshow("result", two_teams);
				//imshow("pitch_hsv", pitch_hsv);
				//imshow("Canny", pitch_canny);
				//imshow("Hough", pitch_hough);
				//imshow("Contours", cont);
				//imshow("Det", detected_edges);
				//imshow("Ball", ball);

				cvShowImage( "roi", roi );
				cvShowImage( "add", addplayer );
				
		
		
                char c = cvWaitKey(33);
                if (c == 27) { // если нажата ESC - выходим
                        break;
                }
				if (c == 32) { // если нажата ESC - выходим
                        imwrite( "result.jpg", two_teams );
                }

        }
        

        // освобождаем ресурсы
		cvReleaseData(&input_image);
		cvReleaseData(&team1_morph);
		cvReleaseData(&team2_morph);
		cvReleaseData(&team1);
		cvReleaseData(&team2);
		cvReleaseData(&pitch_canny);
		cvReleaseData(&pitch_hough);
		cvReleaseImage(&roi);
        
        // удаляем окна
        cvDestroyAllWindows();
        return 0;
}