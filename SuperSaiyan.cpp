#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

#define DEBUG 0

/*
The point (HAIR_X*saiyanHair.cols, HAIR_Y*saiyanHair.rows) is the coordinate
on the superSaiyan (mask) image where the center of the face should be placed.
The reason for defining this ratio rather than the x and y coordinates is
because the when the mask image is resized, these coordinates will change. 
Keep in mind that the coordinate system for images assigns the pixel in 
the top left corner as (0,0).
*/
#define HAIR_X 0.5
#define HAIR_Y 0.865

// lower value means, movement detection is more sensetive
#define MOVEMENT_THRESHOLD 2.2 

String faceCascadeName = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

Mat frame; Mat previousFrame;
vector<Rect> faces; Rect face;
Mat temp; 

String maskImg = "saiyan.png";
String windowName = "Super Saiyan Filter";

void drawSaiyan(Mat &);
bool detectFaces(Mat &);
bool detectMovement(Mat &, Mat &);
float getAverageIntensity(Mat &);


int main(int argc, char** argv) {
	
	namedWindow(windowName);
	if (!face_cascade.load(faceCascadeName)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(left_eye_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	
	// If the file name of the image is provided as the second argument then draw on that frame
	if (argc == 2) {

		frame = imread(argv[1]);

		if (detectFaces(frame)) {
			drawSaiyan(frame);
		}else {
			cout << "Valid face not found " << endl;
		}
		
		imshow(windowName, frame);
		char key = waitKey(30);
		if (key == 27) {
			cout << "Esc key pressed, exiting." << endl;
		}else if (key == 's') {
			cout << "Image saved." << endl;
			imwrite("saiyanMe.bmp", frame);

		}
		return 0;

	}else {

		// Get the frame from the webcam, previousFrame is to be compared to the currentFrame
		VideoCapture cap(0); 
		cap.read(previousFrame); 

		/* 
		Create the an image that's all black, to mask will be super imposed on this 
		This template will then be super imposed on the original frame
		*/
		temp = Mat(previousFrame.rows, previousFrame.cols, CV_8UC3);
		temp = Scalar(0, 0, 0);

		while (true) {

			cap.read(frame); 

			/* 
			If a valid face has been detected then we check if there has been significant
			movement. If so then we recalculate the position of the mask and create a new template,
			if not, then we simply super impose the template we already have. This is just my
			crude way of optimization, having to recalculate the mask's position on every frame
			slows down the program, reducing then number of frames per second and makes the
			flickering more noticeable.
			*/
			if (detectFaces(frame)) {
				if (detectMovement(frame(face), previousFrame(face))) {
					drawSaiyan(frame);
				}else {
					frame = frame + temp; 
				}
			}
			
			imshow(windowName, frame);

			cap.read(previousFrame);
			char key = waitKey(30);
			if (key == 27){
				cout << "Esc key pressed, exiting." << endl;
				break;
			}else if (key == 's') {
				cout << "Image saved." << endl;
				imwrite("saiyanMe.bmp", frame); 
				break;
			}
		}
		return 0;
	}
}


/*
	Input: the image mat to be evaluated
	Output: whether or not a face was found
	The general process before applying any feature detection is to convert to image to grey scale
	and then adjusting the intensity of the pixels such that they are more evenly distributed.
	Fills up vector<Rect> faces, but only the first element is evaluated. Sometimes the rect objects 
	returned by detectMultiScale are erroneous so just check whether or not the rectangle can
	fit in the original image
*/
bool detectFaces(Mat & frame) {
	Mat frameGray;
	cvtColor(frame, frameGray, COLOR_BGR2GRAY);
	equalizeHist(frameGray, frameGray);
	face_cascade.detectMultiScale(frameGray, faces, 1.1, 4, 0, Size(100, 100));
	
	if (faces.size() != 0) {
		face = faces[0];
		if (face.x < frame.cols && face.y < frame.rows
			&& face.x > 0 && face.y > 0) {
			return true;
		}
	}
	return false;
}

/*
	Input: the image mat to be modified
	resizes the saiyanHair image, trims it and adds it onto the src
*/
void drawSaiyan(Mat & m) {

	// This must image file must be read everytime because it gets resized
	Mat saiyanHair = imread(maskImg);

	// the temp mat is global, so it must be reset to all black
	temp = Scalar(0, 0, 0);

	// Calculate the sizing scalethe saiyanHair image should be adjusted to based on the face size
	double factor = 130;
	int constant = 20;
	double scale = (faces[0].width - constant) / factor;

	if (scale < 0) {
		return;
	}else if (scale < 1) {
		resize(saiyanHair, saiyanHair, Size(), scale, scale, INTER_AREA);
	}else if (scale >= 1) {	
		Size size(scale*saiyanHair.cols, scale*saiyanHair.rows);
		resize(saiyanHair, saiyanHair, size, 0, 0, INTER_LINEAR);

	}
	 
	// Coordinates of the center of the face wrt to the frame 
	Point faceCenter(faces[0].x + faces[0].width / 2, faces[0].y + faces[0].height / 2);

	int rightTrim = 0; int leftTrim = 0;
	int topTrim = 0; int botTrim = 0;

	// Initiate the coordinate point wrt the frame (top left corner) where the final mask is to be placed
	Point placement(faceCenter.x - HAIR_X*saiyanHair.cols, faceCenter.y - HAIR_Y*saiyanHair.rows);

	// Initiate the rectangle defining the region of interest on the resized mask
	Rect roi(0, 0, saiyanHair.cols, saiyanHair.rows);

	// If positive then trim the right side of the saiyanHair
	rightTrim = faceCenter.x + HAIR_X*saiyanHair.cols - temp.cols; 
	if (rightTrim < 0) {
		rightTrim = 0;
	}else{
		placement.x = faceCenter.x - HAIR_X*saiyanHair.cols;
		roi.width -= rightTrim;
	}

	// If positive then trim the left side
	leftTrim = - (faceCenter.x - HAIR_X*saiyanHair.cols);
	if (leftTrim < 0) {
		leftTrim = 0;
	}else{
		placement.x = 0;
		roi.width -= leftTrim;
		roi.x = leftTrim;
	}

	// If positive then trim the top side of the saiyanHair 
	topTrim = -(faceCenter.y - HAIR_Y*saiyanHair.rows);
	if (topTrim < 0) {
		topTrim = 0;
	}else{
		placement.y = 0;
		roi.height -= topTrim;
		roi.y = topTrim;
	}

	// If positive then trim the bot 
	botTrim = (faceCenter.y + (saiyanHair.rows - HAIR_Y*saiyanHair.rows) - temp.rows);
	if (botTrim < 0) {
		botTrim = 0;
	}else{
		roi.height -= topTrim;
	}

	if (DEBUG) {
		cout << "m cols" << m.cols << endl;
		cout << "m rows " << m.rows << endl;
		cout << "face x " << faceCenter.x << endl;
		cout << "face y " << faceCenter.y << endl;
		cout << "left Trim " << leftTrim << endl;
		cout << "right Trim " << rightTrim << endl;
		cout << "top Trim " << topTrim << endl;
		cout << "bot Trim " << botTrim << endl;
		cout << "roi heigh " << roi.height << endl;
		cout << "roi width " << roi.width << endl;
		cout << "roi x " << roi.x << endl;
		cout << "roi y " << roi.y << endl;
		cout << "placement x " << placement.x << endl;
		cout << "placement y " << placement.y << endl;
		cout << "saiyanHair.rows (resized) " << saiyanHair.rows << endl;
		cout << "saiyanHair.cols (resized)" << saiyanHair.cols << endl;
	}

	// There was some rounding with the conversion of doubles to ints so just to be safe
	roi.height -= 10;
	roi.width -= 10;

	// Resized and trimmed version of saiyanHair
	Mat	trimmedSaiyanHair = saiyanHair(roi);

	// Rect wrt the frame defining where the trimmedSaiyanHair is to be placed
	Rect hairArea(placement, roi.size());

	// Copy the trimmedSaiyanHair onto the plain black template
	trimmedSaiyanHair.copyTo(temp(hairArea)); 

	// Add the template onto the original
	m = m + temp;
}


/*
	Input: Two frames to be evaluated, obvsiously must be the same size.
	Output: Return true if movement is detected, otherwise, false.
*/
bool detectMovement(Mat & frame, Mat & previousFrame) {
	
	Mat res = frame - previousFrame;
	cvtColor(res, res, COLOR_BGR2GRAY);

	float averageInt = getAverageIntensity(res);
	cout << averageInt << endl;
	if (averageInt > MOVEMENT_THRESHOLD) {
		return true;
	}else{
		return false;
	}
}

/*
	Input: Image mat to be evaluated, must be grey scale
	Output: Average intensity value of all pixels in the image
*/
float getAverageIntensity(Mat &m) {
	
	int count = 0;
	float average = 0;
	float buffer = 0;
	const int bufferSize = 10;
	MatIterator_<uchar> it, end;

	for (it = m.begin<uchar>(), end = m.end<uchar>(); it != end; it++) {
		int value = *it;
		
		if (count == bufferSize) {
			average += buffer / bufferSize;
			buffer = 0;
		}

		buffer += value;
		count++;
	}
	
	return average;
}