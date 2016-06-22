#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

/*
TODO
Extensibility: add different levels of saiyan.. improve the saiyan.png, provide a trackbar
Add eye highlights, 
Add different filters, also a track bar system...

*/


// the point (HAIR_X*saiyanHair.cols, HAIR_Y*saiyanHair.rows) is the coordinate on the saiyanHair image
// where the center of the face should be placed
#define HAIR_X 0.5
#define HAIR_Y 0.865

#define MOVEMENT_THRESHOLD 2.2 // lower value means, movement detection is more sensetive

String faceCascadeName = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

Mat frame; Mat previousFrame;
vector<Rect> faces; Rect face;
Mat temp; // plain black image

String windowName = "Super Saiyan Filter";

void drawSaiyan(Mat &);
bool detectFaces(Mat &);
bool detectMovement(Mat &, Mat &);
float getAverageIntensity(Mat &);


int main(int argc, char** argv) {
	
	namedWindow(windowName);
	if (!face_cascade.load(faceCascadeName)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	
	if (argc == 2) {

		frame = imread(argv[1]);

		if (detectFaces(frame)) {
			drawSaiyan(frame);
		}else {
			cout << "Valid face not found " << endl;
		}
		
		imshow(windowName, frame);
		waitKey(0); // this is the only reason why it doesnt crash immediately
		return 0;

	}else {

		VideoCapture cap(0); //capture the video from webcam
		cap.read(previousFrame); // read a new frame from video

		temp = Mat(previousFrame.rows, previousFrame.cols, CV_8UC3);
		temp = Scalar(0, 0, 0);

		while (true) {

			cap.read(frame); // read a new frame from video

			if (detectFaces(frame)) {
				if (detectMovement(frame(face), previousFrame(face))) {
					drawSaiyan(frame);
				}else {
					// we dont want to recalculate the new position of the face
					// the placement of the hair is preserved from the last frame
					frame = frame + temp; 
				}
			}
			
			imshow(windowName, frame);

			cap.read(previousFrame);
			if (waitKey(30) == 27){
				cout << "Esc key is pressed by user" << endl;
				break;
			}
		}
		return 0;
	}
}


/*
	Input: the Mat to be evaluated
	Fills up vector<Rect> faces 
	Check for and return validation of faces[0]
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
	Input: the src mat to be modified
	resizes the saiyanHair image, trims it and adds it onto the src
*/
void drawSaiyan(Mat & m) {

	// this must be read every time, or else global shit fucks up
	Mat saiyanHair = imread("saiyan.png");
	// this is global too, so reset is required
	temp = Scalar(0, 0, 0);

	// define the relationship between size of face and size of saiyanHair image
	double factor = 130;
	int constant = 20;
	double scale = (faces[0].width - constant) / factor;

	cout << "scale " << scale << endl;
	if (scale < 0) {
		return;
	}else if (scale < 1) {
		resize(saiyanHair, saiyanHair, Size(), scale, scale, INTER_AREA);
	}else if (scale >= 1) {	
		Size size(scale*saiyanHair.cols, scale*saiyanHair.rows);
		resize(saiyanHair, saiyanHair, size, 0, 0, INTER_LINEAR);

	}
	cout << "scale : " << scale << endl;
	// since the src image isnt actually a square, it's not resizing as expected
	// the axis arent actually multipled by some factor, doesnt work that way
	 

	// coordinates of the face's center wrt to src coordinates
	Point faceCenter(faces[0].x + faces[0].width / 2, faces[0].y + faces[0].height / 2);

	int rightTrim = 0; int leftTrim = 0;
	int topTrim = 0; int botTrim = 0;

	// top left corner where the trimmedSaiyanHair is to be placed
	Point placement(faceCenter.x - HAIR_X*saiyanHair.cols, faceCenter.y - HAIR_Y*saiyanHair.rows);

	// defines the area trimmedSaiyanHair is obtained from saiyanHair
	Rect roi(0, 0, saiyanHair.cols, saiyanHair.rows);
	cout << "roi width on init " << roi.width << endl;
	cout << "roi height on init " << roi.height << endl;

	// if positive, then trim the right side of the saiyanHair
	rightTrim = faceCenter.x + HAIR_X*saiyanHair.cols - temp.cols; 
	if (rightTrim < 0) {
		rightTrim = 0;
	}else{
		placement.x = faceCenter.x - HAIR_X*saiyanHair.cols;
		roi.width -= rightTrim;
	}

	// if positive then trim the left side
	leftTrim = - (faceCenter.x - HAIR_X*saiyanHair.cols);
	if (leftTrim < 0) {
		leftTrim = 0;
	}else{
		placement.x = 0;
		roi.width -= leftTrim;
		roi.x = leftTrim;
	}

	// if positive, then trim the top side of the saiyanHair 
	topTrim = -(faceCenter.y - HAIR_Y*saiyanHair.rows);
	if (topTrim < 0) {
		topTrim = 0;
	}else{
		placement.y = 0;
		roi.height -= topTrim;
		roi.y = topTrim;
	}

	// if positive then trim the bot 
	botTrim = (faceCenter.y + (saiyanHair.rows - HAIR_Y*saiyanHair.rows) - temp.rows);
	if (botTrim < 0) {
		botTrim = 0;
	}else{
		roi.height -= topTrim;
	}


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

	// there some rounding going on and it was off by 1
	roi.height -= 10;
	roi.width -= 10;

	// resized and clipped version of saiyanHair
	Mat	trimmedSaiyanHair = saiyanHair(roi);
	cout << " roi selection success" << endl;

	// rect wrt src coordinates defining where the trimmedSaiyanHair is to be placed
	Rect hairArea(placement, roi.size());
	cout << " rect created " << endl;

	// copies the trimmedSaiyanHair onto the plain black image
	trimmedSaiyanHair.copyTo(temp(hairArea)); 
	cout << " copy to success " << endl;

	m = m + temp;

}


/*
	Makes NO modifications to the inputs
	Both inputs better be the same size
	Return true if there's movement
	Otherwise the scene's still so return false
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