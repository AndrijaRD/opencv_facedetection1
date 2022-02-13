#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
	string path = ("TVOJ PUT DO FOLDERA ASSETSS");
	
	double scale = 3.0;
	double neighbors = 2.0;

	Mat frame;
	Mat grayscale;

	bool found = false;

	vector<Rect> profile_loaded_face;
	vector<Rect> frontal_loaded_face;

	CascadeClassifier Cascade_profile;
	CascadeClassifier Cascade_frontal;
	Cascade_profile.load(path + "haarcascade_profileface.xml");
	Cascade_frontal.load(path + "haarcascade_frontalface_alt.xml");

	VideoCapture cap(2);

	while (true) {
		cap.read(frame);

		found = false;

		cvtColor(frame, grayscale, COLOR_BGR2GRAY);
		resize(grayscale, grayscale, Size(grayscale.size().width / scale, grayscale.size().height / scale));
		
		Cascade_profile.detectMultiScale(grayscale, profile_loaded_face, 1.1, neighbors, 0, Size(30, 30));
		Cascade_frontal.detectMultiScale(grayscale, frontal_loaded_face, 1.1, neighbors, 0, Size(30, 30));
		
		if (found == false) {
			for (Rect area : profile_loaded_face) {
				Scalar drawColor = Scalar(0, 255, 0);
				rectangle(frame, Point(cvRound(area.x * scale), cvRound(area.y * scale)),
					Point(cvRound((area.x + area.width - 1) * scale), cvRound((area.y + area.height - 1) * scale)),
					drawColor, 3);
				found = true;
			}
		}
		if (found == false) {
			for (Rect area : frontal_loaded_face) {
				Scalar drawColor = Scalar(255, 0, 0);
				rectangle(frame, Point(cvRound(area.x * scale), cvRound(area.y * scale)),
					Point(cvRound((area.x + area.width - 1) * scale), cvRound((area.y + area.height - 1) * scale)),
					drawColor, 3);
			}
		}

		imshow("Frame", frame);
		if (waitKey(1) == 'q') { break; }
	}
	return 0;
}
