#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace cv;
using namespace std;

const char* params
= "{ help h         |           | Print usage }"
"{ input          | vtest.avi | Path to a video or a sequence of image }"
"{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";

int main(int argc, char* argv[])
{
    //setting up background subtraction
    CommandLineParser parser(argc, argv, params);
    Ptr<BackgroundSubtractor> pBackSub;
    if (parser.get<String>("algo") == "MOG2")
        //setting history (number of frames analyzed to create backround) (currently set to 500) and threshold (currently set to 200)
        //HISTORY AND THRESHOLD CAN BE ADJUSTED TO IMPROVE DETECTION ACCURACY
        pBackSub = createBackgroundSubtractorMOG2(500, 200, false);
    else
        pBackSub = createBackgroundSubtractorKNN();

    Mat frame, fgMask, thresh;
    vector<vector<Point> > cnts;

    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    cap.open(0, cv::CAP_DSHOW);                 //use this to capture frames from webcam (0 for default, 1 for first usb cam, 2 for second etc)
    //cap.open("bug footage cut down.MP4");     //use to capture frames from file

    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    //SETTING UP CAMERA SETTINGS
    //if capturing from file, comment these out
    int fps = 15;
    int brightness = 127;
    int whiteBalance = 255;
    int focus = 20;         //full pan
    //int focus = 25;         //half pan
    int sharpness = 0;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(5, fps);
    cap.set(10, brightness);
    cap.set(17, whiteBalance);
    cap.set(20, sharpness);
    cap.set(28, focus);

    //setting up variables for resolution
    int frame_width = int(cap.get(3));
    int frame_height = int(cap.get(4));
    
    //setting up video capture object
    //VideoWriter video("test recording.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 10.0, Size(frame_width, frame_height));

    //setting up for measuring time
    time_t start = time(0);
    double difTime;

    bool isFirstFrame = true;
    bool motionDetected = false;

    for (;;)
    {
        //capturing frame
        cap.read(frame);

        //check if frame read successfully
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        //writing frame into created video object, saves what the camera sees
        //video.write(frame);

        //applying background subtraction and threshold (currently set to 25) to binarize the output
        //THRESHOLD CAN BE ADJUSTED TO IMPROVE DETECTION ACCURACY
        pBackSub->apply(frame, fgMask);
        threshold(fgMask, thresh, 25, 255, THRESH_BINARY);

        //ceparates each detected clump of moving pixels and processes it as a separate entity
        dilate(thresh, thresh, Mat(), Point(-1, -1), 2);
        findContours(thresh, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<vector<Point> > cnts_poly(cnts.size());
        vector<Rect> boundRect(cnts.size());
        if (isFirstFrame) isFirstFrame = false;             //skips first frame, since first frame will always display motion (background subtraction does that)    
        else {
            for (int i = 0; i < cnts.size(); i++) {
                if (contourArea(cnts[i]) < 10) {            //creates a rectangular outlinearound objects larger than a set size (currently set to 10)
                                                            //CAN BE ADJUSTED TO IMPROVE DETECTION ACCURACY
                    continue;
                }
                approxPolyDP(Mat(cnts[i]), cnts_poly[i], 3, true);
                boundRect[i] = boundingRect(Mat(cnts_poly[i]));
                drawContours(fgMask, cnts, i, Scalar(255, 255, 255), -3);
                rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8, 0);

                //does stuff when motion gets detected
                //conditions can be added, like a certain number of frames with motion being necessary before program reacts
                //CONDITIONS CAN BE ADJUSTED TO IMPROVE DETECTION ACCURACY
                putText(frame, "Motion Detected", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
                motionDetected = true;
            }

            //writing into created video object, saves what camera sees with highlighted moving objects
            //video.write(frame);
        }


        //displays frame with highlithed motion and what alogithms sees
        namedWindow("frame", WINDOW_NORMAL);
        namedWindow("bgsubstr", WINDOW_NORMAL);
        imshow("frame", frame);
        imshow("bgsubstr", fgMask);

        //finishes the session either manually by pressing any button or after a set time 
        difTime = time(0) - start;
        if (waitKey(5) >= 0)
        //if (waitKey(5) >= 0 || difTime > 15)
            break;
    }

    //displays if any motion was detected in the session
    if (motionDetected) cout << "      MOTION DETECTED";
    else cout << "        no motion";

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}