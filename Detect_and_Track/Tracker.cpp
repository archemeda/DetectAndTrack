#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp> // contrib yüklenmeli !!!
#include <opencv2/core/ocl.hpp>
#include <opencv2/gapi/core.hpp> // GPU API library

using namespace cv;
using namespace std; 

#include "model.hpp"
#include "track_utils.hpp"

//PID Struct and functions
typedef struct {

	/* Controller gains */
	float Kp;
	float Ki;
	float Kd;

	/* Derivative low-pass filter time constant */
	float tau;

	/* Output limits */
	float limMin;
	float limMax;

	/* Integrator limits */
	float limMinInt;
	float limMaxInt;

	/* Sample time (in seconds) */
	float T;

	/* Controller "memory" */
	float integrator;
	float prevError;			/* Required for integrator */
	float differentiator;
	float prevMeasurement;		/* Required for differentiator */

	/* Controller output */
	float out;

} PID;

#define PID_KP  2.0f
#define PID_KI  0.5f
#define PID_KD  0.25f

#define PID_TAU 0.02f

#define PID_LIM_MIN   -1000.0f
#define PID_LIM_MAX   1000.0f

#define PID_LIM_MIN_T   0.0f
#define PID_LIM_MAX_T   1000.0f

#define PID_LIM_MIN_INT -5.0f
#define PID_LIM_MAX_INT  5.0f

#define SAMPLE_TIME_S 0.01f


#define frame_ratio 15 // iç boxun ROI ye oraný
#define val 4
int mode = 1; // player modes --> play - 1 : stop - 0   || tuþlar:  esc --> çýk , p --> pause , r--> return  

const char* winname = "Takip ekrani"; 
const int win_size_h = 608, win_size_w = 608; // fixed win sizes

//PID fonksiyon prototipleri
float PID_update(PID* pid, float setpoint, float measurement);
void PID_init(PID* pid);

std::string keys =
"{ help  h     | | Print help message. }"
"{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
"{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
"{ device      |  0 | camera device number. }"
"{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
"{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
"{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
"{ thr         | .5 | Confidence threshold. }"
"{ nms         | .4 | Non-maximum suppression threshold. }"
"{ backend     |  0 | Choose one of computation backends: "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation, "
"4: VKCOM, "
"5: CUDA }"
"{ target      | 0 | Choose one of target computation devices: "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"3: VPU, "
"4: Vulkan, "
"6: CUDA, "
"7: CUDA fp16 (half-float preprocess) }"
"{ async       | 0 | Number of asynchronous forwards at the same time. "
"Choose 0 for synchronous mode }";

int main(int argc, char** argv)
{

	//PID Struct creation and initialisation for manouver calculation 
	PID manouver_control = { PID_KP, PID_KI, PID_KD, PID_TAU,PID_LIM_MIN, PID_LIM_MAX,PID_LIM_MIN_INT, PID_LIM_MAX_INT,SAMPLE_TIME_S }; 
	PID_init(&steer_control);


	CommandLineParser parser(argc, argv, keys);

	const std::string modelName = parser.get<String>("@alias");
	const std::string zooFile = parser.get<String>("zoo");
	keys += genPreprocArguments(modelName, zooFile);

	parser = CommandLineParser(argc, argv, keys);

	CV_Assert(parser.has("model"));
	std::string modelPath = findFile(parser.get<String>("model"));
	std::string configPath = findFile(parser.get<String>("config"));

	model_param param = {modelName, modelPath, configPath, parser.get<String>("framework"), parser.get<int>("backend"), 
						parser.get<int>("target"), parser.get<int>("async")};
	model yolov4(param);
	yolov4.confThreshold = parser.get<float>("thr");
	yolov4.nmsThreshold = parser.get<float>("nms");
	yolov4.scale = parser.get<float>("scale");
	yolov4.swapRB = parser.get<float>("rgb");
	yolov4.mean = parser.get<float>("mean");
	yolov4.inpHeigth = parser.get<int>("height");
	yolov4.inpWidth = parser.get<int>("width");
	if (parser.has("classes"))
		yolov4.get_classes(parser.get<string>("classes"));
	
	string filename;
	if(parser.has("input"))
		filename = parser.get<String>("input");
	Ptr<Tracker>tracker = TrackerMOSSE::create();//Tracker declaration

	VideoCapture video;
	if (!filename.empty())
	{
		video.open(filename);
		video.set(CAP_PROP_FRAME_WIDTH, win_size_w); // resize the screen
		video.set(CAP_PROP_FRAME_HEIGHT, win_size_h);
		cout << "file founded!!!" << endl;
	}
	else
		video.open(0);
	// Exit if video is not opened
	if (!video.isOpened())
	{
		cout << "Could not read video file" << endl;
		waitKey(10);
		return 1;
	}

	cout << cv::getBuildInformation << endl; // get build inf - contrib is installed ?

	Mat frame, grayFrame, grayROI;
	bool check = video.read(frame);// ilk frame'i al
	resize(frame, frame, Size(win_size_w, win_size_h), 0.0, 0.0, INTER_CUBIC); // frame boyutlarýný ayarla 


	Rect2d bbox;//selectROI(frame); // ROI select
	float confidence = yolov4.getObject<Rect2d>(frame, bbox);
	CV_Assert(confidence>0);
	Rect2d exp_bbox = bbox; // expected box -- mossenin verdiði yeni konumda boyutlandýrýlacak box 
	cout << "model has done..." << endl;
	cvtColor(frame, grayFrame, COLOR_BGR2GRAY); // mosse takes single channel img
	grayROI = grayFrame(bbox); // ROI the gray !!!

	
	Mat probmap; // target pixels probabilities map
	Mat back_hist_old = Mat(Size(1, 256), CV_32F, Scalar(0)); // TEST --> eski histogramý tutmak için 
	Size distSize = Size(grayROI.cols / frame_ratio, grayROI.rows / frame_ratio); // outer box extra size
	foregroundHistProb(grayROI, distSize, back_hist_old, probmap, val); // calc prob map that represents target shape


	Size baseSize = momentSize(probmap); // get size from moments with using prob's map
	cout << "base values[width-height] =  " << baseSize << endl;


	// Display inner box(original bbox size) and processed (outer) bbox size.
	Rect innerBox = Rect(Center(bbox) - Point((exp_bbox.width + distSize.width) / 2, (exp_bbox.height + distSize.height) / 2),
		Center(bbox) + Point((exp_bbox.width + distSize.width) / 2, (exp_bbox.height + distSize.height) / 2));
	rectangle(grayROI, innerBox, Scalar(255, 0, 0)); // background rect drawing
	imshow("foreprob", grayROI);
	rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
	imshow(winname, frame);

	waitKey(0);
	tracker->init(frame, bbox); // initialize tracker
	bool track_or_detect = true;
	while (true)
	{
		if (mode)
		{
			if (video.read(frame));
			else
				break; // if frame error occurs

			if (track_or_detect)
			{
				// Start timer
				resize(frame, frame, Size(win_size_w, win_size_h), 0.0, 0.0, INTER_CUBIC); // ekraný tekrar boyutlandýrma 
				double timer = (double)getTickCount(); // sayacý baþlatýyoruz

				cvtColor(frame, grayFrame, COLOR_BGR2GRAY); // frame graye dönüþtürüldü
				check = tracker->update(grayFrame, bbox); // MOSSE uygulandý

				//examined box update
				distSize = Size(grayROI.cols / frame_ratio, grayROI.rows / frame_ratio);
				exp_bbox = Rect(Center(bbox) - Point((exp_bbox.width + distSize.width) / 2, (exp_bbox.height + distSize.height) / 2),
					Center(bbox) + Point((exp_bbox.width + distSize.width) / 2, (exp_bbox.height + distSize.height) / 2)); // get new object square position to calc size
				grayROI = grayFrame(exp_bbox); // ROI with new position but old size 

				foregroundHistProb(grayROI, distSize, back_hist_old, probmap, val);

				// moment hesabý
				Size nSize = momentSize(probmap);
				exp_bbox = Rescale(bbox, baseSize, nSize); // rescale with moments

				float fps = getTickFrequency() / ((double)getTickCount() - timer); // sayacý al

				if (check)
				{
					rectangle(frame, exp_bbox, Scalar(255, 0, 0), 2, 1);
					drawMarker(frame, Center(bbox), Scalar(0, 255, 0)); //mark the center 
						// FPS'i yaz
					putText(frame, "FPS : " + SSTR(int(fps)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
				}
				else
				{
					// Tracking failure detected.
					putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
					putText(frame, "Model initiated", Point(100, 120), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(220, 0, 20), 2);
					track_or_detect = false;
				}
			}
			else
			{
				confidence = yolov4.getObject<Rect2d>(frame, bbox);
				rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
				drawMarker(frame, Center(bbox), Scalar(0, 255, 0)); //mark the center 
				track_or_detect = true;
			}
		}
		// Son frame'i göster
		imshow(winname, frame);
		waitKey(0); // frame by frame gitmek için -- REMOVE BEFORE FLIGHT !!!

		int keyboard = waitKey(5); // kullanýcýdan kontrol tuþu al 
		if (keyboard == 'q' || keyboard == 27) // quit
			break;
		else if (keyboard == 'p' || keyboard == 112) // pause
			mode = 0;
		else if (keyboard == 'r' || keyboard == 114) // return
			mode = 1;

	}
	return 0;
}

//PID Update function 
float PID_update(PID* pid, float setpoint, float measurement) {

	float error = setpoint - measurement;

	/*
	* Proportional
	*/
	float proportional = pid->Kp * error;

	/*
	* Integral
	*/
	pid->integrator = pid->integrator + 0.5f * pid->Ki * pid->T * (error + pid->prevError);

	/* Anti-wind-up via integrator clamping */
	if (pid->integrator > pid->limMaxInt) {

		pid->integrator = pid->limMaxInt;

	}
	else if (pid->integrator < pid->limMinInt) {

		pid->integrator = pid->limMinInt;

	}

	/*
	* Derivative (band-limited differentiator)
	*/

	pid->differentiator = -(2.0f * pid->Kd * (measurement - pid->prevMeasurement)	/* Note: derivative on measurement, therefore minus sign in front of equation! */
		+ (2.0f * pid->tau - pid->T) * pid->differentiator)
		/ (2.0f * pid->tau + pid->T);

	/*
	* Compute output and apply limits
	*/
	pid->out = proportional + pid->integrator + pid->differentiator;

	if (pid->out > pid->limMax) {

		pid->out = pid->limMax;

	}
	else if (pid->out < pid->limMin) {

		pid->out = pid->limMin;

	}

	/* Store error and measurement for later use */
	pid->prevError = error;
	pid->prevMeasurement = measurement;

	/* Return controller output */
	return pid->out;

}

void PID_init(PID* pid) {

	pid->integrator = 0.0f;
	pid->prevError = 0.0f;

	pid->differentiator = 0.0f;
	pid->prevMeasurement = 0.0f;

	pid->out = 0.0f;

}

