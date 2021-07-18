#pragma once
#include <opencv2/dnn.hpp>
#include "common.hpp"

using namespace dnn;

struct model_param
{
	const string modelName;
	string modelPath;
	string configPath;
	string framework;
	int backend;
	int target; // hedef iþlem arayüzü 
	size_t asyncNumReq; // asenkron mode henüz kurulmadý !!! 
};


class model
{
	public:
		// frame options
		float confThreshold;
		float nmsThreshold;
		
		float scale;
		cv::Scalar mean;
		bool swapRB;
		int inpWidth; // preprocess sizes
		int inpHeigth;
		
		std::vector<std::string> classes;
		model_param m_param;

		void get_classes(std::string file);

		model(model_param &param): m_param(param)
		{
			net = readNet(m_param.modelPath, m_param.configPath, m_param.framework); // read nn cfg
			net.setPreferableBackend(m_param.backend); // backend ? 
			net.setPreferableTarget(m_param.target); // target frame platform
			outNames = net.getUnconnectedOutLayersNames(); // get output names
		}

		float getObject(Mat frame, Rect& bbox)
		{
			float confidence;
			preprocess(frame, net, Size(inpWidth, inpHeigth), scale, this->mean, this->swapRB);
			
			std::vector<Mat> outs;
			net.forward(outs, outNames);

			postprocess(frame, outs, net, m_param.backend);
			//if(boxes.size()>1)
				//choosing side.... // !!!!!!!!!!! TAMAMLA !!!!!!

			bbox = this->boxes.back;
			this->boxes.clear;
			confidence = this->confidences.back;
			this->confidences.clear;

			return confidence;
		}
		
	private:
		Net net;
		std::vector<String> outNames;
		std::vector<Rect> boxes;
		std::vector<int> classIds;
		std::vector<float> confidences;

		inline void preprocess(const Mat& frame, Net& net, Size inpSize, float scale, const Scalar& mean, bool swapRB);
		void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, int backend);
};


void model::get_classes(std::string file)
{
	std::ifstream ifs(file.c_str());
	if (!ifs.is_open())
		CV_Error(Error::StsError, "File " + file + " not found");
	std::string line;
	while (std::getline(ifs, line))
	{
		classes.push_back(line);
	}
}

inline void model::preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
	const Scalar& mean, bool swapRB)
{
	static Mat blob;
	// Create a 4D blob from a frame.
	if (inpSize.width <= 0) inpSize.width = frame.cols;
	if (inpSize.height <= 0) inpSize.height = frame.rows;
	blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

	// Run a model.
	net.setInput(blob, "", scale, mean);
	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		resize(frame, frame, inpSize);
		Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
		net.setInput(imInfo, "im_info");
	}
}

void model::postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, int backend)
{
	static std::vector<int> outLayers = net.getUnconnectedOutLayers();
	static std::string outLayerType = net.getLayer(outLayers[0])->type;

	if (outLayerType == "DetectionOutput")
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() > 0);
		for (size_t k = 0; k < outs.size(); k++)
		{
			float* data = (float*)outs[k].data;
			for (size_t i = 0; i < outs[k].total(); i += 7)
			{
				float confidence = data[i + 2];
				if (confidence > confThreshold)
				{
					int left = (int)data[i + 3];
					int top = (int)data[i + 4];
					int right = (int)data[i + 5];
					int bottom = (int)data[i + 6];
					int width = right - left + 1;
					int height = bottom - top + 1;
					if (width <= 2 || height <= 2)
					{
						left = (int)(data[i + 3] * frame.cols);
						top = (int)(data[i + 4] * frame.rows);
						right = (int)(data[i + 5] * frame.cols);
						bottom = (int)(data[i + 6] * frame.rows);
						width = right - left + 1;
						height = bottom - top + 1;
					}
					this->classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
					this->boxes.push_back(Rect(left, top, width, height));
					this->confidences.push_back(confidence);
				}
			}
		}
	}
	else if (outLayerType == "Region")
	{
		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;

			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint); // IoU
			
				if (confidence > confThreshold)
				{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					this->classIds.push_back(classIdPoint.x);
					this->confidences.push_back((float)confidence);
					this->boxes.push_back(Rect(left, top, width, height));
				}
			}
		}
	}
	else
		CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

	// NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
	// or NMS is required if number of outputs > 1
	if (outLayers.size() > 1 || (outLayerType == "Region" && backend != DNN_BACKEND_OPENCV))
	{
		std::map<int, std::vector<size_t> > class2indices;
		for (size_t i = 0; i < this->classIds.size(); i++)
		{
			if (this->confidences[i] >= confThreshold)
			{
				class2indices[this->classIds[i]].push_back(i);
			}
		}
		std::vector<Rect> nmsBoxes;
		std::vector<float> nmsConfidences;
		std::vector<int> nmsClassIds;
		for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
		{
			std::vector<Rect> localBoxes;
			std::vector<float> localConfidences;
			std::vector<size_t> classIndices = it->second;
			for (size_t i = 0; i < classIndices.size(); i++)
			{
				localBoxes.push_back(this->boxes[classIndices[i]]);
				localConfidences.push_back(this->confidences[classIndices[i]]);
			}
			std::vector<int> nmsIndices;
			NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
			for (size_t i = 0; i < nmsIndices.size(); i++)
			{
				size_t idx = nmsIndices[i];
				nmsBoxes.push_back(localBoxes[idx]);
				nmsConfidences.push_back(localConfidences[idx]);
				nmsClassIds.push_back(it->first);
			}
		}
		this->boxes = nmsBoxes;
		this->classIds = nmsClassIds;
		this->confidences = nmsConfidences;
	}
}