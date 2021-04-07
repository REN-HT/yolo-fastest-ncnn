#include "cpu.h"
#include "net.h"
#include "gpu.h"
#include "benchmark.h"
#include "datareader.h"

#include<vector>
#include<stdio.h>
#include<fstream>
#include<algorithm>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

int demo(cv::Mat& image, ncnn::Net& detector, int detector_size_with, int detector_size_height) {
	// 检测类别
	//static const char* class_names[] = { 
	//	"background","aeroplane","bicycle","bird","boat",
	//	"bottle","bus","car","cat","chair","cow","diningtable",
	//	"dog","horse","motorbike","person","pottedplant","sheep",
	//	"sofa","train","tvmonitor"
	//};

	std::vector<std::string> class_names;
	std::ifstream infile("coco_names.txt",std::ios::in);
	for (int i = 0; i < 80; i++) {
		std::string str;
		infile >> str;
		class_names.push_back(str);

	}
	infile.close();


	cv::Mat bgr = image.clone();
	int image_width = bgr.cols;
	int image_height = bgr.rows;

	ncnn::Mat input = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
		bgr.cols, bgr.rows, detector_size_with, detector_size_height);
	// 数据预处理
	const float mean_vals[3] = { 0.f, 0.f, 0.f };
	const float norm_vals[3] = { 1 / 255.f,1 / 255.f,1 / 255.f };
	input.substract_mean_normalize(mean_vals, norm_vals);

	ncnn::Extractor ex = detector.create_extractor();
	ex.set_num_threads(8);
	ex.input("data", input);
	ncnn::Mat out;
	ex.extract("output", out);

	for (int i = 0; i < out.h; ++i) {
		int label;
		float x1, y1, x2, y2, score;
		const float* values = out.row(i);

		label = values[0];
		score = values[1];
		x1 = values[2] * image_width;
		y1 = values[3] * image_height;
		x2 = values[4] * image_width;
		y2 = values[5] * image_height;
		// 处理越界坐标
		if (x1 < 0)x1 = 0;
		if (y1 < 0)y1 = 0;
		if (x2 < 0)x2 = 0;
		if (y2 < 0)y2 = 0;
		if (x1 > image_width) x1 = image_width;
		if (y1 > image_height) y1 = image_height;
		if (x2 > image_width) x2 = image_width;
		if (y2 > image_height) y2 = image_height;

		cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 1, 1, 0);

		char text[256];
		sprintf_s(text, "%s %.1f%%", class_names[label], score * 100);
		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		cv::putText(image, text, cv::Point(x1, y1 + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
	return 0;
}

int test() {
	// 定义yolo-fatest检测器
	ncnn::Net detector;
	detector.load_param("yolo-fastest-1.1.param");
	detector.load_model("yolo-fastest-1.1.bin");
	// 定义输入图像尺寸
	int detector_size_width = 300;
	int detector_size_height = 300;
	// 测试图像
	cv::Mat image = cv::imread("C:/AllProgram/testimage/cat2.jpg");
	// 调用函数开始检测
	demo(image, detector, detector_size_width, detector_size_height);
	// 显示检测结果
	cv::imshow("demo", image);
	cv::waitKey(0);
	return 0;
}

int main() {
	test();
	return 0;
}
