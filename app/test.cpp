#include <iostream>
#include <opencv2/opencv.hpp>
#include <illust_image_similarity.hpp>

using namespace illust_image_similarity;

	using namespace feature;

int main(){
	cv::Mat img = cv::imread("../tentee_patch/dream/0001.png");
	cv::Mat blurred = img | bilateral | sobel | gray | blur(2) | normalize;
	cv::imwrite("blurred.png", blurred);
}

