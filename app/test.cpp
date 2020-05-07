#include <iostream>
#include <opencv2/opencv.hpp>
#include <illust_image_similarity.hpp>

int main(){
	std::cout << Test::test() << std::endl;
	cv::Mat img_s_x, img_s_y, img_s;
	cv::Mat img = cv::imread("../tentee_patch/dream/0001.png", 0);
	cv::Sobel(img, img_s_x, CV_8UC1, 1, 0, 3);
	cv::Sobel(img, img_s_y, CV_8UC1, 0, 1, 3);
	img_s = abs(img_s_x) + abs(img_s_y);
	cv::convertScaleAbs(img_s, img_s, 1, 0);  // 8bitに変換しているだけ
	cv::imwrite("sobel.png", img_s);
}

