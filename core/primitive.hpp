#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include "cpp_filter_util/lib_filter/filter.hpp"
#include "cpp_filter_util/lib_filter/parameterized_filter.hpp"

namespace illust_image_similarity {
	namespace util{
		ParameterizedFilter foreach { [](auto srcs, auto filter){
			decltype(srcs) ress;
			for(auto&& src : srcs)
				ress.emplace_back(src | filter);
			return ress;
		}};
	}
	namespace feature {
		Filter equalizeHist { [](cv::Mat src) -> cv::Mat {
			cv::Mat dst;
			cv::equalizeHist(src, dst);
			return dst;
		}};
		Filter normalize { [](cv::Mat src) -> cv::Mat {
			assert(src.type() == CV_8UC1);
			cv::Mat dst;
			cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, -1);
			return dst;
		}};
		Filter bilateral { [](cv::Mat src) -> cv::Mat {
			cv::Mat dst;
			cv::bilateralFilter(src, dst,
				15,  // 近傍の直径
				250, // color sigma
				50   // coord sigma
			);
			return dst;
		}};
		Filter sobel { [](cv::Mat src) -> cv::Mat {
			cv::Mat img_s_x, img_s_y, img_s;
			cv::Sobel(src, img_s_x, CV_8UC1, 1, 0, 3);
			cv::Sobel(src, img_s_y, CV_8UC1, 0, 1, 3);
			img_s = abs(img_s_x) + abs(img_s_y);
			cv::convertScaleAbs(img_s, img_s, 1, 0);
			return img_s;
		}};
		Filter gray { [](cv::Mat src) -> cv::Mat {
			cv::Mat dst;
			cv::cvtColor(src, dst, CV_RGB2GRAY);
			return dst;
		}};
		ParameterizedFilter conv2D { [](cv::Mat src, cv::Mat kernel) -> cv::Mat {
			cv::Mat dst;
			filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
			return dst;
		}};
		Filter abs{ [](cv::Mat src) -> cv::Mat {
			return cv::abs(src);
		}};
		ParameterizedFilter mean{ [](cv::Mat src, cv::Mat mask) -> double {
			return cv::mean(src, mask)[0];
		}};
		// 返り値は長さ8
		ParameterizedFilter directionVec { [](cv::Mat src, cv::Mat mask) -> std::vector<double> {
			// direction: [0, 8)
			std::vector<double> res(8);
			// create sobel kernel
			constexpr double kernels_raw[8][3][3] = {
				{
					{-1,-1, 0},
					{-1, 0, 1},
					{ 0, 1, 1}
				}, {
					{-1.5,-1.5,   0},
					{   0,   0,   0},
					{   0, 1.5, 1.5}
				}, {
					{-1,-1,-1},
					{ 0, 0, 0},
					{ 1, 1, 1}
				}, {
					{   0,-1.5,-1.5},
					{   0,   0,   0},
					{ 1.5, 1.5,   0}
				}, {
					{ 0,-1,-1},
					{ 1, 0,-1},
					{ 1, 1, 0}
				}, {
					{   0, 0,-1.5},
					{ 1.5, 0,-1.5},
					{ 1.5, 0,   0}
				}, {
					{ 1, 0,-1},
					{ 1, 0,-1},
					{ 1, 0,-1}
				}, {
					{ 1.5, 0,   0},
					{ 1.5, 0,-1.5},
					{   0, 0,-1.5}
				}
			};
			// これは精度が悪かった (方向特性が低い…？)
			/*constexpr double kernels_raw[8][3][3] = {
				{
					{ 1, 1, 0},
					{ 1,-6, 1},
					{ 0, 1, 1}
				}, {
					{ 1.5, 1.5,   0},
					{   0,  -6,   0},
					{   0, 1.5, 1.5}
				}, {
					{ 1, 1, 1},
					{ 0,-6, 0},
					{ 1, 1, 1}
				}, {
					{   0, 1.5, 1.5},
					{   0,  -6,   0},
					{ 1.5, 1.5,   0}
				}, {
					{ 0, 1, 1},
					{ 1,-6, 1},
					{ 1, 1, 0}
				}, {
					{   0, 0, 1.5},
					{ 1.5,-6, 1.5},
					{ 1.5, 0,   0}
				}, {
					{ 1, 0, 1},
					{ 1,-6, 1},
					{ 1, 0, 1}
				}, {
					{ 1.5, 0,   0},
					{ 1.5,-6, 1.5},
					{   0, 0, 1.5}
				}
			};*/
			std::vector<cv::Mat> kernels(8);
			for(int i=0; i<8; ++i){
				kernels[i] = cv::Mat(3, 3, CV_32FC1);
				const double* kernel_target = &kernels_raw[i][0][0];
				for(int n=0; n<9; ++n)
					kernels[i].at<float>(n) = kernel_target[n];
			}
			// Edge Detection
			for(int i=0; i<8; ++i)
				res[i] = src | conv2D(kernels[i]) | abs | mean(mask);

			// 回転は行わない方が結果が良かった
			// Rotate
			//Int offset = std::max_element(res.begin(), res.end()) - res.begin();
			//Std::vector<double> res_tmp = res;
			//For(int i=0; i<8 - 1; ++i)
			//	res_tmp.push_back(res[i]);

			////std::cout << " (";
			////for(int i=0; i<15; ++i) std::cout << res_tmp[i] << ", ";
			////std::cout << " | offset = " << offset << std::endl;

			//For(int i=0; i<8; ++i)
			//	res[i] = res_tmp[i + offset];
			//// Mirror
			//If(res[2] < res[6]){
			//	std::swap(res[1], res[7]);
			//	std::swap(res[2], res[6]);
			//	std::swap(res[3], res[5]);
			//}

			// Normalize
			double norm = std::sqrt(std::accumulate(res.begin(), res.end(), double{0.}, [](double acc, double d) {
				return d*d + acc;
			}));
			if(norm != 0)
				for(int i=0; i<8; ++i)
					res[i] /= norm;
			return res;
		}};
		ParameterizedFilter blur { [](cv::Mat src, int sigma) -> cv::Mat {
			cv::Mat dst;
			cv::GaussianBlur(src, dst, cv::Size(sigma*2+1, sigma*2+1), sigma, sigma, cv::BORDER_REFLECT_101);
			return dst;
		}};
		Filter split { [](cv::Mat src) -> std::vector<cv::Mat>{
			std::vector<cv::Mat> res;
			cv::split(src, res);
			return res;
		}};
		Filter histgram { [](cv::Mat src) -> cv::MatND {
			// memo: https://qiita.com/mask00/items/0797d5505c5159fc6583
			assert(src.type() == CV_8UC3);
			cv::MatND dst;
			const int sizes[] = {256, 256, 256};
			//const int sizes[] = {16, 16, 16};
			const float range[] = {0, 256};
			const float* ranges[] = {range, range, range};
			int channels[] = {0, 1, 2};
			cv::calcHist(&src, 1, channels, cv::Mat(), dst, 3, sizes, ranges);
			return dst;
		}};
		ParameterizedFilter histgramHue { [](cv::Mat src, cv::Mat mask) -> cv::MatND {
			cv::Mat hsv;
			cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
			int histSize[] = {32};
			float hranges[] = { 0, 180 };
			const float* ranges[] = { hranges };
			int channels[] = {0};
			cv::MatND hist;
			cv::calcHist( &hsv, 1, channels, mask, hist, 1, histSize, ranges);
			return hist;
		}};
		ParameterizedFilter histgramValue { [](cv::Mat src, cv::Mat mask) -> cv::MatND {
			cv::Mat hsv;
			cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
			int histSize[] = {32};
			float hranges[] = { 0, 256 };
			const float* ranges[] = { hranges };
			int channels[] = {2};
			cv::MatND hist;
			cv::calcHist( &hsv, 1, channels, mask, hist, 1, histSize, ranges);
			return hist;
		}};
	}
}

