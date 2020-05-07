#pragma once
#include <opencv2/opencv.hpp>
#include "cpp_filter_util/lib_filter/filter.hpp"
#include "cpp_filter_util/lib_filter/parameterized_filter.hpp"

namespace illust_image_similarity {
	namespace feature {
		Filter equalizeHist { [](cv::Mat src) -> cv::Mat {
			cv::Mat dst;
			cv::equalizeHist(src, dst);
			return dst;
		}};
		Filter normalize { [](cv::Mat src) -> cv::Mat {
			assert(src.type() == CV_8UC1);
			cv::Mat dst;
			cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
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
		ParameterizedFilter blur { [](cv::Mat src, int sigma) -> cv::Mat {
			cv::Mat dst;
			cv::GaussianBlur(src, dst, cv::Size(7, 7), sigma, sigma, cv::BORDER_REFLECT_101);
			return dst;
		}};
	}
}


