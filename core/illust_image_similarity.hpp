#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "primitive.hpp"

namespace illust_image_similarity {
	namespace feature {
		// アルゴリズムの流れの定義

		// 色相
		auto hueHistgramAlgorithm = histgramHue;

		// ディティールの分布
		auto   placementAlgorithm = bilateral | sobel | gray | blur(6) | normalize;

		// 模様の方向 (前処理)
		auto  directionPreprocess = blur(2) | gray;
	}
}

