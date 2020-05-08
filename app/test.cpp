#include <iostream>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <illust_image_similarity.hpp>

using namespace illust_image_similarity;

	using namespace feature;



std::vector<std::string> getFileNames(std::string pathStr) {
	std::vector<std::string> res;
	namespace fs = boost::filesystem;
	const fs::path path(pathStr);

	BOOST_FOREACH(const fs::path& p, std::make_pair(fs::directory_iterator(path), fs::directory_iterator())) {
		if (!fs::is_directory(p))
			res.push_back(p.filename().string());
	}
	return res;
}

int main(){
	// b1, b3, y1 -> double edge
	// b2, p1, p2, p3 -> single edge
	// p3 -> no edge
	cv::Mat img_blue1 = cv::imread("../tentee_patch/dream/0001.png");
	cv::Mat img_blue2 = cv::imread("../tentee_patch/dream/0087.png");
	cv::Mat img_blue3 = cv::imread("../tentee_patch/dream/0006.png");
	cv::Mat img_yell1 = cv::imread("../tentee_patch/dream/0005.png");
	cv::Mat img_pink1 = cv::imread("../tentee_patch/dream/0002.png");
	cv::Mat img_pink2 = cv::imread("../tentee_patch/dream/0132.png");
	cv::Mat img_pink3 = cv::imread("../tentee_patch/dream/0138.png");
	cv::Mat img_pink4 = cv::imread("../tentee_patch/dream/0004.png");
	cv::Mat mask = cv::imread("tentee_patch/mask/mask.png", 0);
	// memo: 1と12は近い, 2は遠い

	auto fileList = getFileNames("../tentee_patch/dream/");

	//cv::Mat feature = img | bilateral | sobel | gray | blur(6) | normalize;
	//cv::imwrite("feature.png", feature);

	std::map<std::string, cv::Mat> images = {
		{ "blue1", img_blue1 },
		{ "blue2", img_blue2 },
		{ "blue3", img_blue3 },
		{ "yell1", img_yell1 },
		{ "pink1", img_pink1 },
		{ "pink2", img_pink2 },
		{ "pink3", img_pink3 },
		{ "pink4", img_pink4 }
	};
	/*std::map<std::string, cv::Mat> images;
	{
		int cnt = 0;
		for(auto&& fName : fileList){
			cv::Mat img = cv::imread("../tentee_patch/dream/" + fName);
			if(++cnt % 10 == 0)
				std::cout << "loading: "  << cnt << " images..." << std::endl;
			if(img.data)
				images[fName] = img;
		}
	}*/

std::cout << "start" << std::endl;
	std::map<std::string, cv::MatND> hists;
	for(auto&& [name, img] : images)
		hists[name] = img | histgramHue(mask);
	std::map<std::string, cv::MatND> placements;
	for(auto&& [name, img] : images)
		placements[name] = img | bilateral | sobel | gray | blur(6) | normalize;
	std::map<std::string, std::vector<double>> directions;
	for(auto&& [name, img] : images){
		directions[name] = img | blur(2) | gray | directionVec(mask);
		std::cout << std::accumulate(directions[name].begin(), directions[name].end(), double{0.}, [](double acc, double d) { return d*d + acc; });
		std::cout << "directions[" << name << "] = ("
			<< directions[name][0] << ", "
			<< directions[name][1] << ", "
			<< directions[name][2] << ", "
			<< directions[name][3] << ", "
			<< directions[name][4] << ", "
			<< directions[name][5] << ", "
			<< directions[name][6] << ", "
			<< directions[name][7] << ")"
			<< std::endl;
	}

	int cnt = 0;
	auto cdot = [](const std::vector<double>& lhs, const std::vector<double>& rhs) -> double {
		int dim = std::min(lhs.size(), rhs.size());
		double res = 0;
		for(int i=0; i<dim; ++i)
			res += lhs[i] * rhs[i];
		return res;
	};
	for(auto&& [name, img] : images){
		if(++cnt % 10 == 0)
			std::cout << cnt << " / " << images.size() << " ... " << std::endl;

		std::vector<std::pair<double, std::string>> list_hue;
		std::vector<std::pair<double, std::string>> list_edge;
		std::vector<std::pair<double, std::string>> list_dir;
		auto&& pls  = placements.at(name);
		auto&& hist = hists.at(name);
		auto&& dir  = directions.at(name);
		// target
		for(auto&& [tName, tImg] : images){
			auto&& tPls  = placements.at(tName);
			auto&& tHist = hists.at(tName);
			auto&& tDir  = directions.at(tName);
			list_hue.emplace_back(cv::compareHist(hist, tHist, cv::HISTCMP_CORREL), tName);

			//list_edge.emplace_back(cv::norm(img, tImg, cv::NORM_L1), tName);
			cv::Mat tmp;
			// Normalized Cross-Correlation
			cv::matchTemplate(pls, tPls, tmp, CV_TM_CCOEFF_NORMED);
			list_edge.emplace_back(tmp.at<float>(0,0), tName);

			// dot product
			list_dir.emplace_back(cdot(dir, tDir), tName);
		}

		std::sort(list_edge.begin(), list_edge.end());
		std::reverse(list_edge.begin(), list_edge.end());

		std::sort(list_dir.begin(), list_dir.end());
		std::reverse(list_dir.begin(), list_dir.end());

		std::sort(list_hue.begin(), list_hue.end());
		std::reverse(list_hue.begin(), list_hue.end());

		std::cout << " Hue  > ";
		std::cout << name << " -> ";
		for(auto [tHist, tName] : list_hue)
			std::cout << tName << "(" << tHist << "), ";
		std::cout << std::endl;

		std::cout << " Edge > ";
		std::cout << name << " -> ";
		for(auto [tHist, tName] : list_edge)
			std::cout << tName << "(" << tHist << "), ";
		std::cout << std::endl;

		std::cout << " Dir  > ";
		std::cout << name << " -> ";
		for(auto [tHist, tName] : list_dir)
			std::cout << tName << "(" << tHist << "), ";
		std::cout << std::endl;
		std::cout << std::endl;

	}

std::cout << "end" << std::endl;
	// keypoint
	// std::vector<cv::KeyPoint> kpts;
	// cv::Mat desc;
	// feature = cv::xfeatures2d::SIFT::create();
	// feature->detectAndCompute(img_pink1, img_pink3, kpts, desc);
	// cv::FAST(img_blue1, keypts, orb_params_.ini_fast_thr_, true);

	// std::cout << "near: " << cv::compareHist(hist, hist12, cv::HISTCMP_CORREL) << std::endl;
	// std::cout << "far: " << cv::compareHist(hist, hist2, cv::HISTCMP_CORREL) << std::endl;
	// std::cout << "far: " << cv::compareHist(hist2, hist12, cv::HISTCMP_CORREL) << std::endl;
}

