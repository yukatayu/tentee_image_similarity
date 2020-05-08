#include <iostream>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <illust_image_similarity.hpp>

using namespace illust_image_similarity::feature;

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
	cv::Mat mask = cv::imread("tentee_patch/mask/mask.png", 0);
	// vvv For testing vvv //
	// b1, b3, y1 -> double edge
	// b2, p1, p2, p3 -> single edge
	// p3 -> no edge
	/*std::map<std::string, cv::Mat> images = {
		{ "blue1", cv::imread("../tentee_patch/dream/0001.png") },
		{ "blue2", cv::imread("../tentee_patch/dream/0087.png") },
		{ "blue3", cv::imread("../tentee_patch/dream/0006.png") },
		{ "blue4", cv::imread("../tentee_patch/dream/0255.png") },
		{ "yell1", cv::imread("../tentee_patch/dream/0005.png") },
		{ "pink1", cv::imread("../tentee_patch/dream/0002.png") },
		{ "pink2", cv::imread("../tentee_patch/dream/0132.png") },
		{ "pink3", cv::imread("../tentee_patch/dream/0138.png") },
		{ "pink4", cv::imread("../tentee_patch/dream/0004.png") }
	};*/
	// memo: 1と12は近い, 2は遠い
	// memo: 255と2はどちらも縞
	// ^^^ For testing ^^^ //
	std::map<std::string, cv::Mat> images;

	auto fileList = getFileNames("../tentee_patch/dream/");
	{
		int cnt = 0;
		for(auto&& fName : fileList){
			cv::Mat img = cv::imread("../tentee_patch/dream/" + fName);
			if(++cnt % 10 == 0)
				std::cout << "loading: "  << cnt << " images..." << std::endl;
			if(img.data)
				images[fName] = img;
		}
	}

	std::cout << "start" << std::endl;

	std::map<std::string, cv::MatND> hists;
	for(auto&& [name, img] : images)
		hists[name] = img | hueHistgramAlgorithm(mask);

	std::map<std::string, cv::MatND> placements;
	for(auto&& [name, img] : images)
		placements[name] = img | placementAlgorithm;

	std::map<std::string, std::vector<double>> directions;
	for(auto&& [name, img] : images)
		directions[name] = img | directionPreprocess | directionVec(mask);

	int cnt = 0;
	auto cdot = [](const std::vector<double>& lhs, const std::vector<double>& rhs) -> double {
		int dim = std::min(lhs.size(), rhs.size());
		double res = 0;
		for(int i=0; i<dim; ++i)
			res += lhs[i] * rhs[i];
		return res;
	};
	// from, type [hue, edge, dir], to, index (0-), score (0-1)
	const int maxIndex = 5;
	std::ofstream data("recommend_data.csv");
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
			if(name == tName) continue; // 自身とは比較しない (edge,dir,hueのスコアは常に1)
			auto&& tPls  = placements.at(tName);
			auto&& tHist = hists.at(tName);
			auto&& tDir  = directions.at(tName);
			list_hue.emplace_back(cv::compareHist(hist, tHist, cv::HISTCMP_CORREL), tName);

			cv::Mat tmp;
			// Normalized Cross-Correlation
			//list_edge.emplace_back(cv::norm(img, tImg, cv::NORM_L1), tName);
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

		for(int i=0; i<maxIndex && i < list_hue.size(); ++i){
			auto [tHist, tName] = list_hue[i];
			data << name << "," << "hue" << "," << tName << "," << i << "," << tHist << "\n";
		}

		for(int i=0; i<maxIndex && i < list_edge.size(); ++i){
			auto [tHist, tName] = list_edge[i];
			data << name << "," << "edge" << "," << tName << "," << i << "," << tHist << "\n";
		}

		for(int i=0; i<maxIndex && i < list_dir.size(); ++i){
			auto [tHist, tName] = list_dir[i];
			data << name << "," << "dir" << "," << tName << "," << i << "," << tHist << "\n";
		}
		data << std::flush;

	}

	std::cout << "end" << std::endl;
}

