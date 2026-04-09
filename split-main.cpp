/*
 * @Description: 
 * @version: 
 * @Author: ccy
 * @Date: 2025-11-25 14:00:06
 * @LastEditors: ccy
 * @LastEditTime: 2026-01-09 10:43:50
 */
#include <iostream>
#include <deque>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "cuda_runtime_api.h"
#include "cuda_utils.h"
#include "logging.h"
#include "preprocess.h"
#define DEVICE 0  // GPU id

using namespace nvinfer1;
static Logger gLogger;

std::vector<std::vector<cv::Point>> splitLanes(
        const cv::Mat& seg_mask_,
        const cv::Mat& lane_seg,
        const cv::Mat& road_mask);
void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
void free_buffers(float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
void processMultiLaneSegmentation(cv::Mat& road_mask, cv::Mat& lane_seg, cv::Mat& display_img);

struct LaneLine {
    std::vector<cv::Point> points;
    float avg_slope;
    float x_at_bottom;
};

template <typename T>
T clamp(T value, T low, T high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

bool isImageFile(const std::string& name) {
    static const std::vector<std::string> exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"
    };

    for (const auto& ext : exts) {
        if (name.size() >= ext.size() &&
            name.compare(name.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    return false;
}

std::deque<std::string> listImagesInDir(const std::string& dirPath)
{
    std::deque<std::string> result;

    DIR* dir = opendir(dirPath.c_str());
    if (!dir) {
        perror("opendir");
        return result;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;

        // 过滤 . 和 ..
        if (name == "." || name == "..") continue;

        // 拼接为绝对路径
        std::string fullPath = dirPath + "/" + name;

        // 判断是否是文件而不是目录
        struct stat st;
        if (stat(fullPath.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            if (isImageFile(name)) {
                result.push_back(fullPath);
            }
        }
    }

    closedir(dir);
    return result;
}

int main()
{
	cudaSetDevice(DEVICE);
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
	char* modelData{ nullptr };
	size_t modelSize{ 0 };
	std::vector<std::vector<int>> color_list = { { 245, 249, 58 },{137, 131, 220},{42, 152, 73},{100, 196, 211} };
	bool debug = false;
	const std::string engine_file_path = "./model/scene.engine";
	const std::string img_dir = "./image";
    const std::string result_dir = "./result";


	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	std::ifstream engine_file(engine_file_path, std::ios::binary);
	if (engine_file.good()) {
		engine_file.seekg(0, engine_file.end);
		modelSize = engine_file.tellg();
		engine_file.seekg(0, engine_file.beg);
		modelData = new char[modelSize];
		assert(modelData);
		engine_file.read(modelData, modelSize);
		engine_file.close();
	}
	else
	{
		std::cerr << "cannot read engine file£¡" << std::endl;
	}

    ICudaEngine* engine = nullptr;
    try
    {
        engine = runtime->deserializeCudaEngine(modelData, modelSize, nullptr);
        assert(engine != nullptr);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
	if (debug)
	{
		for (int bi = 0; bi < engine->getNbBindings(); bi++)
		{
			if (engine->bindingIsInput(bi) == true)
			{
				printf("Binding %d (%s): ", bi, engine->getBindingName(bi));
			}
			else
			{
				printf("Binding %d (%s): ", bi, engine->getBindingName(bi));
			}
			auto dims = engine->getTensorShape(engine->getBindingName(bi));
			std::cout << "(";
			for (int i = 0; i < dims.nbDims; ++i) {
				std::cout << dims.d[i];
				if (i != dims.nbDims - 1)std::cout << ",";
			}
			std::cout << ")" << std::endl;;
		}
	}

	cuda_preprocess_init();

	float* cpu_output_buffer = nullptr;
	//float* cpu_input_buffer = new float[3 * 384 * 640];//tmp
	float* gpu_buffers[2];
	float pad[2] = { 0.0,12.0 };
	prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);

	context->setTensorAddress("input", gpu_buffers[0]);
	context->setTensorAddress("output", gpu_buffers[1]);

	cv::Mat lane_set = cv::Mat::zeros(1080, 1920, CV_8U);

    std::deque<std::string> images = listImagesInDir(img_dir);

	for(auto img_path : images)
	{
        size_t pos = img_path.find_last_of("/");

        std::string filename;
        if (pos != std::string::npos)
            filename = img_path.substr(pos + 1);
        else
            filename = img_path; // 本身就是文件名

		cv::Mat frame;
		cv::Mat input_img;
		cv::Mat input_img_clone;
		frame = cv::imread(img_path);

		if (frame.empty())
		{
			break;
		}

		cv::resize(frame, input_img, cv::Size(1920, 1080), 0, 0, cv::INTER_NEAREST);
		input_img_clone=input_img.clone();
		cuda_preprocess(input_img.ptr(), input_img.cols, input_img.rows, gpu_buffers[0], 640, 384, stream);

		//CUDA_CHECK(cudaMemcpyAsync(cpu_input_buffer, gpu_buffers[0], 3 * 384 * 640 * sizeof(float), cudaMemcpyDeviceToHost, stream));
		//cudaStreamSynchronize(stream);

		context->enqueueV3(stream);
		CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], 5 * 384 * 640 * sizeof(float), cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream);

		cv::Mat seg_mask = cv::Mat::zeros(360, 640, CV_8U);
		for (int h = 12; h < 372; ++h)
		{
			for (int w = 0; w < 640; ++w)
			{
				unsigned char max = 0;
				for (int c = 0; c < 5; c++)
				{
					if (cpu_output_buffer[c * 640 * 384 + h * 640 + w] > cpu_output_buffer[max * 640 * 384 + h * 640 + w])
					{
						max = c;
					}
				}
				auto seg_mask_ptr = seg_mask.ptr<unsigned char>((h - 12), w);
				seg_mask_ptr[0] = max;
			}
		}

		cv::Mat seg_mask_;
		cv::resize(seg_mask, seg_mask_, cv::Size(1920, 1080), 0, 0, cv::INTER_NEAREST);

		cv::Mat lane_seg;
		cv::threshold(seg_mask_, lane_seg, 1, 255, cv::THRESH_BINARY);

		cv::Mat road_mask;
		cv::threshold(seg_mask_, road_mask, 0, 255, cv::THRESH_BINARY);

        cv::Mat write_frame;
        cv::resize(frame, write_frame, cv::Size(1920, 1080));
        cv::Mat result_canvas = write_frame.clone(); 
        processMultiLaneSegmentation(road_mask, lane_seg, result_canvas);

        // 6. 最后叠加显示
        cv::addWeighted(write_frame, 0.5, result_canvas, 0.5, 0, write_frame);

		/////////////////////////////////////////////////////////////////////

        // cv::Mat color_seg = frame.clone();
        // for (int row = 0; row < seg_mask_.rows; row++)
        // {
        //     auto seg_mask__ptr = seg_mask_.ptr(row);
        //     auto color_seg_ptr = color_seg.ptr(row);
        //     for (int col = 0; col < seg_mask_.cols; col++)
        //     {
        //         if (seg_mask__ptr[col] != 0)
        //         {
        //             (color_seg_ptr + (col * 3))[0] = color_list[seg_mask__ptr[col] - 1][0];
        //             (color_seg_ptr + (col * 3))[1] = color_list[seg_mask__ptr[col] - 1][1];
        //             (color_seg_ptr + (col * 3))[2] = color_list[seg_mask__ptr[col] - 1][2];
        //         }
        //     }
        // }

        // cv::addWeighted(frame, 1, color_seg, 0.25, 0, color_seg);

        /////////////////////////////////////////////////////////////////////

        // // 计算变换比例
        // float ratio_x = frame.cols / 1920.0;
        // float ratio_y = frame.rows / 1080.0;
        
        // // 找到掩码中的边界，检测图像中的轮廓
        // std::vector<std::vector<cv::Point>> contours;
        // cv::findContours(road_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // // 对每个找到边界进行拟合
        // std::vector<std::vector<cv::Point>> polygonRegions;
        
        // for (size_t i = 0; i < contours.size(); i++) 
        // {
        //     // 计算轮廓面积
        //     double area = cv::contourArea(contours[i]);
        //     if (area < 5000)
        //     {
        //         // 轮廓面积太小则过滤，可能是噪声
        //         continue;
        //     }
        //     //std::cout << area << std::endl;

        //     std::vector<cv::Point> approx;
        //     // 第三个参数为拟合精度, 越低越贴合分割结果，但边数也越多，特定场景中可能需要调整
        //     cv::approxPolyDP(contours[i], approx, 35, true);

        //     std::vector<cv::Point> transformedPoints;
        //     for (const auto& point : approx) 
        //     {
        //         int x = static_cast<int>(point.x * ratio_x);
        //         int y = static_cast<int>(point.y * ratio_y);
        //         transformedPoints.emplace_back(x, y);
        //     }
            
        //     polygonRegions.push_back(transformedPoints);

        //     // 把多边形画到视频中
        //     for (size_t j = 0; j < transformedPoints.size(); j++) 
        //     {
        //         cv::line(color_seg, transformedPoints[j], transformedPoints[(j + 1) % transformedPoints.size()], cv::Scalar(255), 2);
        //     }
        // }

        // std::vector<std::vector<cv::Point>> lane_polygons =
        // splitLanes(seg_mask_, lane_seg, road_mask);

        // for (auto& poly : lane_polygons)
        // {
        //     for(auto& point: poly)
        //     {
        //         point.x = static_cast<int>(point.x * ratio_x);
        //         point.y = static_cast<int>(point.y * ratio_y);
        //     }
            
        //     for (size_t i = 0; i < poly.size(); i++)
        //     {
        //         cv::line(color_seg, poly[i], poly[(i + 1) % poly.size()], cv::Scalar(0,255,0), 3);
        //     }
        // }
        /////////////////////////////////////////////////////////////////////

        cv::imwrite(result_dir + "/result_" + filename, write_frame);

        // cv::rectangle(frame, cv::Point(ori_minX, ori_minY), cv::Point(ori_maxX, ori_maxY), cv::Scalar(0, 255, 0), 2);
        // output_video_test.write(frame);
		
		
	}

	cudaStreamDestroy(stream);
	context->destroy();
	engine->destroy();
	runtime->destroy();
	cuda_preprocess_destroy();
	free_buffers(&gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);
	return 0;
}

// 进一步优化的多车道直线补全函数
void processMultiLaneSegmentation(cv::Mat& road_mask, cv::Mat& lane_seg, cv::Mat& display_img) {
    int img_w = road_mask.cols;
    int img_h = road_mask.rows;

    // 1. 提取线段
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(lane_seg, lines, 1, CV_PI / 180, 40, 30, 80);

    std::vector<LaneLine> clusters;
    float slope_threshold = 0.15; // 斜率差异阈值
    float x_threshold = img_w * 0.1; // 底部截距差异阈值 (10% 屏幕宽度)

    for (auto& l : lines) {
        float dx = l[2] - l[0];
        float dy = l[3] - l[1];
        float slope = dx / (dy + 1e-6); // 使用 x/y 避免垂直线斜率无穷大
        
        // 过滤掉近乎水平的干扰线
        if (std::abs(slope) > 2.0) continue; 

        // 计算该线段延伸到图像底部时的 x 坐标
        float x_bottom = l[0] + (img_h - l[1]) * slope;

        bool found_cluster = false;
        for (auto& cluster : clusters) {
            if (std::abs(slope - cluster.avg_slope) < slope_threshold &&
                std::abs(x_bottom - cluster.x_at_bottom) < x_threshold) {
                cluster.points.push_back(cv::Point(l[0], l[1]));
                cluster.points.push_back(cv::Point(l[2], l[3]));
                found_cluster = true;
                break;
            }
        }

        if (!found_cluster) {
            clusters.push_back({{cv::Point(l[0], l[1]), cv::Point(l[2], l[3])}, slope, x_bottom});
        }
    }

    // 2. 绘制补全的切割线
    cv::Mat refined_lane_mask = cv::Mat::zeros(road_mask.size(), CV_8UC1);
    for (auto& cluster : clusters) {
        if (cluster.points.size() < 4) continue; // 过滤掉孤立的小线段

        cv::Vec4f line_params;
        cv::fitLine(cluster.points, line_params, cv::DIST_L2, 0, 0.01, 0.01);

        float vx = line_params[0];
        float vy = line_params[1];
        float x0 = line_params[2];
        float y0 = line_params[3];

        // 延伸直线贯穿图像垂直高度
        int x_top = x0 + (0 - y0) * vx / vy;
        int x_bottom = x0 + (img_h - y0) * vx / vy;

        // 绘制粗线用于切割
        cv::line(refined_lane_mask, cv::Point(x_top, 0), cv::Point(x_bottom, img_h), cv::Scalar(255), 10);
    }

    // 3. 切割路面并提取多边形
    cv::Mat individual_lanes;
    cv::subtract(road_mask, refined_lane_mask, individual_lanes);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(individual_lanes, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& cnt : contours) {
        if (cv::contourArea(cnt) < 40000) continue; 

        // 拟合多边形并填充随机颜色
        std::vector<cv::Point> approx;
        cv::approxPolyDP(cnt, approx, 10, true);
        
        cv::Scalar random_color(rand() % 255, rand() % 255, rand() % 255);
        cv::fillPoly(display_img, std::vector<std::vector<cv::Point>>{approx}, random_color);
        
        // 绘制轮廓边缘线
        cv::polylines(display_img, approx, true, cv::Scalar(0, 255, 0), 2);
    }
}

// ---------------------------------------------------------
// 基于车道线切割道路区域，生成独立的单车道多边形
// 输入：
//   seg_mask_: 语义分割结果 (1080x1920)
//   lane_seg: 车道线二值图 (1080x1920)
//   road_mask: 道路区域二值图 (1080x1920)
//
// 输出：
//   laneRegions: 每个元素是一条车道的多边形坐标
// ---------------------------------------------------------
std::vector<std::vector<cv::Point>> splitLanes(
        const cv::Mat& seg_mask_,
        const cv::Mat& lane_seg,
        const cv::Mat& road_mask)
{
    std::vector<std::vector<cv::Point>> laneRegions;

    //===================== 1. 车道线闭操作（补全虚线） =====================//
    cv::Mat lane_closed;
    cv::morphologyEx(lane_seg, lane_closed, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));

    //===================== 2. HoughLinesP 检测车道线段 =====================//
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(lane_closed, lines,
                    1, CV_PI / 180,      // 分辨率
                    40,                  // 最小投票数
                    60,                  // 最小线段长度
                    30);                 // 最大间隔

    // 如果检测不到车道线，则直接返回整块道路
    if (lines.empty())
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(road_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        return contours; // 整块区域
    }

    //===================== 3. 延长车道线到整幅图像 =====================//
    std::vector<cv::Vec4i> fullLines;
    for (auto& l : lines)
    {
        float x1 = l[0], y1 = l[1];
        float x2 = l[2], y2 = l[3];

        if (fabs(x2 - x1) < 1e-6)  // 垂直线特殊处理
        {
            fullLines.emplace_back(x1, 0, x1, 1080);
            continue;
        }

        float k = (y2 - y1) / (x2 - x1);
        float b = y1 - k * x1;

        int top_x    = (0   - b) / k;
        int bottom_x = (1080 - b) / k;

        fullLines.emplace_back(top_x, 0, bottom_x, 1080);
    }

    //===================== 4. 将延长线画成切割 Mask =====================//
    cv::Mat split_mask = cv::Mat::zeros(road_mask.size(), CV_8U);

    for (auto& l : fullLines)
    {
        cv::line(split_mask,
                 cv::Point(l[0], l[1]),
                 cv::Point(l[2], l[3]),
                 cv::Scalar(255), 12); // 线要够粗，确保完全切断
    }

    //===================== 5. 用切割线抠掉道路区域，形成分段 =====================//
    cv::Mat cut_road = road_mask.clone();
    cut_road.setTo(0, split_mask);  // 分割出多个区域

    //===================== 6. 找所有独立的车道块 =====================//
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cut_road, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //===================== 7. 对每个区域做多边形拟合 =====================//
    for (auto& c : contours)
    {
        double area = cv::contourArea(c);
        if (area < 10000) continue;  // 小块噪声过滤

        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 30, true);

        laneRegions.push_back(approx);
    }

    return laneRegions;
}


void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
	assert(engine->getNbBindings() == 2);
	const int inputIndex = engine->getBindingIndex("input");
	const int outputIndex = engine->getBindingIndex("output");
	assert(inputIndex == 0);
	assert(outputIndex == 1);

	// Create GPU buffers on device
	CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, 1 * 3 * 384 * 640 * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, 1 * 5 * 384 * 640 * sizeof(float)));

	// Alloc CPU buffers
	*cpu_output_buffer = new float[1 * 5 * 384 * 640];
}

void free_buffers(float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer)
{
	CUDA_CHECK(cudaFree(*gpu_input_buffer));
	CUDA_CHECK(cudaFree(*gpu_output_buffer));
	delete[] * cpu_output_buffer;
}

