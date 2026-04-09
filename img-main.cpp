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

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
void free_buffers(float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);

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

		// cv::Mat seg_mask_;
		// cv::resize(seg_mask, seg_mask_, cv::Size(1920, 1080), 0, 0, cv::INTER_NEAREST);
        // cv::imwrite(img_dir + "/seg_mask_" + filename, seg_mask_);

		// cv::Mat lane_seg;
		// cv::threshold(seg_mask_, lane_seg, 1, 1, cv::THRESH_BINARY);
        // cv::imwrite(img_dir + "/lane_seg_" + filename, lane_seg);

		// cv::Mat road_mask;
		// cv::threshold(seg_mask_, road_mask, 0, 1, cv::THRESH_BINARY);
        // cv::imwrite(img_dir + "/road_mask_" + filename, road_mask);

        cv::Mat seg_mask_;
        cv::resize(seg_mask, seg_mask_, cv::Size(1920, 1080), 0, 0, cv::INTER_NEAREST);

        // 保存原始 mask（0/1 → 0/255）
        cv::Mat seg_mask_u8;
        seg_mask_.convertTo(seg_mask_u8, CV_8U, 255.0);
        cv::imwrite(img_dir + "/seg_mask_" + filename, seg_mask_u8);

        // lane mask
        cv::Mat lane_seg;
        cv::threshold(seg_mask_, lane_seg, 1, 1, cv::THRESH_BINARY);
        cv::Mat lane_seg_u8;
        lane_seg.convertTo(lane_seg_u8, CV_8U, 255.0);
        cv::imwrite(img_dir + "/lane_seg_" + filename, lane_seg_u8);

        // road mask
        cv::Mat road_mask;
        cv::threshold(seg_mask_, road_mask, 0, 1, cv::THRESH_BINARY);
        cv::Mat road_mask_u8;
        road_mask.convertTo(road_mask_u8, CV_8U, 255.0);
        cv::imwrite(img_dir + "/road_mask_" + filename, road_mask_u8);


		/////////////////////////////////////////////////////////////////////


		
		/////////////////////////////////////////////////////////////////////

		if (1)
		{
			cv::Mat color_seg(seg_mask_.rows, seg_mask_.cols, CV_8UC3, cv::Scalar(0, 0, 0));
			for (int row = 0; row < seg_mask_.rows; row++)
			{
				auto seg_mask__ptr = seg_mask_.ptr(row);
				auto color_seg_ptr = color_seg.ptr(row);
				for (int col = 0; col < seg_mask_.cols; col++)
				{
					if (seg_mask__ptr[col] != 0)
					{
						(color_seg_ptr + (col * 3))[0] = color_list[seg_mask__ptr[col] - 1][0];
						(color_seg_ptr + (col * 3))[1] = color_list[seg_mask__ptr[col] - 1][1];
						(color_seg_ptr + (col * 3))[2] = color_list[seg_mask__ptr[col] - 1][2];
					}
				}
			}

			cv::addWeighted(input_img_clone, 1, color_seg, 0.25, 0, color_seg);
			

			/////////////////////////////////////////////////////////////////////

			// 计算变换比例
			float ratio_x = frame.cols / 1920.0;
			float ratio_y = frame.rows / 1080.0;
			
			// 找到掩码中的边界，检测图像中的轮廓
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(road_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			
			// 对每个找到边界进行拟合
			std::vector<cv::Point> approx;
			for (size_t i = 0; i < contours.size(); i++) 
			{
                // 计算轮廓面积
				double area = cv::contourArea(contours[i]);
				if (area < 5000)
				{
                    // 轮廓面积太小则过滤，可能是噪声
					continue;
				}
				//std::cout << area << std::endl;

				// 第三个参数为拟合精度, 越低越贴合分割结果，但边数也越多，特定场景中可能需要调整
				cv::approxPolyDP(contours[i], approx, 35, true);

				std::vector<cv::Point> transformedPoints;
				for (const auto& point : approx) 
				{
					int x = static_cast<int>(point.x * ratio_x);
					int y = static_cast<int>(point.y * ratio_y);
					transformedPoints.emplace_back(x, y);
				}

				// 把多边形画到视频中
				for (size_t j = 0; j < approx.size(); j++) 
				{
					cv::line(color_seg, approx[j], approx[(j + 1) % approx.size()], cv::Scalar(255), 2);
				}
			}
			/////////////////////////////////////////////////////////////////////

            cv::imwrite("./result/result_" + filename, color_seg);

			// cv::rectangle(frame, cv::Point(ori_minX, ori_minY), cv::Point(ori_maxX, ori_maxY), cv::Scalar(0, 255, 0), 2);
			// output_video_test.write(frame);
		}
		
	}

	cudaStreamDestroy(stream);
	context->destroy();
	engine->destroy();
	runtime->destroy();
	cuda_preprocess_destroy();
	free_buffers(&gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);
	return 0;
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

