/// @file  facedetect.cpp
/// @brief facedetect, using OpenCV DNN API (OpenVINO)

#include <iostream>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>

void facedetect(std::string device, double threshold)
{
  // load model
  cv::dnn::Net net = cv::dnn::readNet("face-detection-adas-0001.xml",
                                      "face-detection-adas-0001.bin");

  // input size of Neural Network
  cv::Size inputsize(672, 384);

  // output name
  std::vector<std::string> outNames = net.getLayerNames();

  // select backend
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);

  // select inference device
  if (device == "MYRIAD") {
    net.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
  } else if (device == "CPU") {
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  } else {
    throw std::runtime_error("Unknown device: " + device);
  }

  // capture device
  cv::VideoCapture cap;
  cap.open(0);

  // set capture image size
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 800);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 600);

  cv::Mat frame;

  while (cv::waitKey(16) != 27) {

    // capture 1 frame
    cap >> frame;

    // create a 4D blob from a frame.
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0, inputsize, cv::Scalar(), false, false);

    // set the input to the network
    net.setInput(blob);

    // inference
    cv::Mat out = net.forward(outNames[0]);

    // [1, 1, N, 7] -> [N * 7]
    float* data = (float *)out.data; // .reshape(-1)

    for (int i = 0; i < out.total(); i += 7) {
      float confidence = data[i + 2];

      if (confidence > threshold) {
        int x_min = data[i + 3] * frame.cols;
        int y_min = data[i + 4] * frame.rows;
        int x_max = data[i + 5] * frame.cols;
        int y_max = data[i + 6] * frame.rows;
        cv::rectangle(frame, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(0, 255, 0), 2);
      }
    }
    cv::imshow("facedetect", frame);
  }
}

DEFINE_string(d, "CPU", "device: CPU or MYRIAD");
DEFINE_double(t, 0.9, "threshold");

int main(int argc, char *argv[])
{
  try {

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    facedetect(FLAGS_d, FLAGS_t);

    return EXIT_SUCCESS;
  }

  // -*- error -*- //

  catch (std::bad_alloc &e) {
    std::cerr << "BAD ALLOC Exception : " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  catch (const std::exception& e) {
    std::cerr << "Error: "  << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  catch (...) {
    std::cerr << "unknown exception" << std::endl;
    return EXIT_FAILURE;
  }
}

/// Local Variables: ///
/// truncate-lines:t ///
/// End: ///
