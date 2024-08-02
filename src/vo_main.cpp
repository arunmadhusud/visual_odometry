#include <iostream>
#include <vector>
#include "vo_feature.h"

#define MIN_NUM_FEAT 2000 // Minimum number of features to track

int main(int argc, char** argv) {
    cv::Mat img_1, img_2;
    cv::Mat R_f, t_f; // Final rotation and translation vectors
    
    YAML::Node config = YAML::LoadFile("/home/arun/SFND_Camera/vo/config.yaml");
    std::string root_path = config["image_data"].as<std::string>();
    std::string true_pose = config["true_pose"].as<std::string>();
    std::string gps_data = config["gps_data"].as<std::string>();


    double scale = 1.00;
    char filename1[200];
    char filename2[200];

    sprintf(filename1, "%s/%010d.png", root_path.c_str(), 0);
    sprintf(filename2, "%s/%010d.png", root_path.c_str(), 1);

    char text[100];
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;  
    cv::Point textOrg(10, 50);

    // Read the first two frames from the dataset
    cv::Mat img_1_c = cv::imread(filename1);
    cv::Mat img_2_c = cv::imread(filename2);

    if (!img_1_c.data || !img_2_c.data) { 
        std::cerr << " --(!) Error reading images " << std::endl;
        return -1;
    }

    // Convert to grayscale
    cv::cvtColor(img_1_c, img_1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2_c, img_2, cv::COLOR_BGR2GRAY);

    // Feature detection and tracking
    std::vector<cv::Point2f> points1, points2;
    featureDetection(img_1, points1); // Detect features in img_1
    std::vector<uchar> status;
    featureTracking(img_1, img_2, points1, points2, status); // Track features to img_2

    // Camera intrinsics (from KITTI dataset)
    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157); // Principal point

    // Recovering the pose and the essential matrix
    cv::Mat E, R, t, mask;
    E = cv::findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, points2, points1, R, t, focal, pp, mask);

    cv::Mat prevImage = img_2;
    cv::Mat currImage;
    std::vector<cv::Point2f> prevFeatures = points2;
    std::vector<cv::Point2f> currFeatures;

    char filename[100];

    R_f = R.clone();
    t_f = t.clone();

    clock_t begin = clock();

    cv::namedWindow("Road facing camera", cv::WINDOW_AUTOSIZE); // Create a window for display
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE); // Create a window for displaying trajectory
    cv::namedWindow("Feature points", cv::WINDOW_AUTOSIZE); // Create a window for displaying feature points

    cv::Mat traj = cv::Mat::zeros(1000, 1000, CV_8UC3); // Create an image for trajectory visualization

    int numFrame = 2;
    while (true) { 
        sprintf(filename, "%s/%010d.png", root_path.c_str(), numFrame);
        cv::Mat currImage_c = cv::imread(filename);
        if (currImage_c.empty()) {
            // save the trajectory image and break
            cv::imwrite("map.png", traj);
            break;
        }

        cv::cvtColor(currImage_c, currImage, cv::COLOR_BGR2GRAY); // Convert to grayscale
        std::vector<uchar> status;
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status); // Track features

        E = cv::findEssentialMat(currFeatures, prevFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

        scale = getAbsoluteScale(numFrame, gps_data); // Get the scale from the GPS data
        // std::cout << "Scale is " << scale << std::endl;

        if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
            t_f = t_f + scale * (R_f * t);
            R_f = R_f * R;
        } else {
            // Incorrect translation, pose not updated
        }
        
        // Redetect features if the number falls below a threshold
        if (prevFeatures.size() < MIN_NUM_FEAT) {
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        }
        
        cv::Mat img_keypoints = currImage_c.clone();
        // Draw optical flow lines on the image
        for (size_t i = 0; i < prevFeatures.size(); i++) {
            cv::line(img_keypoints, prevFeatures[i], currFeatures[i], cv::Scalar(255, 0, 0), 1);
        }

        prevImage = currImage.clone();
        prevFeatures = currFeatures;

        // Update trajectory
        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 100;
        cv::circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);

        //update true position
        double x_true, y_true, z_true;
        truePose(numFrame, x_true, y_true, z_true, true_pose);
        int x_true_int = int(x_true) + 300;
        int y_true_int = int(z_true) + 100;
        cv::circle(traj, cv::Point(x_true_int, y_true_int), 1, CV_RGB(0, 255, 0), 1.5);

        cv::rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        cv::putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);

        //put a rectangle at bottom right saying that red is estimated trajectory and green is ground truth
        cv::rectangle(traj, cv::Point(700, 900), cv::Point(900, 950), CV_RGB(0, 0, 0), cv::FILLED);
        cv::circle(traj, cv::Point(750, 925), 5, CV_RGB(255, 0, 0), 5);
        cv::putText(traj, "Estimated Trajectory", cv::Point(775, 930), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, 8);   
        cv::rectangle(traj, cv::Point(700, 950), cv::Point(900, 1000), CV_RGB(0, 0, 0), cv::FILLED);
        cv::circle(traj, cv::Point(750, 975), 5, CV_RGB(0, 255, 0), 5);
        cv::putText(traj, "Ground Truth", cv::Point(775, 980), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, 8);

        // Display images
        cv::imshow("Road facing camera", currImage_c);
        cv::imshow("Trajectory", traj);
        cv::imshow("Feature points", img_keypoints);

        cv::waitKey(1);
        numFrame++;
    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Total time taken: " << elapsed_secs << "s" << std::endl;

    return 0;
}
