#include <iostream>
#include <vector>
#include "vo_feature.h"
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <map>

#define MIN_NUM_FEAT 2000 // Minimum number of features to track

using namespace DBoW2;
using namespace std;


// Define a strut to store the rotation and translation vectors
struct framePose {
    cv::Mat R;
    cv::Mat t;
};

Eigen::Quaterniond rotationMatrixToQuaternion(const cv::Mat &R)
{
    Eigen::Matrix3d eigen_R;
    cv::cv2eigen(R, eigen_R);
    return Eigen::Quaterniond(eigen_R);
}


void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
    out.resize(plain.rows);

    for (int i = 0; i < plain.rows; i++)
    {
        out[i] = plain.row(i);
    }
}

int main(int argc, char** argv) {

    cv::Mat img_1, img_2;
    cv::Mat R_f, t_f; // Final rotation and translation vectors
    // Map to store frame poses
    std::map<int, framePose> framePoses;
    
    YAML::Node config = YAML::LoadFile("../config.yaml");
    std::string root_path = config["image_data"].as<std::string>();
    std::string true_pose = config["true_pose"].as<std::string>();
    std::string gps_data = config["gps_data"].as<std::string>();
    
    std::ofstream g2o_file("../trajectory.g2o");
    if (!g2o_file.is_open()) {
        std::cerr << "Error: Unable to open file trajectory.g2o" << std::endl;
        // Handle the error as needed
    }
    // else write the first vertex to the file starting from the origin
    else {
        g2o_file << "VERTEX_SE3:QUAT " << "0" << " 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000" << std::endl;
    }
    
    // write the ground truth to the file groundtruth.txt
    std::ofstream groundtruth_file("../groundtruth.txt");
    if (!groundtruth_file.is_open()) {
        std::cerr << "Error: Unable to open file groundtruth.txt" << std::endl;
        // Handle the error as needed
    }
    // else write the first vertex to the file starting from the origin
    else {
        groundtruth_file << "0 0.000000 0.000000 0.000000" << std::endl;
    }

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

    // change R[1][1] to 1
    // R.at<double>(1, 1) = 1.0;

    // Initialize ORB for loop detection
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    OrbVocabulary voc("../small_voc.yml.gz");
    OrbDatabase db(voc, false, 0);

    cv::Mat prevImage = img_2;
    cv::Mat currImage;
    std::vector<cv::Point2f> prevFeatures = points2;
    std::vector<cv::Point2f> currFeatures;

    char filename[100];

    R_f = R.clone();
    t_f = t.clone();
    framePose firstPose;
    firstPose.R = R_f.clone();
    firstPose.t = t_f.clone();

    framePoses[1] = firstPose;

    std::cout << "R: " << R_f << std::endl;
    std::cout << "t: " << t_f << std::endl;

    // write the first vertex to the file starting from the origin
    if (g2o_file.is_open()) {
        Eigen::Quaterniond q = rotationMatrixToQuaternion(R_f);
        g2o_file << "VERTEX_SE3:QUAT " << "1" << " "
                // << t_f.at<double>(0) << " " << "0.000000" << " " << t_f.at<double>(2) << " "
                << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        // add edge between the first two frames
        g2o_file << "EDGE_SE3:QUAT " << "0" << " " << "1" << " "
                // << t_f.at<double>(0) << " " << "0.000000" << " " << t_f.at<double>(2) << " "
                << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
                << "1 0 0 0 0 0 "
                << "1 0 0 0 0 "
                << "1 0 0 0 "
                << "1 0 0 "
                << "1 0 "
                << "1" << std::endl;    
    }

    // store the Rf and tf for the previous frame
    cv::Mat R_prev = R_f.clone();
    cv::Mat t_prev = t_f.clone();

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
        // R.at<double>(1, 1) = 1.0;

        double gps_distance = getAbsoluteScale(numFrame, gps_data); // Get the absolute distance from GPS
        double vo_distance = cv::norm(t); // Get the distance from VO
        scale = gps_distance / vo_distance;
        // std::cout << "Scale is " << scale << std::endl;

        // Update camera pose if the estimated motion is primarily forward
        // Conditions:
        // 1. Scale factor is significant (> 0.1)
        // 2. Forward motion (z-axis) is greater than lateral (x-axis) and vertical (y-axis) motions
        if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
            // R and t are the relative rotation and translation of the current frame w.r.t the previous frame
            // the global tranlsation 
            R_f = R_f * R;
            // the global rotation
            t_f =  scale * (R_f * t) + t_f;

            framePose currPose;
            currPose.R = R_f.clone();
            currPose.t = t_f.clone();
            framePoses[numFrame] = currPose;

            // debugging
            if (numFrame>0){
                std::cout << "R: " << R_f << std::endl;
                std::cout << "t: " << t_f << std::endl;              
            }
        } else {
            // Incorrect translation, pose not updated
        }
        
        // Redetect features if the number falls below a threshold
        if (prevFeatures.size() < MIN_NUM_FEAT) {
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        }


        // ORB loop detection
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        vector<cv::Mat> features;
        orb->detectAndCompute(currImage_c, cv::Mat(), keypoints, descriptors);
        changeStructure(descriptors, features);
        db.add(features);

        QueryResults ret;
        db.query(features, ret, 2);
        if (ret.size() > 0 && (ret[0].Id - ret[1].Id > 20) && ret[1].Score > 0.5) {
            std::cout << "Loop detected! current frame: " << ret[0].Id << " Best match: " << ret[1].Id << " Score: " << ret[1].Score << std::endl;
            std::cout << "difference: " << ret[0].Id - ret[1].Id << std::endl;

            int currFrame = ret[0].Id;
            int prevFrame = currFrame - 1;
            int bestMatchFrame = ret[1].Id;

            try {
                // Calculate relative rotation and translation
                // cv::Mat R_rel = framePoses[bestMatchFrame].R.t() * framePoses[currFrame].R;
                // cv::Mat t_rel = framePoses[bestMatchFrame].R.t() * (framePoses[currFrame].t - framePoses[bestMatchFrame].t);
                cv::Mat R_rel = framePoses[currFrame].R.t() * framePoses[prevFrame].R;
                cv::Mat t_rel = framePoses[currFrame].R.t() * (framePoses[prevFrame].t - framePoses[currFrame].t);

                // Write the edge to the file
                if (g2o_file.is_open()) {
                    Eigen::Quaterniond q_rel = rotationMatrixToQuaternion(R_rel);
                    g2o_file << "EDGE_SE3:QUAT " << bestMatchFrame << " " << prevFrame << " "
                            // << t_rel.at<double>(0) << " " << "0.000000" << " " << t_rel.at<double>(2) << " "
                            << t_rel.at<double>(0) << " " << t_rel.at<double>(1) << " " << t_rel.at<double>(2) << " "
                            << q_rel.x() << " " << q_rel.y() << " " << q_rel.z() << " " << q_rel.w() << " "
                            << "100 0 0 0 0 0 "
                            << "100 0 0 0 0 "
                            << "100 0 0 0 "
                            << "100 0 0 "
                            << "100 0 "
                            << "100" << std::endl;
                }
            } catch (const cv::Exception& e) {
                // Log the error message and skip the problematic frame
                std::cerr << "OpenCV exception at frame " << currFrame << ": " << e.what() << std::endl;
            } catch (const std::exception& e) {
                // Catch any other standard exceptions
                std::cerr << "Standard exception at frame " << currFrame << ": " << e.what() << std::endl;
            }



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

        if (g2o_file.is_open()) {
            // std::cout << "Writing to g2o file" << std::endl;
            Eigen::Quaterniond q = rotationMatrixToQuaternion(R_f);
            g2o_file << "VERTEX_SE3:QUAT " << numFrame << " "
                    // << t_f.at<double>(0) << " " << "0.000000" << " " << t_f.at<double>(2) << " "
                    << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << " "
                    << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
            
            std::cout << "quat" << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            
            // find the relative transformation between the current frame and the previous frame
            cv::Mat R_rel = R_prev.t() * R_f;
            cv::Mat t_rel = R_prev.t() * (t_f - t_prev);

            // write the edge to the file
            Eigen::Quaterniond q_rel = rotationMatrixToQuaternion(R_rel);
            g2o_file << "EDGE_SE3:QUAT " << (numFrame-1) << " " << numFrame << " "
                    // << t_rel.at<double>(0) << " " << "0.000000" << " " << t_rel.at<double>(2) << " "
                    << t_rel.at<double>(0) << " " << t_rel.at<double>(1) << " " << t_rel.at<double>(2) << " "
                    << q_rel.x() << " " << q_rel.y() << " " << q_rel.z() << " " << q_rel.w() << " "
                    << "1 0 0 0 0 0 "
                    << "1 0 0 0 0 "
                    << "1 0 0 0 "
                    << "1 0 0 "
                    << "1 0 "
                    << "1" << std::endl;
            
            // store the Rf and tf for the previous frame
            R_prev = R_f.clone();
            t_prev = t_f.clone();

        }                 

        //update true position
        double x_true, y_true, z_true;
        truePose(numFrame, x_true, y_true, z_true, true_pose);
        // write the ground truth to the file
        if (groundtruth_file.is_open()) {
            groundtruth_file << numFrame << " " << x_true << " " << y_true << " " << z_true << std::endl;
        }

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

        // if (numFrame == 1700) {
        //     g2o_file.close(); // Move this outside the loop, after it finishes.
        //     break;
        // }
    }
    g2o_file.close();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Total time taken: " << elapsed_secs << "s" << std::endl;

    return 0;
}
