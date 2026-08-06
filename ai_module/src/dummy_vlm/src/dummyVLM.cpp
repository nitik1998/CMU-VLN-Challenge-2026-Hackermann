#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <unistd.h>
#include "rclcpp/rclcpp.hpp"

#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int32.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <tf2/transform_datatypes.h>
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using namespace std;

string waypoint_file_dir;
string object_list_file_dir;
double waypointReachDis = 1.0;

vector<float> waypointX, waypointY, waypointHeading;

int objID;
float objMidX, objMidY, objMidZ, objL, objW, objH, objHeading;
string objLabel;

float vehicleX = 0, vehicleY = 0;

string question;

shared_ptr<rclcpp::Node> nh;

rclcpp::Publisher<std_msgs::msg::String>::SharedPtr chainOfThoughtPub;
std_msgs::msg::String chainOfThoughtMsg;

// reading waypoints from file function
void readWaypointFile()
{
  FILE* waypoint_file = fopen(waypoint_file_dir.c_str(), "r");
  if (waypoint_file == NULL) {
    printf ("\nCannot read input files, exit.\n\n");
    exit(1);
  }

  char str[50];
  int val, pointNum;
  string strCur, strLast;
  while (strCur != "end_header") {
    val = fscanf(waypoint_file, "%s", str);
    if (val != 1) {
      printf ("\nError reading input files, exit.\n\n");
      exit(1);
    }

    strLast = strCur;
    strCur = string(str);

    if (strCur == "vertex" && strLast == "element") {
      val = fscanf(waypoint_file, "%d", &pointNum);
      if (val != 1) {
        printf ("\nError reading input files, exit.\n\n");
        exit(1);
      }
    }
  }

  float x, y, heading;
  int val1, val2, val3;
  for (int i = 0; i < pointNum; i++) {
    val1 = fscanf(waypoint_file, "%f", &x);
    val2 = fscanf(waypoint_file, "%f", &y);
    val3 = fscanf(waypoint_file, "%f", &heading);

    if (val1 != 1 || val2 != 1 || val3 != 1) {
      printf ("\nError reading input files, exit.\n\n");
      exit(1);
    }

    waypointX.push_back(x);
    waypointY.push_back(y);
    waypointHeading.push_back(heading);
  }

  fclose(waypoint_file);
}

// reading objects from file function
void readObjectListFile()
{
  FILE* object_list_file = fopen(object_list_file_dir.c_str(), "r");
  if (object_list_file == NULL) {
    printf ("\nCannot read input files, exit.\n\n");
    exit(1);
  }

  char s[100], s2[100];
  int val1, val2, val3, val4, val5, val6, val7, val8, val9;
  val1 = fscanf(object_list_file, "%d", &objID);
  val2 = fscanf(object_list_file, "%f", &objMidX);
  val3 = fscanf(object_list_file, "%f", &objMidY);
  val4 = fscanf(object_list_file, "%f", &objMidZ);
  val5 = fscanf(object_list_file, "%f", &objL);
  val6 = fscanf(object_list_file, "%f", &objW);
  val7 = fscanf(object_list_file, "%f", &objH);
  val8 = fscanf(object_list_file, "%f", &objHeading);
  val9 = fscanf(object_list_file, "%s", s);

  if (val1 != 1 || val2 != 1 || val3 != 1 || val4 != 1 || val5 != 1 || val6 != 1 || val7 != 1 || val8 != 1 || val9 != 1) {
    exit(1);
  }

  while (s[strlen(s) - 1] != '"') {
    val9 = fscanf(object_list_file, "%s", s2);

    if (val9 != 1) break;

    strcat(s, " ");
    strcat(s, s2);
  }

  for (int i = 1; s[i] != '"' && i < 100; i++) objLabel += s[i];

  fclose(object_list_file);
}

void pubPathWaypoints(
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr& waypointPub,
  geometry_msgs::msg::Pose2D& waypointMsgs,
  rclcpp::Rate& rate)
{
  int waypointID = 0;
  int waypointNum = waypointX.size();

  if (waypointNum == 0) {
    printf ("\nNo waypoint available, exit.\n\n");
    exit(1);
  }

  // publish first waypoint
  waypointMsgs.x = waypointX[waypointID];
  waypointMsgs.y = waypointY[waypointID];
  waypointMsgs.theta = waypointHeading[waypointID];
  waypointPub->publish(waypointMsgs);

  bool status = rclcpp::ok();
  while (status) {
    rclcpp::spin_some(nh);

    float disX = vehicleX - waypointX[waypointID];
    float disY = vehicleY - waypointY[waypointID];

    // move to the next waypoint and publish
    if (sqrt(disX * disX + disY * disY) < waypointReachDis) {
      if (waypointID == waypointNum - 1) break;
      waypointID++;

      waypointMsgs.x = waypointX[waypointID];
      waypointMsgs.y = waypointY[waypointID];
      waypointMsgs.theta = waypointHeading[waypointID];
      waypointPub->publish(waypointMsgs);
    }

    status = rclcpp::ok();
    rate.sleep();
  }
}

void pubObjectWaypoint(
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr& waypointPub,
  geometry_msgs::msg::Pose2D& waypointMsgs)
{
  waypointMsgs.x = objMidX;
  waypointMsgs.y = objMidY;
  waypointMsgs.theta = 0;
  waypointPub->publish(waypointMsgs);
}

void pubObjectMarker(
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr& objectMarkerPub,
  visualization_msgs::msg::Marker& objectMarkerMsgs)
{
  objectMarkerMsgs.header.frame_id = "map";
  objectMarkerMsgs.header.stamp = nh->now();
  objectMarkerMsgs.ns = objLabel;
  objectMarkerMsgs.id = objID;
  objectMarkerMsgs.action = visualization_msgs::msg::Marker::ADD;
  objectMarkerMsgs.type = visualization_msgs::msg::Marker::CUBE;
  objectMarkerMsgs.pose.position.x = objMidX;
  objectMarkerMsgs.pose.position.y = objMidY;
  objectMarkerMsgs.pose.position.z = objMidZ;
  tf2::Quaternion quat_tf;
  quat_tf.setRPY(0, 0, objHeading);
  geometry_msgs::msg::Quaternion geoQuat;
  tf2::convert(quat_tf, geoQuat);
  objectMarkerMsgs.pose.orientation = geoQuat;
  objectMarkerMsgs.scale.x = objL;
  objectMarkerMsgs.scale.y = objW;
  objectMarkerMsgs.scale.z = objH;
  objectMarkerMsgs.color.a = 0.5;
  objectMarkerMsgs.color.r = 0;
  objectMarkerMsgs.color.g = 0;
  objectMarkerMsgs.color.b = 1.0;
  objectMarkerPub->publish(objectMarkerMsgs);
}

void delObjectMarker(
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr& objectMarkerPub,
  visualization_msgs::msg::Marker& objectMarkerMsgs)
{
  objectMarkerMsgs.header.frame_id = "map";
  objectMarkerMsgs.header.stamp = nh->now();
  objectMarkerMsgs.ns = objLabel;
  objectMarkerMsgs.id = objID;
  objectMarkerMsgs.action = visualization_msgs::msg::Marker::DELETE;
  objectMarkerMsgs.type = visualization_msgs::msg::Marker::CUBE;
  objectMarkerPub->publish(objectMarkerMsgs);
}

void pubNumericalAnswer(
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr& numericalAnswerPub,
  std_msgs::msg::Int32& numericalResponseMsg,
  int32_t numericalResponse)
{
  numericalResponseMsg.data = numericalResponse;
  numericalAnswerPub->publish(numericalResponseMsg);
}

void pubChainOfThought(
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr& cotPub,
  std_msgs::msg::String& cotMsg,
  const std::string& text)
{
  cotMsg.data = text;
  cotPub->publish(cotMsg);
}

// vehicle pose callback function
void poseHandler(const nav_msgs::msg::Odometry::ConstSharedPtr pose)
{
  vehicleX = pose->pose.pose.position.x;
  vehicleY = pose->pose.pose.position.y;
}

// question callback function
void questionHandler(const std_msgs::msg::String::ConstSharedPtr msg)
{
  RCLCPP_INFO(nh->get_logger(), "Received question");
  pubChainOfThought(chainOfThoughtPub, chainOfThoughtMsg, "Received question: \"" + msg->data + "\"");
  question = msg->data;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  nh = rclcpp::Node::make_shared("dummyVLM");

  nh->declare_parameter<std::string>("waypoint_file_dir", waypoint_file_dir);
  nh->declare_parameter<std::string>("object_list_file_dir", object_list_file_dir);
  nh->declare_parameter<double>("waypointReachDis", waypointReachDis);

  nh->get_parameter("waypoint_file_dir", waypoint_file_dir);
  nh->get_parameter("object_list_file_dir", object_list_file_dir);
  nh->get_parameter("waypointReachDis", waypointReachDis);

  auto subPose = nh->create_subscription<nav_msgs::msg::Odometry>("/state_estimation", 5, poseHandler);
  auto subQuestion = nh->create_subscription<std_msgs::msg::String>("/challenge_question", 5, questionHandler);

  auto waypointPub = nh->create_publisher<geometry_msgs::msg::Pose2D>("/way_point_with_heading", 5);
  geometry_msgs::msg::Pose2D waypointMsgs;

  auto objectMarkerPub = nh->create_publisher<visualization_msgs::msg::Marker>("/selected_object_marker", 5);
  visualization_msgs::msg::Marker objectMarkerMsgs;

  auto numericalAnswerPub = nh->create_publisher<std_msgs::msg::Int32>("/numerical_response", 5);
  std_msgs::msg::Int32 numericalResponseMsg;

  chainOfThoughtPub = nh->create_publisher<std_msgs::msg::String>("/vlm_chain_of_thought", 5);

  // read waypoints from file
  readWaypointFile();

  // read objects from file
  readObjectListFile();

  rclcpp::Rate rate(100);

  RCLCPP_INFO(nh->get_logger(), "Awaiting question...");

  bool status = rclcpp::ok();
  while (status) {
    rclcpp::spin_some(nh);

    if (question.empty()) {
      rate.sleep();
      status = rclcpp::ok();
      continue;
    }

    if (question.rfind("Find", 0) == 0 || question.rfind("find", 0) == 0) {
      RCLCPP_INFO(nh->get_logger(), "Marking and navigating to object.");
      pubChainOfThought(chainOfThoughtPub, chainOfThoughtMsg,
        "Classified as object-reference question. Marking and navigating to object \"" + objLabel + "\".");
      pubObjectMarker(objectMarkerPub, objectMarkerMsgs);
      pubObjectWaypoint(waypointPub, waypointMsgs);
    } else if (question.rfind("How many", 0) == 0 || question.rfind("how many", 0) == 0) {
      delObjectMarker(objectMarkerPub, objectMarkerMsgs);
      int32_t number = (rand() % 10) + 1;
      RCLCPP_INFO(nh->get_logger(), "%d", number);
      pubChainOfThought(chainOfThoughtPub, chainOfThoughtMsg,
        "Classified as numerical question. Answer: " + std::to_string(number));
      pubNumericalAnswer(numericalAnswerPub, numericalResponseMsg, number);
    } else {
      delObjectMarker(objectMarkerPub, objectMarkerMsgs);
      RCLCPP_INFO(nh->get_logger(), "Navigation starts.");
      pubChainOfThought(chainOfThoughtPub, chainOfThoughtMsg,
        "Classified as instruction-following question. Navigation starts.");
      pubPathWaypoints(waypointPub, waypointMsgs, rate);
      RCLCPP_INFO(nh->get_logger(), "Navigation ends.");
      pubChainOfThought(chainOfThoughtPub, chainOfThoughtMsg, "Navigation ends.");
    }

    question.clear();
    RCLCPP_INFO(nh->get_logger(), "Awaiting question...");
    pubChainOfThought(chainOfThoughtPub, chainOfThoughtMsg, "Awaiting question...");
    status = rclcpp::ok();
  }

  return 0;
}
