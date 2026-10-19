#include <rclcpp/rclcpp.hpp>
#include "nodes/ReplanTrackerROS2Wrapper.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ReplanTrackerROS2Wrapper>("replan_tracker_node");
    
    // Lets client responses resolve while tick() runs
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    
    RCLCPP_INFO(node->get_logger(), "Starting ReplanTrackerNode...");
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}
