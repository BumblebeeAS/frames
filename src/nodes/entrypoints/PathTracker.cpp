#include <rclcpp/rclcpp.hpp>
#include "nodes/PathTrackerROS2Wrapper.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PathTrackerROS2Wrapper>("path_tracker_node");
    
    // Lets client responses resolve while tick() runs
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    
    RCLCPP_INFO(node->get_logger(), "Starting PathTrackerNode...");
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}
