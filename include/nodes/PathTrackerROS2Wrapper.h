#pragma once

#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2/utils.hpp>

#include <bb_controls_msgs/action/locomotion.hpp>
#include <bb_controls_msgs/action/track_path.hpp>
#include <bb_planner_msgs/srv/get_pose_to_controls_frame.hpp>
#include <deque>
#include <memory>
#include <mutex>
#include <nav2_msgs/action/compute_path_to_pose.hpp>
#include <nav2_msgs/srv/is_path_valid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <string>
#include <vector>

class PathTrackerROS2Wrapper : public rclcpp::Node {
public:
    using TrackPath           = bb_controls_msgs::action::TrackPath;
    using GoalHandleTrackPath = rclcpp_action::ServerGoalHandle<TrackPath>;
    using ComputePathToPose   = nav2_msgs::action::ComputePathToPose;
    using Locomotion          = bb_controls_msgs::action::Locomotion;

    explicit PathTrackerROS2Wrapper(const std::string& name);
    ~PathTrackerROS2Wrapper() override;

private:
    // ROS 2 Interfaces
    rclcpp_action::Server<TrackPath>::SharedPtr action_server_;
    rclcpp_action::Client<ComputePathToPose>::SharedPtr nav2_action_client_;
    rclcpp_action::Client<Locomotion>::SharedPtr locomotion_action_client_;

    rclcpp::Client<nav2_msgs::srv::IsPathValid>::SharedPtr is_path_valid_client_;
    rclcpp::Client<bb_planner_msgs::srv::GetPoseToControlsFrame>::SharedPtr pose_conversion_client_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::TimerBase::SharedPtr tracking_timer_;
    rclcpp::TimerBase::SharedPtr validity_timer_;

    // Parameters
    double tracking_loop_rate_;
    double path_validity_check_rate_;
    double simplification_epsilon_;
    int lookahead_waypoints_;
    double pruning_distance_threshold_;
    std::string global_frame_id_;
    std::string base_link_frame_id_;

    // State
    std::mutex state_mutex_;
    std::shared_ptr<GoalHandleTrackPath> current_goal_handle_;
    std::deque<geometry_msgs::msg::PoseStamped> dense_path_;
    std::deque<geometry_msgs::msg::PoseStamped> simplified_path_;
    int total_simplified_waypoints_;
    int remaining_replans_;
    bool is_tracking_;
    bool lookahead_updated_;
    rclcpp_action::ClientGoalHandle<Locomotion>::SharedPtr active_locomotion_goal_;
    uint64_t locomotion_seq_ = 0;  // Identifies the latest Locomotion goal sent

    // Action Server Callbacks
    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID& uuid,
                                            std::shared_ptr<const TrackPath::Goal> goal);
    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleTrackPath> goal_handle);
    void handle_accepted(const std::shared_ptr<GoalHandleTrackPath> goal_handle);

    // Core Logic
    void execute_track_path();
    void request_nav2_plan(const geometry_msgs::msg::PoseStamped& goal_pose);
    void on_nav2_plan_received(const rclcpp_action::ClientGoalHandle<ComputePathToPose>::WrappedResult& result);

    void tracking_loop();
    void validity_loop();

    // Helper functions
    void send_locomotion_goal(std::shared_ptr<const TrackPath::Goal> goal, const std::vector<geometry_msgs::msg::PoseStamped>& poses);
    void prune_paths();
    geometry_msgs::msg::PoseStamped get_current_pose();
    std::vector<geometry_msgs::msg::PoseStamped> ramer_douglas_peucker(
        const std::vector<geometry_msgs::msg::PoseStamped>& points, double epsilon);
    double perpendicular_distance(const geometry_msgs::msg::PoseStamped& pt,
                                  const geometry_msgs::msg::PoseStamped& line_start,
                                  const geometry_msgs::msg::PoseStamped& line_end);
};
