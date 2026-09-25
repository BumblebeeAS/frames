#pragma once

#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2/utils.hpp>

#include <bb_controls_msgs/action/locomotion.hpp>
#include <bb_controls_msgs/action/track_path.hpp>
#include <bb_planner_msgs/srv/get_pose_to_controls_frame.hpp>
#include <future>
#include <memory>
#include <nav2_msgs/action/compute_path_to_pose.hpp>
#include <nav2_msgs/srv/is_path_valid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <string>
#include <vector>

// All goal state is owned by tick(); server callbacks share its mutually exclusive group and only queue.
class PathTrackerROS2Wrapper : public rclcpp::Node {
public:
    using TrackPath           = bb_controls_msgs::action::TrackPath;
    using GoalHandleTrackPath = rclcpp_action::ServerGoalHandle<TrackPath>;
    using ComputePathToPose   = nav2_msgs::action::ComputePathToPose;
    using Locomotion          = bb_controls_msgs::action::Locomotion;
    using IsPathValid         = nav2_msgs::srv::IsPathValid;
    using GetPose             = bb_planner_msgs::srv::GetPoseToControlsFrame;
    using PoseStamped         = geometry_msgs::msg::PoseStamped;

    explicit PathTrackerROS2Wrapper(const std::string& name);

private:
    enum class State { IDLE, PLANNING, TRACKING, STOPPING };
    enum class AfterStop { TERMINATE, REPLAN };
    enum class LocoEvent { NONE, SUCCEEDED, FAILED };

    // ROS 2 Interfaces
    rclcpp::CallbackGroup::SharedPtr server_cb_group_;
    rclcpp::CallbackGroup::SharedPtr client_cb_group_;
    rclcpp_action::Server<TrackPath>::SharedPtr action_server_;
    rclcpp_action::Client<ComputePathToPose>::SharedPtr nav2_action_client_;
    rclcpp_action::Client<Locomotion>::SharedPtr locomotion_action_client_;
    rclcpp::Client<IsPathValid>::SharedPtr is_path_valid_client_;
    rclcpp::Client<GetPose>::SharedPtr pose_conversion_client_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr tick_timer_;

    // Parameters
    double tick_rate_;
    double validity_period_;
    double simplification_epsilon_;
    int lookahead_waypoints_;
    double pruning_distance_threshold_;
    double search_distance_;
    double progress_radius_;
    double progress_timeout_;
    double replan_backoff_;
    double plan_timeout_;
    double conversion_timeout_;
    double send_timeout_;
    double stop_timeout_;
    int tf_failure_limit_;
    std::string global_frame_id_;
    std::string base_link_frame_id_;

    // Goal
    State state_ = State::IDLE;
    std::shared_ptr<GoalHandleTrackPath> active_;
    std::shared_ptr<GoalHandleTrackPath> pending_;
    int replan_failures_   = 0;
    int tf_failures_       = 0;
    double next_plan_time_ = 0.0;

    // Stopping
    AfterStop after_stop_  = AfterStop::TERMINATE;
    uint8_t stop_status_   = TrackPath::Result::STATUS_FAILURE;
    double stop_deadline_  = 0.0;
    bool loco_cancel_sent_ = false;

    // Plan request
    bool plan_in_flight_   = false;
    double plan_sent_time_ = 0.0;
    std::shared_future<rclcpp_action::ClientGoalHandle<ComputePathToPose>::SharedPtr> plan_send_future_;
    std::shared_future<rclcpp_action::ClientGoalHandle<ComputePathToPose>::WrappedResult> plan_result_future_;

    // Path: dense poses, cumulative arc length, and waypoints as dense indices
    std::vector<PoseStamped> dense_;
    std::vector<double> dense_s_;
    std::vector<size_t> waypoints_;
    size_t cursor_       = 0;  // dense index, monotonic
    size_t next_wp_      = 0;  // first waypoint not yet passed
    bool need_send_      = false;
    bool blocked_beyond_ = false;  // path past the committed span is invalid

    // Window conversion (split requests are sequential)
    int conv_stage_        = 0;  // 0 idle, 1 first request, 2 second request
    double conv_sent_time_ = 0.0;
    std::optional<rclcpp::Client<GetPose>::FutureAndRequestId> conv_future_;
    std::vector<PoseStamped> conv_poses_;
    size_t conv_wp_begin_ = 0;
    size_t conv_wp_end_   = 0;
    bool conv_split_      = false;

    // Locomotion: only the latest send is tracked, superseded goals are preempted by controls
    std::optional<std::shared_future<rclcpp_action::ClientGoalHandle<Locomotion>::SharedPtr>> loco_send_future_;
    double loco_sent_time_ = 0.0;
    rclcpp_action::ClientGoalHandle<Locomotion>::SharedPtr loco_handle_;
    std::optional<std::shared_future<rclcpp_action::ClientGoalHandle<Locomotion>::WrappedResult>> loco_result_future_;
    bool loco_owned_      = false;  // false once the path it was sent for is replaced
    bool sent_final_      = false;
    size_t sent_wp_last_  = 0;
    size_t committed_end_ = 0;  // dense index of the last sent waypoint
    bool has_committed_   = false;

    // Validity requests
    double next_validity_time_ = 0.0;
    std::optional<rclcpp::Client<IsPathValid>::FutureAndRequestId> committed_valid_future_;
    std::optional<rclcpp::Client<IsPathValid>::FutureAndRequestId> remaining_valid_future_;
    double validity_sent_time_ = 0.0;

    // Progress checker
    geometry_msgs::msg::Point progress_anchor_;
    double progress_anchor_time_ = 0.0;

    // Action Server Callbacks
    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID& uuid,
                                            std::shared_ptr<const TrackPath::Goal> goal);
    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleTrackPath> goal_handle);
    void handle_accepted(const std::shared_ptr<GoalHandleTrackPath> goal_handle);

    // Tick
    void tick();
    void start_goal(const std::shared_ptr<GoalHandleTrackPath>& goal_handle);
    void terminate(uint8_t status);
    void begin_stop(AfterStop after, uint8_t status);
    void step_stop();
    void step_track(const PoseStamped& pose);

    // Planning
    void request_plan();
    void poll_plan();
    void on_plan_failed(const char* reason);
    void install_path(const std::vector<PoseStamped>& poses);
    void clear_plan_request();
    void clear_conversion();

    // Windows
    void start_conversion(size_t wp_begin, size_t wp_end);
    bool send_conversion_request(const std::vector<PoseStamped>& poses, const std::string& anchor);
    void poll_conversion();
    void send_locomotion_goal(const std::vector<PoseStamped>& poses, size_t wp_begin, size_t wp_last);
    LocoEvent poll_locomotion();
    void clear_locomotion();

    // Validity
    void request_validity();
    void poll_validity();
    void clear_validity();

    // Helpers
    double now_s() const;  // steady clock seconds
    std::optional<PoseStamped> get_current_pose();
    void advance_cursor(const PoseStamped& pose);
    void reset_progress(const PoseStamped& pose);
    void publish_feedback();
    std::string anchor_frame() const;
    nav_msgs::msg::Path make_path(size_t begin, size_t end) const;
    void ramer_douglas_peucker(size_t begin, size_t end, double epsilon, std::vector<size_t>& out) const;
    static double perpendicular_distance(const PoseStamped& pt, const PoseStamped& line_start,
                                         const PoseStamped& line_end);
    static double planar_distance(const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b);
};
