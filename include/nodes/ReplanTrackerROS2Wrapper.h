#pragma once

#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2/LinearMath/Transform.hpp>
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

// Sends the whole simplified path to controls and leaves it running until a check fails: the route ahead is
// blocked, the robot is off the route, or a map to odom correction moved it. Then it replans from the current
// pose and sends again. A replan is plan, simplify, convert, drop waypoints passed meanwhile, send.
// All goal state is owned by tick(); server callbacks share its mutually exclusive group and only queue.
class ReplanTrackerROS2Wrapper : public rclcpp::Node {
public:
    using TrackPath           = bb_controls_msgs::action::TrackPath;
    using GoalHandleTrackPath = rclcpp_action::ServerGoalHandle<TrackPath>;
    using ComputePathToPose   = nav2_msgs::action::ComputePathToPose;
    using Locomotion          = bb_controls_msgs::action::Locomotion;
    using IsPathValid         = nav2_msgs::srv::IsPathValid;
    using GetPose             = bb_planner_msgs::srv::GetPoseToControlsFrame;
    using PoseStamped         = geometry_msgs::msg::PoseStamped;

    explicit ReplanTrackerROS2Wrapper(const std::string& name);

private:
    enum class State { IDLE, TRACKING, STOPPING };
    enum class Stage { IDLE, PLANNING, CONVERTING };
    enum class LocoEvent { NONE, SUCCEEDED, FAILED };

    // ROS 2 Interfaces
    rclcpp::CallbackGroup::SharedPtr server_cb_group_;
    rclcpp::CallbackGroup::SharedPtr client_cb_group_;
    rclcpp_action::Server<TrackPath>::SharedPtr action_server_;
    rclcpp_action::Client<ComputePathToPose>::SharedPtr nav2_action_client_;
    rclcpp_action::Client<Locomotion>::SharedPtr locomotion_action_client_;
    rclcpp::Client<IsPathValid>::SharedPtr is_path_valid_client_;
    rclcpp::Client<GetPose>::SharedPtr pose_conversion_client_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr simplified_path_pub_;  // null when disabled

    // GUI click goals, forwarded to our own action server
    rclcpp::Subscription<PoseStamped>::SharedPtr click_goal_sub_;
    rclcpp_action::Client<TrackPath>::SharedPtr click_goal_client_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr tick_timer_;

    // Parameters
    double tick_rate_;
    double check_period_;
    double simplification_epsilon_;
    double pruning_distance_threshold_;
    double search_distance_;
    double stop_distance_;
    double resend_tracking_error_;
    double resend_threshold_;
    double progress_radius_;
    double progress_timeout_;
    double plan_timeout_;
    double conversion_timeout_;
    double send_timeout_;
    double stop_timeout_;
    int tf_failure_limit_;
    bool click_goal_2d_;
    bool click_goal_specified_heading_;
    std::string global_frame_id_;
    std::string base_link_frame_id_;
    std::string odom_frame_id_;

    // Goal
    State state_ = State::IDLE;
    std::shared_ptr<GoalHandleTrackPath> active_;
    std::shared_ptr<GoalHandleTrackPath> pending_;
    int plan_failures_         = 0;
    int tf_failures_           = 0;
    double next_check_time_    = 0.0;
    double next_replan_time_   = 0.0;
    double initial_length_     = -1.0;  // first plan's length, for percentage feedback
    double remaining_length_   = 0.0;
    bool replan_needed_        = false;
    const char* replan_reason_ = "";
    bool holding_              = false;  // stopped short of a blockage until a replan is sent

    // Stopping
    uint8_t stop_status_   = TrackPath::Result::STATUS_FAILURE;
    double stop_deadline_  = 0.0;
    bool loco_cancel_sent_ = false;

    // Replan in flight
    Stage stage_            = Stage::IDLE;
    double stage_sent_time_ = 0.0;
    std::shared_future<rclcpp_action::ClientGoalHandle<ComputePathToPose>::SharedPtr> plan_send_future_;
    std::shared_future<rclcpp_action::ClientGoalHandle<ComputePathToPose>::WrappedResult> plan_result_future_;
    std::vector<PoseStamped> dense_;
    std::vector<double> dense_s_;
    std::vector<size_t> send_wps_;  // dense indices to send, start pose excluded

    // Conversion (split requests are sequential)
    int conv_stage_  = 0;  // 1 first request, 2 second request
    bool conv_split_ = false;
    std::optional<rclcpp::Client<GetPose>::FutureAndRequestId> conv_future_;
    std::vector<PoseStamped> conv_poses_;

    // Route the running trajectory follows: the dense plan it was sent from, in the global frame
    std::vector<PoseStamped> route_;
    std::vector<double> route_s_;
    size_t route_cursor_ = 0;  // dense index nearest the robot, monotonic
    std::optional<tf2::Transform> route_global_to_odom_;

    // Validity of the route ahead, checked in parallel as the part within stop_distance and the rest
    std::optional<rclcpp::Client<IsPathValid>::FutureAndRequestId> near_valid_future_;
    std::optional<rclcpp::Client<IsPathValid>::FutureAndRequestId> far_valid_future_;
    double valid_sent_time_ = 0.0;

    // Locomotion: only the latest send is tracked, superseded goals are preempted by controls
    std::optional<std::shared_future<rclcpp_action::ClientGoalHandle<Locomotion>::SharedPtr>> loco_send_future_;
    double loco_sent_time_ = 0.0;
    rclcpp_action::ClientGoalHandle<Locomotion>::SharedPtr loco_handle_;
    std::optional<std::shared_future<rclcpp_action::ClientGoalHandle<Locomotion>::WrappedResult>> loco_result_future_;
    bool loco_owned_ = false;  // false once replaced by another goal or canceled to hold

    // Progress checker
    geometry_msgs::msg::Point progress_anchor_;
    double progress_anchor_time_ = -1.0;

    // Action Server Callbacks
    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID& uuid,
                                            std::shared_ptr<const TrackPath::Goal> goal);
    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleTrackPath> goal_handle);
    void handle_accepted(const std::shared_ptr<GoalHandleTrackPath> goal_handle);
    void on_click_goal(const PoseStamped::SharedPtr msg);

    // Tick
    void tick();
    void start_goal(const std::shared_ptr<GoalHandleTrackPath>& goal_handle);
    void terminate(uint8_t status);
    void begin_stop(uint8_t status);
    void step_stop();
    bool check_progress(const PoseStamped& pose);

    // Checks
    void run_checks(const PoseStamped& pose);
    void poll_validity();
    void send_validity_request(size_t begin, size_t end,
                               std::optional<rclcpp::Client<IsPathValid>::FutureAndRequestId>& future);
    void request_replan(const char* reason);
    void hold();
    std::optional<tf2::Transform> lookup_global_to_odom() const;

    // Replan
    void request_plan();
    void poll_plan();
    void on_plan_failed(const char* reason);
    void on_plan(const std::vector<PoseStamped>& poses);
    void start_conversion();
    bool send_conversion_request(const std::vector<PoseStamped>& poses, const std::string& anchor);
    void poll_conversion(const PoseStamped& pose);
    void clear_cycle();
    void clear_validity();

    // Locomotion
    bool send_locomotion_goal(const std::vector<PoseStamped>& poses);
    LocoEvent poll_locomotion();
    void clear_locomotion();

    // Helpers
    double now_s() const;  // steady clock seconds
    std::optional<PoseStamped> get_current_pose();
    void publish_feedback();
    std::string anchor_frame() const;
    void ramer_douglas_peucker(size_t begin, size_t end, double epsilon, std::vector<size_t>& out) const;
    static size_t nearest_index(const std::vector<PoseStamped>& path, const std::vector<double>& s, size_t from,
                                double search_distance, const geometry_msgs::msg::Point& p, double* distance = nullptr);
    static double perpendicular_distance(const PoseStamped& pt, const PoseStamped& line_start,
                                         const PoseStamped& line_end);
    static double planar_distance(const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b);
};
