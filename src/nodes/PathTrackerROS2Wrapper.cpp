#include "nodes/PathTrackerROS2Wrapper.h"

#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::placeholders;

PathTrackerROS2Wrapper::PathTrackerROS2Wrapper(const std::string& name)
    : Node(name), is_tracking_(false), lookahead_updated_(false) {
    // Parameters
    tracking_loop_rate_         = this->declare_parameter("tracking_loop_rate", 5.0);
    path_validity_check_rate_   = this->declare_parameter("path_validity_check_rate", 1.0);
    simplification_epsilon_     = this->declare_parameter("simplification_epsilon", 0.5);
    lookahead_waypoints_        = this->declare_parameter("lookahead_waypoints", 3);
    pruning_distance_threshold_ = this->declare_parameter("pruning_distance_threshold", 1.0);

    std::string nav2_action_name           = this->declare_parameter("nav2_action_name", "compute_path_to_pose");
    std::string nav2_is_path_valid_service = this->declare_parameter("nav2_is_path_valid_service", "is_path_valid");
    std::string locomotion_action_name     = this->declare_parameter("locomotion_action_name", "in/locomotion/poly");
    std::string pose_conversion_service_name
        = this->declare_parameter("pose_conversion_service_name", "convert_to_controls_pose");

    global_frame_id_    = this->declare_parameter("global_frame_id", "map");
    base_link_frame_id_ = this->declare_parameter("base_link_frame_id", "base_link");

    // TF
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Action Server
    action_server_ = rclcpp_action::create_server<TrackPath>(
        this, "in/locomotion/track_path", std::bind(&PathTrackerROS2Wrapper::handle_goal, this, _1, _2),
        std::bind(&PathTrackerROS2Wrapper::handle_cancel, this, _1),
        std::bind(&PathTrackerROS2Wrapper::handle_accepted, this, _1));

    // Clients
    nav2_action_client_       = rclcpp_action::create_client<ComputePathToPose>(this, nav2_action_name);
    locomotion_action_client_ = rclcpp_action::create_client<Locomotion>(this, locomotion_action_name);
    is_path_valid_client_     = this->create_client<nav2_msgs::srv::IsPathValid>(nav2_is_path_valid_service);
    pose_conversion_client_
        = this->create_client<bb_planner_msgs::srv::GetPoseToControlsFrame>(pose_conversion_service_name);

    // Timers
    tracking_timer_ = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000.0 / tracking_loop_rate_)),
                                              std::bind(&PathTrackerROS2Wrapper::tracking_loop, this));
    tracking_timer_->cancel();
    validity_timer_
        = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000.0 / path_validity_check_rate_)),
                                  std::bind(&PathTrackerROS2Wrapper::validity_loop, this));
    validity_timer_->cancel();
}

PathTrackerROS2Wrapper::~PathTrackerROS2Wrapper() {
}

rclcpp_action::GoalResponse PathTrackerROS2Wrapper::handle_goal(const rclcpp_action::GoalUUID& uuid,
                                                                std::shared_ptr<const TrackPath::Goal> goal) {
    (void)uuid;
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (current_goal_handle_ && current_goal_handle_->is_active()) {
        RCLCPP_INFO(this->get_logger(), "Rejecting new goal; another is currently active.");
        return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse PathTrackerROS2Wrapper::handle_cancel(
    const std::shared_ptr<GoalHandleTrackPath> goal_handle) {
    RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
}

void PathTrackerROS2Wrapper::handle_accepted(const std::shared_ptr<GoalHandleTrackPath> goal_handle) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_goal_handle_ = goal_handle;
    remaining_replans_   = goal_handle->get_goal()->max_replans;
    is_tracking_         = false;

    // Start execution on a separate thread to avoid blocking the executor
    std::thread{std::bind(&PathTrackerROS2Wrapper::execute_track_path, this)}.detach();
}

void PathTrackerROS2Wrapper::execute_track_path() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!current_goal_handle_) return;
    auto goal = current_goal_handle_->get_goal();
    RCLCPP_INFO(this->get_logger(), "Starting TrackPath request. Requesting Nav2 Plan...");
    request_nav2_plan(goal->goal_pose);
}

void PathTrackerROS2Wrapper::request_nav2_plan(const geometry_msgs::msg::PoseStamped& goal_pose) {
    if (!nav2_action_client_->wait_for_action_server(std::chrono::seconds(3))) {
        RCLCPP_ERROR(this->get_logger(), "Nav2 Action server not available after waiting");
        if (current_goal_handle_) {
            auto result    = std::make_shared<TrackPath::Result>();
            result->status = TrackPath::Result::STATUS_FAILURE;
            current_goal_handle_->abort(result);
        }
        return;
    }

    auto goal_msg      = ComputePathToPose::Goal();
    goal_msg.goal      = goal_pose;
    goal_msg.use_start = false;  // Use current robot pose

    auto send_goal_options            = rclcpp_action::Client<ComputePathToPose>::SendGoalOptions();
    send_goal_options.result_callback = std::bind(&PathTrackerROS2Wrapper::on_nav2_plan_received, this, _1);

    nav2_action_client_->async_send_goal(goal_msg, send_goal_options);
}

// Implement RDP
double PathTrackerROS2Wrapper::perpendicular_distance(const geometry_msgs::msg::PoseStamped& pt,
                                                      const geometry_msgs::msg::PoseStamped& line_start,
                                                      const geometry_msgs::msg::PoseStamped& line_end) {
    double dx  = line_end.pose.position.x - line_start.pose.position.x;
    double dy  = line_end.pose.position.y - line_start.pose.position.y;
    double mag = std::sqrt(dx * dx + dy * dy);
    if (mag > 0.0) {
        dx /= mag;
        dy /= mag;
    }
    double pvx   = pt.pose.position.x - line_start.pose.position.x;
    double pvy   = pt.pose.position.y - line_start.pose.position.y;
    double pvdot = pvx * dx + pvy * dy;
    double dsx   = pvdot * dx;
    double dsy   = pvdot * dy;
    double ax    = pvx - dsx;
    double ay    = pvy - dsy;
    return std::sqrt(ax * ax + ay * ay);
}

std::vector<geometry_msgs::msg::PoseStamped> PathTrackerROS2Wrapper::ramer_douglas_peucker(
    const std::vector<geometry_msgs::msg::PoseStamped>& points, double epsilon) {
    if (points.size() < 3) return points;
    double dmax  = 0.0;
    size_t index = 0;
    size_t end   = points.size() - 1;
    for (size_t i = 1; i < end; i++) {
        double d = perpendicular_distance(points[i], points[0], points[end]);
        if (d > dmax) {
            index = i;
            dmax  = d;
        }
    }
    std::vector<geometry_msgs::msg::PoseStamped> ResultList;
    if (dmax > epsilon) {
        std::vector<geometry_msgs::msg::PoseStamped> recResults1 = ramer_douglas_peucker(
            std::vector<geometry_msgs::msg::PoseStamped>(points.begin(), points.begin() + index + 1), epsilon);
        std::vector<geometry_msgs::msg::PoseStamped> recResults2 = ramer_douglas_peucker(
            std::vector<geometry_msgs::msg::PoseStamped>(points.begin() + index, points.end()), epsilon);
        ResultList.assign(recResults1.begin(), recResults1.end() - 1);
        ResultList.insert(ResultList.end(), recResults2.begin(), recResults2.end());
    } else {
        ResultList.push_back(points[0]);
        ResultList.push_back(points[end]);
    }
    return ResultList;
}

void PathTrackerROS2Wrapper::on_nav2_plan_received(
    const rclcpp_action::ClientGoalHandle<ComputePathToPose>::WrappedResult& result) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!current_goal_handle_) return;

    if (result.code != rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_ERROR(this->get_logger(), "Nav2 failed to compute path");
        auto tracker_result    = std::make_shared<TrackPath::Result>();
        tracker_result->status = TrackPath::Result::STATUS_FAILURE;
        current_goal_handle_->abort(tracker_result);
        return;
    }

    auto path = result.result->path.poses;
    if (path.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Nav2 returned empty path");
        auto tracker_result    = std::make_shared<TrackPath::Result>();
        tracker_result->status = TrackPath::Result::STATUS_FAILURE;
        current_goal_handle_->abort(tracker_result);
        return;
    }

    dense_path_ = std::deque<geometry_msgs::msg::PoseStamped>(path.begin(), path.end());

    auto simplified             = ramer_douglas_peucker(path, simplification_epsilon_);
    simplified_path_            = std::deque<geometry_msgs::msg::PoseStamped>(simplified.begin(), simplified.end());
    total_simplified_waypoints_ = simplified_path_.size();

    RCLCPP_INFO(this->get_logger(), "Nav2 Plan received. Dense size: %zu, Simplified size: %d", dense_path_.size(),
                total_simplified_waypoints_);
    is_tracking_       = true;
    lookahead_updated_ = true;
    active_locomotion_goal_.reset();
    tracking_timer_->reset();
    validity_timer_->reset();
}

geometry_msgs::msg::PoseStamped PathTrackerROS2Wrapper::get_current_pose() {
    geometry_msgs::msg::PoseStamped pose;
    try {
        auto tf               = tf_buffer_->lookupTransform(global_frame_id_, base_link_frame_id_, tf2::TimePointZero);
        pose.header.frame_id  = global_frame_id_;
        pose.pose.position.x  = tf.transform.translation.x;
        pose.pose.position.y  = tf.transform.translation.y;
        pose.pose.position.z  = tf.transform.translation.z;
        pose.pose.orientation = tf.transform.rotation;
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "Could not get current pose: %s", ex.what());
    }
    return pose;
}

void PathTrackerROS2Wrapper::prune_paths() {
    auto current_pose = get_current_pose();

    // Prune simplified path
    while (simplified_path_.size() > 1) {
        double dx   = simplified_path_.front().pose.position.x - current_pose.pose.position.x;
        double dy   = simplified_path_.front().pose.position.y - current_pose.pose.position.y;
        double dist = std::sqrt(dx * dx + dy * dy);

        if (dist < pruning_distance_threshold_) {
            simplified_path_.pop_front();
            lookahead_updated_ = true;
        } else {
            break;
        }
    }

    // Prune dense path
    while (dense_path_.size() > 1) {
        double dx   = dense_path_.front().pose.position.x - current_pose.pose.position.x;
        double dy   = dense_path_.front().pose.position.y - current_pose.pose.position.y;
        double dist = std::sqrt(dx * dx + dy * dy);

        if (dist < pruning_distance_threshold_) {
            dense_path_.pop_front();
        } else {
            break;
        }
    }

    // Publish Feedback
    if (current_goal_handle_) {
        auto feedback                 = std::make_shared<TrackPath::Feedback>();
        int passed                    = total_simplified_waypoints_ - simplified_path_.size();
        feedback->percentage_complete = (static_cast<float>(passed) / total_simplified_waypoints_) * 100.0f;
        // RCLCPP_INFO(this->get_logger(), "Tracking progress: %.2f%% complete", feedback->percentage_complete);
        current_goal_handle_->publish_feedback(feedback);
    }
}

void PathTrackerROS2Wrapper::tracking_loop() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!is_tracking_ || !current_goal_handle_) return;

    if (current_goal_handle_->is_canceling()) {
        auto result    = std::make_shared<TrackPath::Result>();
        result->status = TrackPath::Result::STATUS_ABORTED;
        current_goal_handle_->canceled(result);
        if (active_locomotion_goal_) {
            locomotion_action_client_->async_cancel_goal(active_locomotion_goal_);
        }
        is_tracking_ = false;
        tracking_timer_->cancel();
        validity_timer_->cancel();
        return;
    }

    prune_paths();

    if (!lookahead_updated_) {
        return;
    }
    lookahead_updated_ = false;

    // We don't check for empty anymore, pruning guarantees size >= 1 and lookahead_updated controls the flow.

    // Take lookahead window
    bool final_in_window = (simplified_path_.size() <= (size_t)lookahead_waypoints_);
    size_t window_size   = std::min((size_t)lookahead_waypoints_, simplified_path_.size());

    auto goal          = current_goal_handle_->get_goal();
    bool split_request = goal->last_goal_anchor_only && final_in_window && window_size > 1;

    auto request1     = std::make_shared<bb_planner_msgs::srv::GetPoseToControlsFrame::Request>();
    request1->timeout = 0.5;

    if (split_request) {
        for (size_t i = 0; i < window_size - 1; ++i) {
            request1->input_poses.push_back(simplified_path_[i]);
        }
        request1->anchor_frame_name = base_link_frame_id_;
    } else {
        for (size_t i = 0; i < window_size; ++i) {
            request1->input_poses.push_back(simplified_path_[i]);
        }
        if (goal->last_goal_anchor_only) {
            request1->anchor_frame_name = final_in_window ? goal->anchor_frame_name : base_link_frame_id_;
        } else {
            request1->anchor_frame_name = goal->anchor_frame_name;
        }
    }

    if (!pose_conversion_client_->wait_for_service(std::chrono::milliseconds(100))) {
        RCLCPP_WARN(this->get_logger(), "Pose conversion service not ready");
        return;
    }

    auto result_future = pose_conversion_client_->async_send_request(
        request1, [this, goal, split_request,
                   window_size](rclcpp::Client<bb_planner_msgs::srv::GetPoseToControlsFrame>::SharedFuture future1) {
            std::lock_guard<std::mutex> inner_lock(this->state_mutex_);
            if (!this->is_tracking_) return;

            auto response1 = future1.get();
            if (!response1->tf_success || response1->output_poses.empty()) {
                RCLCPP_WARN(this->get_logger(), "Pose conversion failed for request 1");
                return;
            }

            if (split_request) {
                auto request2     = std::make_shared<bb_planner_msgs::srv::GetPoseToControlsFrame::Request>();
                request2->timeout = 0.5;
                request2->input_poses.push_back(this->simplified_path_[window_size - 1]);
                request2->anchor_frame_name = goal->anchor_frame_name;

                this->pose_conversion_client_->async_send_request(
                    request2, [this, goal, response1](
                                  rclcpp::Client<bb_planner_msgs::srv::GetPoseToControlsFrame>::SharedFuture future2) {
                        std::lock_guard<std::mutex> loc_lock(this->state_mutex_);
                        if (!this->is_tracking_) return;
                        auto response2 = future2.get();
                        if (!response2->tf_success || response2->output_poses.empty()) {
                            RCLCPP_WARN(this->get_logger(), "Pose conversion failed for request 2");
                            return;
                        }

                        std::vector<geometry_msgs::msg::PoseStamped> final_poses = response1->output_poses;
                        final_poses.insert(final_poses.end(), response2->output_poses.begin(),
                                           response2->output_poses.end());

                        this->send_locomotion_goal(goal, final_poses);
                    });
            } else {
                this->send_locomotion_goal(goal, response1->output_poses);
            }
        });
}

void PathTrackerROS2Wrapper::send_locomotion_goal(std::shared_ptr<const TrackPath::Goal> goal,
                                                  const std::vector<geometry_msgs::msg::PoseStamped>& poses) {
    auto loc_goal              = Locomotion::Goal();
    loc_goal.move_rel          = false;
    loc_goal.depth_rel         = goal->ignore_depth;
    loc_goal.heading_rel       = false;
    loc_goal.depth_ctrl        = Locomotion::Goal::DEPTH_MODE_DEPTH;
    loc_goal.specified_heading = goal->specified_heading;

    for (const auto& pose : poses) {
        loc_goal.forward_setpoints.push_back(pose.pose.position.x);
        loc_goal.sidemove_setpoints.push_back(pose.pose.position.y);

        if (goal->ignore_depth) {
            loc_goal.depth_setpoints.push_back(0.0);
        } else if (goal->depth_override_value != 0.0) {
            loc_goal.depth_setpoints.push_back(goal->depth_override_value);
        } else {
            loc_goal.depth_setpoints.push_back(pose.pose.position.z);
        }

        double yaw = tf2::getYaw(pose.pose.orientation);
        loc_goal.heading_setpoints.push_back(yaw * 180.0 / M_PI);

        loc_goal.roll_setpoints.push_back(0.0);
        loc_goal.pitch_setpoints.push_back(0.0);
    }

    loc_goal.forward_tolerance   = goal->forward_tolerance;
    loc_goal.sidemove_tolerance  = goal->sidemove_tolerance;
    loc_goal.heading_tolerance   = goal->heading_tolerance;
    loc_goal.depth_tolerance     = goal->depth_tolerance;
    loc_goal.max_correction_time = goal->stabilize_duration;

    const uint64_t seq                  = ++locomotion_seq_;
    auto send_options                   = rclcpp_action::Client<Locomotion>::SendGoalOptions();
    send_options.goal_response_callback
        = [this, seq](const rclcpp_action::ClientGoalHandle<Locomotion>::SharedPtr& handle) {
              std::lock_guard<std::mutex> inner_lock(this->state_mutex_);
              if (seq != this->locomotion_seq_) return;  // Response for a superseded goal
              this->active_locomotion_goal_ = handle;
          };
    send_options.result_callback
        = [this, seq](const rclcpp_action::ClientGoalHandle<Locomotion>::WrappedResult& loc_result) {
              std::lock_guard<std::mutex> loc_lock(this->state_mutex_);
              if (!this->is_tracking_) return;

              if (seq != this->locomotion_seq_) {
                  return;  // Ignore result of old preempted goal
              }

              if (loc_result.code == rclcpp_action::ResultCode::SUCCEEDED) {
                  if (this->simplified_path_.size() <= (size_t)this->lookahead_waypoints_) {
                      RCLCPP_INFO(this->get_logger(), "Path Tracker finished successfully!");
                      auto final_res    = std::make_shared<TrackPath::Result>();
                      final_res->status = TrackPath::Result::STATUS_SUCCESS;
                      this->current_goal_handle_->succeed(final_res);
                      this->is_tracking_ = false;
                      this->tracking_timer_->cancel();
                      this->validity_timer_->cancel();
                  }
              } else if (loc_result.code != rclcpp_action::ResultCode::CANCELED) {
                  RCLCPP_WARN(this->get_logger(), "Locomotion failed unexpectedly.");
                  this->is_tracking_ = false;
                  this->tracking_timer_->cancel();
                  this->validity_timer_->cancel();

                  auto final_res    = std::make_shared<TrackPath::Result>();
                  final_res->status = TrackPath::Result::STATUS_FAILURE;
                  this->current_goal_handle_->abort(final_res);
              }
          };

    this->locomotion_action_client_->async_send_goal(loc_goal, send_options);
}

void PathTrackerROS2Wrapper::validity_loop() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!is_tracking_ || dense_path_.empty() || !current_goal_handle_) return;

    if (!is_path_valid_client_->wait_for_service(std::chrono::milliseconds(100))) {
        return;
    }

    auto request = std::make_shared<nav2_msgs::srv::IsPathValid::Request>();
    nav_msgs::msg::Path path_msg;
    path_msg.header.frame_id = global_frame_id_;
    path_msg.header.stamp    = this->get_clock()->now();
    path_msg.poses.assign(dense_path_.begin(), dense_path_.end());
    request->path = path_msg;

    is_path_valid_client_->async_send_request(
        request, [this](rclcpp::Client<nav2_msgs::srv::IsPathValid>::SharedFuture future) {
            std::lock_guard<std::mutex> inner_lock(this->state_mutex_);
            if (!this->is_tracking_) return;

            auto response = future.get();
            if (!response->is_valid) {
                RCLCPP_WARN(this->get_logger(), "Path became invalid! Dynamic obstacle detected.");
                this->is_tracking_ = false;  // Abort current tracking
                this->tracking_timer_->cancel();
                this->validity_timer_->cancel();

                if (this->active_locomotion_goal_) {
                    this->locomotion_action_client_->async_cancel_goal(this->active_locomotion_goal_);
                }

                if (this->remaining_replans_ > 0) {
                    this->remaining_replans_--;
                    RCLCPP_INFO(this->get_logger(), "Replanning... (%d replans left)", this->remaining_replans_);
                    std::thread{std::bind(&PathTrackerROS2Wrapper::execute_track_path, this)}.detach();
                } else {
                    auto final_res    = std::make_shared<TrackPath::Result>();
                    final_res->status = TrackPath::Result::STATUS_FAILURE;
                    this->current_goal_handle_->abort(final_res);
                }
            }
        });
}
