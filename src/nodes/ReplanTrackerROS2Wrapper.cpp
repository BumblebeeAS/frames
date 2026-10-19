#include "nodes/ReplanTrackerROS2Wrapper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::placeholders;

namespace {
template <typename F>
bool is_ready(const F& f) {
    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}
}  // namespace

ReplanTrackerROS2Wrapper::ReplanTrackerROS2Wrapper(const std::string& name) : Node(name) {
    // Parameters
    tick_rate_                  = this->declare_parameter("tracking_loop_rate", 10.0);
    check_period_               = 1.0 / this->declare_parameter("check_rate", 1.0);
    simplification_epsilon_     = this->declare_parameter("simplification_epsilon", 0.2);
    pruning_distance_threshold_ = this->declare_parameter("pruning_distance_threshold", 0.5);
    search_distance_            = this->declare_parameter("search_distance", 3.0);
    stop_distance_              = this->declare_parameter("stop_distance", 5.0);
    resend_tracking_error_      = this->declare_parameter("resend_tracking_error", 1.0);
    resend_threshold_           = this->declare_parameter("resend_threshold", 0.3);
    progress_radius_            = this->declare_parameter("progress_radius", 0.5);
    progress_timeout_           = this->declare_parameter("progress_timeout", 30.0);
    plan_timeout_               = this->declare_parameter("plan_timeout", 5.0);
    conversion_timeout_         = this->declare_parameter("conversion_timeout", 10.0);
    send_timeout_               = this->declare_parameter("send_timeout", 2.0);
    stop_timeout_               = this->declare_parameter("stop_timeout", 3.0);
    tf_failure_limit_           = this->declare_parameter("tf_failure_limit", 10);

    const bool click_goal_enable       = this->declare_parameter("click_goal_enable", false);
    const std::string click_goal_topic = this->declare_parameter("click_goal_topic", "click_goal");
    click_goal_2d_                     = this->declare_parameter("click_goal_2d", true);
    click_goal_specified_heading_      = this->declare_parameter("click_goal_specified_heading", true);

    const bool simplified_path_viz_enable = this->declare_parameter("simplified_path_viz_enable", false);
    const std::string simplified_path_viz_topic
        = this->declare_parameter("simplified_path_viz_topic", "simplified_path");

    std::string nav2_action_name           = this->declare_parameter("nav2_action_name", "compute_path_to_pose");
    std::string nav2_is_path_valid_service = this->declare_parameter("nav2_is_path_valid_service", "is_path_valid");
    std::string locomotion_action_name     = this->declare_parameter("locomotion_action_name", "in/locomotion/poly");
    std::string pose_conversion_service_name
        = this->declare_parameter("pose_conversion_service_name", "convert_to_controls_pose");

    global_frame_id_    = this->declare_parameter("global_frame_id", "map");
    base_link_frame_id_ = this->declare_parameter("base_link_frame_id", "base_link");
    odom_frame_id_      = this->declare_parameter("odom_frame_id", "odom");

    // TF
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Server callbacks and tick never run concurrently; client responses resolve futures on their own group
    server_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    client_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // Action Server
    action_server_ = rclcpp_action::create_server<TrackPath>(
        this, "in/locomotion/track_path", std::bind(&ReplanTrackerROS2Wrapper::handle_goal, this, _1, _2),
        std::bind(&ReplanTrackerROS2Wrapper::handle_cancel, this, _1),
        std::bind(&ReplanTrackerROS2Wrapper::handle_accepted, this, _1), rcl_action_server_get_default_options(),
        server_cb_group_);

    // Clients
    nav2_action_client_ = rclcpp_action::create_client<ComputePathToPose>(this, nav2_action_name, client_cb_group_);
    locomotion_action_client_
        = rclcpp_action::create_client<Locomotion>(this, locomotion_action_name, client_cb_group_);
    is_path_valid_client_
        = this->create_client<IsPathValid>(nav2_is_path_valid_service, rclcpp::ServicesQoS(), client_cb_group_);
    pose_conversion_client_
        = this->create_client<GetPose>(pose_conversion_service_name, rclcpp::ServicesQoS(), client_cb_group_);

    // Latched so a viewer that connects later still gets the current plan
    if (simplified_path_viz_enable) {
        simplified_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(simplified_path_viz_topic,
                                                                           rclcpp::QoS(1).transient_local());
    }

    // Clicks go through the action server so they behave exactly like an action call, preemption included
    if (click_goal_enable) {
        click_goal_client_ = rclcpp_action::create_client<TrackPath>(this, "in/locomotion/track_path", client_cb_group_);
        rclcpp::SubscriptionOptions click_options;
        click_options.callback_group = client_cb_group_;
        click_goal_sub_              = this->create_subscription<PoseStamped>(
            click_goal_topic, 1, std::bind(&ReplanTrackerROS2Wrapper::on_click_goal, this, _1), click_options);
        RCLCPP_INFO(this->get_logger(), "Accepting click goals on %s (%s)", click_goal_sub_->get_topic_name(),
                    click_goal_2d_ ? "2D" : "3D");
    }

    tick_timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / tick_rate_),
                                          std::bind(&ReplanTrackerROS2Wrapper::tick, this), server_cb_group_);
}

double ReplanTrackerROS2Wrapper::now_s() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

rclcpp_action::GoalResponse ReplanTrackerROS2Wrapper::handle_goal(const rclcpp_action::GoalUUID& uuid,
                                                                  std::shared_ptr<const TrackPath::Goal> goal) {
    (void)uuid;
    (void)goal;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse ReplanTrackerROS2Wrapper::handle_cancel(
    const std::shared_ptr<GoalHandleTrackPath> goal_handle) {
    (void)goal_handle;
    RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
    return rclcpp_action::CancelResponse::ACCEPT;
}

void ReplanTrackerROS2Wrapper::handle_accepted(const std::shared_ptr<GoalHandleTrackPath> goal_handle) {
    // Only queue; tick() adopts it
    if (pending_) {
        auto result    = std::make_shared<TrackPath::Result>();
        result->status = TrackPath::Result::STATUS_ABORTED;
        pending_->abort(result);
    }
    pending_ = goal_handle;
}

void ReplanTrackerROS2Wrapper::on_click_goal(const PoseStamped::SharedPtr msg) {
    if (!click_goal_client_->action_server_is_ready()) {
        RCLCPP_WARN(this->get_logger(), "Click goal dropped, TrackPath server not ready");
        return;
    }

    TrackPath::Goal goal;
    goal.goal_pose              = *msg;
    goal.goal_pose.header.stamp = builtin_interfaces::msg::Time();  // latest TF, so replans never look up a stale stamp
    goal.specified_heading      = click_goal_specified_heading_;
    if (click_goal_2d_) {
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, tf2::getYaw(msg->pose.orientation));
        goal.goal_pose.pose.orientation = tf2::toMsg(q);
        goal.goal_pose.pose.position.z  = 0.0;
        goal.ignore_depth               = true;
    } else {
        goal.ignore_depth = false;
    }

    RCLCPP_INFO(this->get_logger(), "Click goal: (%.2f, %.2f, %.2f) yaw %.1f deg in %s%s",
                goal.goal_pose.pose.position.x, goal.goal_pose.pose.position.y, goal.goal_pose.pose.position.z,
                tf2::getYaw(goal.goal_pose.pose.orientation) * 180.0 / M_PI, msg->header.frame_id.c_str(),
                goal.specified_heading ? "" : ", heading unspecified");
    click_goal_client_->async_send_goal(goal);
}

void ReplanTrackerROS2Wrapper::tick() {
    if (pending_ && pending_->is_canceling()) {
        auto result    = std::make_shared<TrackPath::Result>();
        result->status = TrackPath::Result::STATUS_ABORTED;
        pending_->canceled(result);
        pending_.reset();
    }

    if (state_ == State::IDLE) {
        if (pending_) {
            auto goal_handle = std::move(pending_);
            start_goal(goal_handle);
        }
        return;
    }

    if (state_ == State::STOPPING) {
        step_stop();
        return;
    }

    if (active_->is_canceling()) {
        RCLCPP_INFO(this->get_logger(), "Goal canceled, stopping");
        begin_stop(TrackPath::Result::STATUS_ABORTED);
        return;
    }

    // Replace without stopping; the running trajectory continues until the new goal's first send preempts it
    if (pending_) {
        RCLCPP_INFO(this->get_logger(), "New goal received, preempting active goal");
        auto result    = std::make_shared<TrackPath::Result>();
        result->status = TrackPath::Result::STATUS_ABORTED;
        active_->abort(result);
        auto goal_handle = std::move(pending_);
        start_goal(goal_handle);
        return;
    }

    auto pose = get_current_pose();
    if (!pose) {
        if (++tf_failures_ >= tf_failure_limit_) {
            RCLCPP_ERROR(this->get_logger(), "No robot pose for %d ticks, aborting", tf_failures_);
            begin_stop(TrackPath::Result::STATUS_FAILURE);
        }
        return;
    }
    tf_failures_ = 0;
    if (!check_progress(*pose)) return;

    // Every send ends at the goal with its tolerances, so any success means the goal is reached
    switch (poll_locomotion()) {
    case LocoEvent::SUCCEEDED:
        RCLCPP_INFO(this->get_logger(), "Path Tracker finished successfully!");
        terminate(TrackPath::Result::STATUS_SUCCESS);
        return;
    case LocoEvent::FAILED:
        begin_stop(TrackPath::Result::STATUS_FAILURE);
        return;
    case LocoEvent::NONE:
        break;
    }

    poll_validity();
    if (!replan_needed_ && stage_ == Stage::IDLE && !near_valid_future_ && !far_valid_future_ && !route_.empty()
        && now_s() >= next_check_time_) {
        run_checks(*pose);
    }
    if (replan_needed_ && stage_ == Stage::IDLE && now_s() >= next_replan_time_) request_plan();
    poll_plan();
    if (state_ != State::TRACKING) return;
    poll_conversion(*pose);
    publish_feedback();
}

void ReplanTrackerROS2Wrapper::start_goal(const std::shared_ptr<GoalHandleTrackPath>& goal_handle) {
    active_               = goal_handle;
    plan_failures_        = 0;
    tf_failures_          = 0;
    next_check_time_      = 0.0;
    next_replan_time_     = 0.0;
    initial_length_       = -1.0;
    remaining_length_     = 0.0;
    holding_              = false;
    progress_anchor_time_ = -1.0;
    loco_owned_           = false;
    route_.clear();
    route_s_.clear();
    route_global_to_odom_.reset();
    clear_cycle();
    clear_validity();

    RCLCPP_INFO(this->get_logger(), "Starting TrackPath goal");
    state_ = State::TRACKING;
    request_replan("first send");
}

void ReplanTrackerROS2Wrapper::terminate(uint8_t status) {
    auto result    = std::make_shared<TrackPath::Result>();
    result->status = status;
    if (active_->is_canceling()) {
        active_->canceled(result);
    } else if (status == TrackPath::Result::STATUS_SUCCESS) {
        active_->succeed(result);
    } else {
        active_->abort(result);
    }
    active_.reset();
    clear_cycle();
    clear_validity();
    route_.clear();
    replan_needed_ = false;
    holding_       = false;
    state_         = State::IDLE;
}

void ReplanTrackerROS2Wrapper::begin_stop(uint8_t status) {
    state_         = State::STOPPING;
    stop_status_   = status;
    stop_deadline_ = now_s() + stop_timeout_;
    clear_cycle();
    clear_validity();
    step_stop();
}

void ReplanTrackerROS2Wrapper::step_stop() {
    // Wait for acceptance of an in-flight send so it can be canceled, then for its result
    if (loco_send_future_ && is_ready(*loco_send_future_)) {
        loco_handle_ = loco_send_future_->get();
        loco_send_future_.reset();
        if (loco_handle_) {
            try {
                loco_result_future_ = locomotion_action_client_->async_get_result(loco_handle_);
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "Could not get locomotion result: %s", e.what());
            }
        }
    }
    if (loco_handle_ && !loco_cancel_sent_) {
        try {
            locomotion_action_client_->async_cancel_goal(loco_handle_);
        } catch (const std::exception& e) {
            RCLCPP_DEBUG(this->get_logger(), "Locomotion cancel skipped: %s", e.what());
        }
        loco_cancel_sent_ = true;
    }

    const bool confirmed
        = !loco_send_future_ && (!loco_handle_ || (loco_result_future_ && is_ready(*loco_result_future_)));
    if (!confirmed) {
        if (now_s() < stop_deadline_) return;
        RCLCPP_ERROR(this->get_logger(), "Locomotion stop not confirmed within %.1f s", stop_timeout_);
    }
    clear_locomotion();
    terminate(stop_status_);
}

bool ReplanTrackerROS2Wrapper::check_progress(const PoseStamped& pose) {
    if (progress_anchor_time_ < 0.0 || planar_distance(progress_anchor_, pose.pose.position) > progress_radius_) {
        progress_anchor_      = pose.pose.position;
        progress_anchor_time_ = now_s();
        return true;
    }
    if (now_s() - progress_anchor_time_ <= progress_timeout_) return true;
    RCLCPP_ERROR(this->get_logger(), "No progress for %.1f s, aborting", progress_timeout_);
    begin_stop(TrackPath::Result::STATUS_FAILURE);
    return false;
}

// Checks

void ReplanTrackerROS2Wrapper::run_checks(const PoseStamped& pose) {
    next_check_time_ = now_s() + check_period_;

    double off_route = 0.0;
    route_cursor_    = nearest_index(route_, route_s_, route_cursor_, search_distance_, pose.pose.position, &off_route);
    remaining_length_ = route_s_.back() - route_s_[route_cursor_];

    if (off_route > resend_tracking_error_) {
        RCLCPP_WARN(this->get_logger(), "%.2f m off the route, replanning", off_route);
        request_replan("off the route");
        return;
    }

    // The route lives in the global frame, so only a changed global to odom transform moves its setpoints
    const auto global_to_odom = lookup_global_to_odom();
    if (global_to_odom && route_global_to_odom_) {
        for (size_t i = route_cursor_; i < route_.size(); ++i) {
            const auto& p = route_[i].pose.position;
            const tf2::Vector3 v(p.x, p.y, 0.0);
            const double shift = ((*global_to_odom) * v - (*route_global_to_odom_) * v).length();
            if (shift > resend_threshold_) {
                RCLCPP_WARN(this->get_logger(), "Map to odom correction moved the route %.2f m, replanning", shift);
                request_replan("map to odom moved");
                return;
            }
        }
    }

    // The service only says whether a path is blocked, not where, so the near part is checked on its own
    if (!is_path_valid_client_->service_is_ready()) return;
    size_t split = route_cursor_;
    while (split + 1 < route_.size() && route_s_[split + 1] - route_s_[route_cursor_] <= stop_distance_) ++split;
    valid_sent_time_ = now_s();
    send_validity_request(route_cursor_, split + 1, near_valid_future_);
    if (split + 1 < route_.size()) send_validity_request(split, route_.size(), far_valid_future_);
}

void ReplanTrackerROS2Wrapper::send_validity_request(
    size_t begin, size_t end, std::optional<rclcpp::Client<IsPathValid>::FutureAndRequestId>& future) {
    auto request                  = std::make_shared<IsPathValid::Request>();
    request->path.header.frame_id = global_frame_id_;
    request->path.header.stamp    = this->get_clock()->now();
    request->path.poses.assign(route_.begin() + static_cast<std::ptrdiff_t>(begin),
                               route_.begin() + static_cast<std::ptrdiff_t>(end));
    future.emplace(is_path_valid_client_->async_send_request(request));
}

void ReplanTrackerROS2Wrapper::poll_validity() {
    if (!near_valid_future_ && !far_valid_future_) return;
    const bool ready = (!near_valid_future_ || is_ready(*near_valid_future_))
                       && (!far_valid_future_ || is_ready(*far_valid_future_));
    if (!ready) {
        if (now_s() - valid_sent_time_ > check_period_) clear_validity();
        return;
    }
    const bool near_ok = !near_valid_future_ || near_valid_future_->get()->is_valid;
    const bool far_ok  = !far_valid_future_ || far_valid_future_->get()->is_valid;
    clear_validity();
    if (replan_needed_ || route_.empty()) return;

    if (!near_ok) {
        RCLCPP_WARN(this->get_logger(), "Route blocked within %.1f m, stopping to replan", stop_distance_);
        hold();
        request_replan("route blocked");
    } else if (!far_ok) {
        RCLCPP_WARN(this->get_logger(), "Route blocked beyond %.1f m, replanning while moving", stop_distance_);
        request_replan("route blocked");
    }
}

void ReplanTrackerROS2Wrapper::request_replan(const char* reason) {
    replan_needed_ = true;
    replan_reason_ = reason;
}

// Cancel the running trajectory so controls holds position; its result is ignored
void ReplanTrackerROS2Wrapper::hold() {
    holding_ = true;
    if (loco_handle_ && !loco_cancel_sent_) {
        try {
            locomotion_action_client_->async_cancel_goal(loco_handle_);
        } catch (const std::exception& e) {
            RCLCPP_DEBUG(this->get_logger(), "Locomotion cancel skipped: %s", e.what());
        }
        loco_cancel_sent_ = true;
    }
    loco_owned_ = false;  // an unacknowledged send is canceled once accepted, see poll_locomotion
}

std::optional<tf2::Transform> ReplanTrackerROS2Wrapper::lookup_global_to_odom() const {
    try {
        tf2::Transform t;
        tf2::fromMsg(tf_buffer_->lookupTransform(odom_frame_id_, global_frame_id_, tf2::TimePointZero).transform, t);
        return t;
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "No %s to %s transform: %s",
                             global_frame_id_.c_str(), odom_frame_id_.c_str(), ex.what());
        return std::nullopt;
    }
}

// Replan

void ReplanTrackerROS2Wrapper::request_plan() {
    next_replan_time_ = now_s() + check_period_;  // retry pace after a failure
    if (!nav2_action_client_->action_server_is_ready()) {
        on_plan_failed("planner server unavailable");
        return;
    }
    ComputePathToPose::Goal goal;
    goal.goal           = active_->get_goal()->goal_pose;
    goal.use_start      = false;  // Use current robot pose
    plan_send_future_   = nav2_action_client_->async_send_goal(goal);
    plan_result_future_ = {};
    stage_              = Stage::PLANNING;
    stage_sent_time_    = now_s();
}

void ReplanTrackerROS2Wrapper::poll_plan() {
    if (stage_ != Stage::PLANNING) return;

    if (now_s() - stage_sent_time_ > plan_timeout_) {
        if (is_ready(plan_send_future_) && plan_send_future_.get()) {
            try {
                nav2_action_client_->async_cancel_goal(plan_send_future_.get());
            } catch (const std::exception&) {
            }
        }
        on_plan_failed("timed out");
        return;
    }

    if (!plan_result_future_.valid()) {
        if (!is_ready(plan_send_future_)) return;
        auto handle = plan_send_future_.get();
        if (!handle) {
            on_plan_failed("rejected");
            return;
        }
        try {
            plan_result_future_ = nav2_action_client_->async_get_result(handle);
        } catch (const std::exception& e) {
            on_plan_failed(e.what());
            return;
        }
    }
    if (!is_ready(plan_result_future_)) return;

    auto wrapped = plan_result_future_.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED || !wrapped.result || wrapped.result->path.poses.empty()) {
        on_plan_failed(wrapped.result && !wrapped.result->error_msg.empty() ? wrapped.result->error_msg.c_str()
                                                                            : "no path");
        return;
    }
    if (!wrapped.result->path.header.frame_id.empty() && wrapped.result->path.header.frame_id != global_frame_id_) {
        on_plan_failed("path frame differs from global_frame_id");
        return;
    }
    // Theta* leaves pose headers empty; give them the path header as Smac does, since the conversion service
    // transforms each pose by its own frame and pairs it with odometry by its stamp
    std::vector<PoseStamped> poses = wrapped.result->path.poses;
    for (auto& pose : poses) {
        if (pose.header.frame_id.empty()) pose.header = wrapped.result->path.header;
    }
    on_plan(poses);
}

// The running trajectory, or the hold, continues; the replan is retried after check_period
void ReplanTrackerROS2Wrapper::on_plan_failed(const char* reason) {
    clear_cycle();
    const int max_replans = active_->get_goal()->max_replans;
    if (++plan_failures_ > max_replans) {
        RCLCPP_ERROR(this->get_logger(), "Planning failed (%s), %d consecutive failures, aborting", reason,
                     plan_failures_);
        begin_stop(TrackPath::Result::STATUS_FAILURE);
        return;
    }
    RCLCPP_WARN(this->get_logger(), "Planning failed (%s), retry %d/%d", reason, plan_failures_, max_replans);
}

void ReplanTrackerROS2Wrapper::on_plan(const std::vector<PoseStamped>& poses) {
    plan_failures_ = 0;
    dense_         = poses;
    dense_s_.assign(dense_.size(), 0.0);
    for (size_t i = 1; i < dense_.size(); ++i) {
        dense_s_[i] = dense_s_[i - 1] + planar_distance(dense_[i - 1].pose.position, dense_[i].pose.position);
    }
    std::vector<size_t> waypoints = {0};
    if (dense_.size() > 1) ramer_douglas_peucker(0, dense_.size() - 1, simplification_epsilon_, waypoints);

    if (simplified_path_pub_) {
        nav_msgs::msg::Path path;
        path.header.frame_id = global_frame_id_;
        path.header.stamp    = this->get_clock()->now();
        for (const size_t idx : waypoints) path.poses.push_back(dense_[idx]);
        simplified_path_pub_->publish(path);
    }

    // The path starts at the robot and controls starts from its measured pose, so the start is not sent
    send_wps_.assign(waypoints.begin() + (waypoints.size() > 1 ? 1 : 0), waypoints.end());
    start_conversion();
}

void ReplanTrackerROS2Wrapper::start_conversion() {
    // Every send ends at the goal, so the goal anchor applies to its last pose only when requested
    const auto goal = active_->get_goal();
    conv_split_     = goal->last_goal_anchor_only && send_wps_.size() > 1;
    conv_poses_.clear();

    std::vector<PoseStamped> poses;
    for (size_t i = 0; i < send_wps_.size() - (conv_split_ ? 1 : 0); ++i) poses.push_back(dense_[send_wps_[i]]);
    if (!send_conversion_request(poses, conv_split_ ? base_link_frame_id_ : anchor_frame())) {
        clear_cycle();
        return;
    }
    stage_      = Stage::CONVERTING;
    conv_stage_ = 1;
}

bool ReplanTrackerROS2Wrapper::send_conversion_request(const std::vector<PoseStamped>& poses,
                                                       const std::string& anchor) {
    if (!pose_conversion_client_->service_is_ready()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Pose conversion service not ready");
        return false;
    }
    auto request               = std::make_shared<GetPose::Request>();
    request->input_poses       = poses;
    request->anchor_frame_name = anchor;
    request->timeout           = 0.5;
    conv_future_.emplace(pose_conversion_client_->async_send_request(request));
    stage_sent_time_ = now_s();
    return true;
}

// A failed conversion leaves the replan pending; it is retried with a fresh plan after check_period
void ReplanTrackerROS2Wrapper::poll_conversion(const PoseStamped& pose) {
    if (stage_ != Stage::CONVERTING) return;
    if (!is_ready(*conv_future_)) {
        if (now_s() - stage_sent_time_ > conversion_timeout_) {
            RCLCPP_WARN(this->get_logger(), "Pose conversion timed out, retrying");
            clear_cycle();
        }
        return;
    }

    auto response = conv_future_->get();
    conv_future_.reset();
    if (!response->tf_success || response->output_poses.empty()) {
        RCLCPP_WARN(this->get_logger(), "Pose conversion failed, retrying");
        clear_cycle();
        return;
    }
    conv_poses_.insert(conv_poses_.end(), response->output_poses.begin(), response->output_poses.end());

    if (conv_stage_ == 1 && conv_split_) {
        if (send_conversion_request({dense_[send_wps_.back()]}, anchor_frame())) {
            conv_stage_ = 2;
        } else {
            clear_cycle();
        }
        return;
    }
    clear_cycle();

    if (conv_poses_.size() != send_wps_.size()) {
        RCLCPP_WARN(this->get_logger(), "Pose conversion returned %zu poses for %zu, retrying", conv_poses_.size(),
                    send_wps_.size());
        return;
    }

    // Drop waypoints passed while planning and converting; the goal is never dropped
    const size_t cursor = nearest_index(dense_, dense_s_, 0, search_distance_, pose.pose.position);
    size_t skip         = 0;
    while (skip + 1 < send_wps_.size()
           && (send_wps_[skip] <= cursor
               || planar_distance(dense_[send_wps_[skip]].pose.position, pose.pose.position)
                      < pruning_distance_threshold_)) {
        ++skip;
    }
    conv_poses_.erase(conv_poses_.begin(), conv_poses_.begin() + static_cast<std::ptrdiff_t>(skip));
    if (!send_locomotion_goal(conv_poses_)) return;

    // The sent plan becomes the route the checks watch
    route_                = dense_;
    route_s_              = dense_s_;
    route_cursor_         = cursor;
    route_global_to_odom_ = lookup_global_to_odom();
    remaining_length_     = route_s_.back() - route_s_[route_cursor_];
    if (initial_length_ < 0.0) initial_length_ = route_s_.back();
    replan_needed_   = false;
    holding_         = false;
    next_check_time_ = now_s() + check_period_;
    clear_validity();  // a pending check was for the old route
}

void ReplanTrackerROS2Wrapper::clear_cycle() {
    plan_send_future_   = {};
    plan_result_future_ = {};
    if (conv_future_) pose_conversion_client_->remove_pending_request(*conv_future_);
    conv_future_.reset();
    conv_stage_ = 0;
    stage_      = Stage::IDLE;
}

void ReplanTrackerROS2Wrapper::clear_validity() {
    if (near_valid_future_) is_path_valid_client_->remove_pending_request(*near_valid_future_);
    if (far_valid_future_) is_path_valid_client_->remove_pending_request(*far_valid_future_);
    near_valid_future_.reset();
    far_valid_future_.reset();
}

// Locomotion

bool ReplanTrackerROS2Wrapper::send_locomotion_goal(const std::vector<PoseStamped>& poses) {
    if (!locomotion_action_client_->action_server_is_ready()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Locomotion server not ready");
        return false;
    }
    auto goal = active_->get_goal();

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

    // Every send ends at the goal, so every send converges on it
    loc_goal.forward_tolerance   = goal->forward_tolerance;
    loc_goal.sidemove_tolerance  = goal->sidemove_tolerance;
    loc_goal.heading_tolerance   = goal->heading_tolerance;
    loc_goal.depth_tolerance     = goal->depth_tolerance;
    loc_goal.max_correction_time = goal->stabilize_duration;

    clear_locomotion();
    loco_send_future_ = locomotion_action_client_->async_send_goal(loc_goal);
    loco_sent_time_   = now_s();
    loco_owned_       = true;

    RCLCPP_INFO(this->get_logger(), "Sent path (%s): %zu waypoints", replan_reason_, poses.size());
    return true;
}

// Results of a trajectory left over from a replaced goal, or canceled to hold, are ignored
ReplanTrackerROS2Wrapper::LocoEvent ReplanTrackerROS2Wrapper::poll_locomotion() {
    if (loco_send_future_) {
        if (!is_ready(*loco_send_future_)) {
            if (now_s() - loco_sent_time_ <= send_timeout_) return LocoEvent::NONE;
            if (!loco_owned_) {
                clear_locomotion();
                return LocoEvent::NONE;
            }
            RCLCPP_ERROR(this->get_logger(), "Locomotion goal not acknowledged within %.1f s", send_timeout_);
            return LocoEvent::FAILED;  // kept in flight so the stop can cancel it
        }
        loco_handle_ = loco_send_future_->get();
        loco_send_future_.reset();
        const bool owned = loco_owned_;
        if (!loco_handle_) {
            clear_locomotion();
            if (owned) RCLCPP_ERROR(this->get_logger(), "Locomotion goal rejected");
            return owned ? LocoEvent::FAILED : LocoEvent::NONE;
        }
        if (holding_ && !loco_cancel_sent_) {  // accepted after the hold began
            try {
                locomotion_action_client_->async_cancel_goal(loco_handle_);
            } catch (const std::exception& e) {
                RCLCPP_DEBUG(this->get_logger(), "Locomotion cancel skipped: %s", e.what());
            }
            loco_cancel_sent_ = true;
        }
        try {
            loco_result_future_ = locomotion_action_client_->async_get_result(loco_handle_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Could not get locomotion result: %s", e.what());
            clear_locomotion();
            return owned ? LocoEvent::FAILED : LocoEvent::NONE;
        }
    }
    if (!loco_result_future_ || !is_ready(*loco_result_future_)) return LocoEvent::NONE;

    const auto code  = loco_result_future_->get().code;
    const bool owned = loco_owned_;
    clear_locomotion();
    if (!owned) return LocoEvent::NONE;
    if (code == rclcpp_action::ResultCode::SUCCEEDED) return LocoEvent::SUCCEEDED;
    RCLCPP_WARN(this->get_logger(), "Locomotion ended unexpectedly (code %d)", static_cast<int>(code));
    return LocoEvent::FAILED;
}

void ReplanTrackerROS2Wrapper::clear_locomotion() {
    loco_send_future_.reset();
    loco_handle_.reset();
    loco_result_future_.reset();
    loco_cancel_sent_ = false;
    loco_owned_       = false;
}

// Helpers

std::optional<ReplanTrackerROS2Wrapper::PoseStamped> ReplanTrackerROS2Wrapper::get_current_pose() {
    try {
        auto tf = tf_buffer_->lookupTransform(global_frame_id_, base_link_frame_id_, tf2::TimePointZero);
        PoseStamped pose;
        pose.header.frame_id  = global_frame_id_;
        pose.pose.position.x  = tf.transform.translation.x;
        pose.pose.position.y  = tf.transform.translation.y;
        pose.pose.position.z  = tf.transform.translation.z;
        pose.pose.orientation = tf.transform.rotation;
        return pose;
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Could not get current pose: %s", ex.what());
        return std::nullopt;
    }
}

void ReplanTrackerROS2Wrapper::publish_feedback() {
    if (initial_length_ < 0.0) return;  // nothing sent yet
    auto feedback                 = std::make_shared<TrackPath::Feedback>();
    feedback->distance_remaining  = remaining_length_;
    feedback->percentage_complete = initial_length_ > 0.0
                                        ? std::clamp(100.0 * (1.0 - remaining_length_ / initial_length_), 0.0, 100.0)
                                        : 100.0;
    active_->publish_feedback(feedback);
}

std::string ReplanTrackerROS2Wrapper::anchor_frame() const {
    const auto& anchor = active_->get_goal()->anchor_frame_name;
    return anchor.empty() ? base_link_frame_id_ : anchor;
}

// Nearest pose to p among those up to search_distance along the path past from; optionally its distance
size_t ReplanTrackerROS2Wrapper::nearest_index(const std::vector<PoseStamped>& path, const std::vector<double>& s,
                                               size_t from, double search_distance,
                                               const geometry_msgs::msg::Point& p, double* distance) {
    size_t best      = from;
    double best_dist = planar_distance(path[from].pose.position, p);
    for (size_t i = from + 1; i < path.size() && s[i] - s[from] <= search_distance; ++i) {
        const double dist = planar_distance(path[i].pose.position, p);
        if (dist < best_dist) {
            best      = i;
            best_dist = dist;
        }
    }
    if (distance) *distance = best_dist;
    return best;
}

double ReplanTrackerROS2Wrapper::planar_distance(const geometry_msgs::msg::Point& a,
                                                 const geometry_msgs::msg::Point& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

double ReplanTrackerROS2Wrapper::perpendicular_distance(const PoseStamped& pt, const PoseStamped& line_start,
                                                        const PoseStamped& line_end) {
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
    double ax    = pvx - pvdot * dx;
    double ay    = pvy - pvdot * dy;
    return std::sqrt(ax * ax + ay * ay);
}

// Appends the kept dense indices in (begin, end]
void ReplanTrackerROS2Wrapper::ramer_douglas_peucker(size_t begin, size_t end, double epsilon,
                                                     std::vector<size_t>& out) const {
    double dmax  = 0.0;
    size_t index = begin;
    for (size_t i = begin + 1; i < end; ++i) {
        const double d = perpendicular_distance(dense_[i], dense_[begin], dense_[end]);
        if (d > dmax) {
            index = i;
            dmax  = d;
        }
    }
    if (dmax > epsilon) {
        ramer_douglas_peucker(begin, index, epsilon, out);
        ramer_douglas_peucker(index, end, epsilon, out);
    } else {
        out.push_back(end);
    }
}
