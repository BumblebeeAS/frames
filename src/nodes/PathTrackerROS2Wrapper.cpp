#include "nodes/PathTrackerROS2Wrapper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::placeholders;

namespace {
template <typename F>
bool is_ready(const F& f) {
    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}
}  // namespace

PathTrackerROS2Wrapper::PathTrackerROS2Wrapper(const std::string& name) : Node(name) {
    // Parameters
    tick_rate_                  = this->declare_parameter("tracking_loop_rate", 10.0);
    validity_period_            = 1.0 / this->declare_parameter("path_validity_check_rate", 1.0);
    simplification_epsilon_     = this->declare_parameter("simplification_epsilon", 0.5);
    lookahead_waypoints_        = std::max<int>(1, this->declare_parameter("lookahead_waypoints", 3));
    pruning_distance_threshold_ = this->declare_parameter("pruning_distance_threshold", 1.0);
    search_distance_            = this->declare_parameter("search_distance", 3.0);
    progress_radius_            = this->declare_parameter("progress_radius", 0.5);
    progress_timeout_           = this->declare_parameter("progress_timeout", 30.0);
    replan_backoff_             = this->declare_parameter("replan_backoff", 1.0);
    plan_timeout_               = this->declare_parameter("plan_timeout", 5.0);
    conversion_timeout_         = this->declare_parameter("conversion_timeout", 3.0);
    send_timeout_               = this->declare_parameter("send_timeout", 2.0);
    stop_timeout_               = this->declare_parameter("stop_timeout", 3.0);
    tf_failure_limit_           = this->declare_parameter("tf_failure_limit", 10);

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

    // Server callbacks and tick never run concurrently; client responses resolve futures on their own group
    server_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    client_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // Action Server
    action_server_ = rclcpp_action::create_server<TrackPath>(
        this, "in/locomotion/track_path", std::bind(&PathTrackerROS2Wrapper::handle_goal, this, _1, _2),
        std::bind(&PathTrackerROS2Wrapper::handle_cancel, this, _1),
        std::bind(&PathTrackerROS2Wrapper::handle_accepted, this, _1), rcl_action_server_get_default_options(),
        server_cb_group_);

    // Clients
    nav2_action_client_ = rclcpp_action::create_client<ComputePathToPose>(this, nav2_action_name, client_cb_group_);
    locomotion_action_client_
        = rclcpp_action::create_client<Locomotion>(this, locomotion_action_name, client_cb_group_);
    is_path_valid_client_
        = this->create_client<IsPathValid>(nav2_is_path_valid_service, rclcpp::ServicesQoS(), client_cb_group_);
    pose_conversion_client_
        = this->create_client<GetPose>(pose_conversion_service_name, rclcpp::ServicesQoS(), client_cb_group_);

    tick_timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / tick_rate_),
                                          std::bind(&PathTrackerROS2Wrapper::tick, this), server_cb_group_);
}

double PathTrackerROS2Wrapper::now_s() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

rclcpp_action::GoalResponse PathTrackerROS2Wrapper::handle_goal(const rclcpp_action::GoalUUID& uuid,
                                                                std::shared_ptr<const TrackPath::Goal> goal) {
    (void)uuid;
    (void)goal;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse PathTrackerROS2Wrapper::handle_cancel(
    const std::shared_ptr<GoalHandleTrackPath> goal_handle) {
    (void)goal_handle;
    RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
    return rclcpp_action::CancelResponse::ACCEPT;
}

void PathTrackerROS2Wrapper::handle_accepted(const std::shared_ptr<GoalHandleTrackPath> goal_handle) {
    // Only queue; tick() adopts it
    if (pending_) {
        auto result    = std::make_shared<TrackPath::Result>();
        result->status = TrackPath::Result::STATUS_ABORTED;
        pending_->abort(result);
    }
    pending_ = goal_handle;
}

void PathTrackerROS2Wrapper::tick() {
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
        begin_stop(AfterStop::TERMINATE, TrackPath::Result::STATUS_ABORTED);
        return;
    }

    // Replace and replan without stopping; the running window continues until the new plan supersedes it
    if (pending_) {
        RCLCPP_INFO(this->get_logger(), "New goal received, preempting active goal");
        auto result    = std::make_shared<TrackPath::Result>();
        result->status = TrackPath::Result::STATUS_ABORTED;
        active_->abort(result);
        auto goal_handle = std::move(pending_);
        start_goal(goal_handle);
        return;
    }

    poll_plan();

    if (state_ == State::PLANNING) {
        poll_locomotion();  // unowned window from a preempted goal; resolve it for a possible stop
        return;
    }
    if (state_ != State::TRACKING) return;

    auto pose = get_current_pose();
    if (!pose) {
        if (++tf_failures_ >= tf_failure_limit_) {
            RCLCPP_ERROR(this->get_logger(), "No robot pose for %d ticks, aborting", tf_failures_);
            begin_stop(AfterStop::TERMINATE, TrackPath::Result::STATUS_FAILURE);
        }
        return;
    }
    tf_failures_ = 0;
    step_track(*pose);
}

void PathTrackerROS2Wrapper::start_goal(const std::shared_ptr<GoalHandleTrackPath>& goal_handle) {
    active_               = goal_handle;
    replan_failures_      = 0;
    tf_failures_          = 0;
    next_plan_time_       = 0.0;
    progress_anchor_time_ = -1.0;

    dense_.clear();
    dense_s_.clear();
    waypoints_.clear();
    cursor_         = 0;
    next_wp_        = 0;
    need_send_      = false;
    blocked_beyond_ = false;
    has_committed_  = false;
    sent_final_     = false;
    loco_owned_     = false;
    clear_plan_request();
    clear_conversion();
    clear_validity();

    RCLCPP_INFO(this->get_logger(), "Starting TrackPath goal, requesting plan");
    state_ = State::PLANNING;
    request_plan();
}

void PathTrackerROS2Wrapper::terminate(uint8_t status) {
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
    clear_plan_request();
    clear_conversion();
    clear_validity();
    dense_.clear();
    state_ = State::IDLE;
}

void PathTrackerROS2Wrapper::begin_stop(AfterStop after, uint8_t status) {
    state_         = State::STOPPING;
    after_stop_    = after;
    stop_status_   = status;
    stop_deadline_ = now_s() + stop_timeout_;
    clear_plan_request();
    clear_conversion();
    clear_validity();
    step_stop();
}

void PathTrackerROS2Wrapper::step_stop() {
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

    if (after_stop_ == AfterStop::REPLAN && !active_->is_canceling()) {
        RCLCPP_INFO(this->get_logger(), "Stopped, replanning");
        next_plan_time_ = 0.0;
        state_          = State::PLANNING;
        return;
    }
    terminate(stop_status_);
}

void PathTrackerROS2Wrapper::step_track(const PoseStamped& pose) {
    const double now = now_s();
    if (progress_anchor_time_ < 0.0 || planar_distance(progress_anchor_, pose.pose.position) > progress_radius_) {
        reset_progress(pose);
    } else if (now - progress_anchor_time_ > progress_timeout_) {
        RCLCPP_ERROR(this->get_logger(), "No progress for %.1f s, aborting", progress_timeout_);
        begin_stop(AfterStop::TERMINATE, TrackPath::Result::STATUS_FAILURE);
        return;
    }

    advance_cursor(pose);
    const size_t last_wp = waypoints_.size() - 1;
    while (next_wp_ < last_wp
           && (waypoints_[next_wp_] <= cursor_
               || planar_distance(dense_[waypoints_[next_wp_]].pose.position, pose.pose.position)
                      < pruning_distance_threshold_)) {
        ++next_wp_;
        if (!sent_final_) need_send_ = true;
    }

    switch (poll_locomotion()) {
    case LocoEvent::SUCCEEDED:
        if (sent_final_) {
            RCLCPP_INFO(this->get_logger(), "Path Tracker finished successfully!");
            terminate(TrackPath::Result::STATUS_SUCCESS);
            return;
        }
        // Window finished before the next one went out; its last waypoint is reached
        next_wp_   = std::max(next_wp_, std::min(sent_wp_last_ + 1, last_wp));
        need_send_ = true;
        break;
    case LocoEvent::FAILED:
        begin_stop(AfterStop::TERMINATE, TrackPath::Result::STATUS_FAILURE);
        return;
    case LocoEvent::NONE:
        break;
    }

    poll_validity();
    if (state_ != State::TRACKING) return;
    if (!committed_valid_future_ && !remaining_valid_future_ && now >= next_validity_time_) request_validity();

    if (need_send_ && conv_stage_ == 0 && !loco_send_future_) {
        const size_t wp_end = std::min(next_wp_ + static_cast<size_t>(lookahead_waypoints_), waypoints_.size());
        if (blocked_beyond_ && (!has_committed_ || waypoints_[wp_end - 1] > committed_end_)) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Path ahead blocked, holding window until replan succeeds");
        } else {
            start_conversion(next_wp_, wp_end);
        }
    }
    poll_conversion();
    publish_feedback();
}

// Planning

void PathTrackerROS2Wrapper::request_plan() {
    if (!nav2_action_client_->action_server_is_ready()) {
        on_plan_failed("planner server unavailable");
        return;
    }
    ComputePathToPose::Goal goal;
    goal.goal           = active_->get_goal()->goal_pose;
    goal.use_start      = false;  // Use current robot pose
    plan_send_future_   = nav2_action_client_->async_send_goal(goal);
    plan_result_future_ = {};
    plan_in_flight_     = true;
    plan_sent_time_     = now_s();
}

void PathTrackerROS2Wrapper::poll_plan() {
    if (!plan_in_flight_) {
        const bool wanted = state_ == State::PLANNING || (state_ == State::TRACKING && blocked_beyond_);
        if (wanted && now_s() >= next_plan_time_) request_plan();
        return;
    }

    if (now_s() - plan_sent_time_ > plan_timeout_) {
        if (is_ready(plan_send_future_) && plan_send_future_.get()) {
            try {
                nav2_action_client_->async_cancel_goal(plan_send_future_.get());
            } catch (const std::exception&) {
            }
        }
        clear_plan_request();
        on_plan_failed("timed out");
        return;
    }

    if (!plan_result_future_.valid()) {
        if (!is_ready(plan_send_future_)) return;
        auto handle = plan_send_future_.get();
        if (!handle) {
            clear_plan_request();
            on_plan_failed("rejected");
            return;
        }
        try {
            plan_result_future_ = nav2_action_client_->async_get_result(handle);
        } catch (const std::exception& e) {
            clear_plan_request();
            on_plan_failed(e.what());
            return;
        }
    }
    if (!is_ready(plan_result_future_)) return;

    auto wrapped = plan_result_future_.get();
    clear_plan_request();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED || !wrapped.result || wrapped.result->path.poses.empty()) {
        on_plan_failed(wrapped.result && !wrapped.result->error_msg.empty() ? wrapped.result->error_msg.c_str()
                                                                            : "no path");
        return;
    }
    if (!wrapped.result->path.header.frame_id.empty() && wrapped.result->path.header.frame_id != global_frame_id_) {
        on_plan_failed("path frame differs from global_frame_id");
        return;
    }
    install_path(wrapped.result->path.poses);
}

void PathTrackerROS2Wrapper::on_plan_failed(const char* reason) {
    const int max_replans = active_->get_goal()->max_replans;
    if (++replan_failures_ > max_replans) {
        RCLCPP_ERROR(this->get_logger(), "Planning failed (%s), %d consecutive failures, aborting", reason,
                     replan_failures_);
        begin_stop(AfterStop::TERMINATE, TrackPath::Result::STATUS_FAILURE);
        return;
    }
    RCLCPP_WARN(this->get_logger(), "Planning failed (%s), retry %d/%d in %.1f s", reason, replan_failures_,
                max_replans, replan_backoff_);
    next_plan_time_ = now_s() + replan_backoff_;
}

void PathTrackerROS2Wrapper::install_path(const std::vector<PoseStamped>& poses) {
    dense_ = poses;
    dense_s_.assign(dense_.size(), 0.0);
    for (size_t i = 1; i < dense_.size(); ++i) {
        dense_s_[i] = dense_s_[i - 1] + planar_distance(dense_[i - 1].pose.position, dense_[i].pose.position);
    }
    waypoints_ = {0};
    if (dense_.size() > 1) ramer_douglas_peucker(0, dense_.size() - 1, simplification_epsilon_, waypoints_);

    cursor_          = 0;
    next_wp_         = 0;
    need_send_       = true;
    blocked_beyond_  = false;
    has_committed_   = false;
    sent_final_      = false;
    loco_owned_      = false;
    replan_failures_ = 0;
    clear_conversion();
    clear_validity();
    next_validity_time_ = now_s() + validity_period_;
    state_              = State::TRACKING;

    RCLCPP_INFO(this->get_logger(), "Plan received. Dense size: %zu, Simplified size: %zu", dense_.size(),
                waypoints_.size());
}

void PathTrackerROS2Wrapper::clear_plan_request() {
    plan_in_flight_     = false;
    plan_send_future_   = {};
    plan_result_future_ = {};
}

// Windows

void PathTrackerROS2Wrapper::start_conversion(size_t wp_begin, size_t wp_end) {
    auto goal        = active_->get_goal();
    const bool final = wp_end == waypoints_.size();
    conv_split_      = goal->last_goal_anchor_only && final && (wp_end - wp_begin) > 1;
    conv_wp_begin_   = wp_begin;
    conv_wp_end_     = wp_end;
    conv_poses_.clear();

    std::vector<PoseStamped> poses;
    for (size_t i = wp_begin; i < wp_end - (conv_split_ ? 1 : 0); ++i) poses.push_back(dense_[waypoints_[i]]);
    const bool base_anchor = conv_split_ || (goal->last_goal_anchor_only && !final);
    if (send_conversion_request(poses, base_anchor ? base_link_frame_id_ : anchor_frame())) conv_stage_ = 1;
}

bool PathTrackerROS2Wrapper::send_conversion_request(const std::vector<PoseStamped>& poses, const std::string& anchor) {
    if (!pose_conversion_client_->service_is_ready()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Pose conversion service not ready");
        return false;
    }
    auto request               = std::make_shared<GetPose::Request>();
    request->input_poses       = poses;
    request->anchor_frame_name = anchor;
    request->timeout           = 0.5;
    conv_future_.emplace(pose_conversion_client_->async_send_request(request));
    conv_sent_time_ = now_s();
    return true;
}

void PathTrackerROS2Wrapper::poll_conversion() {
    if (conv_stage_ == 0) return;
    if (!is_ready(*conv_future_)) {
        if (now_s() - conv_sent_time_ > conversion_timeout_) {
            RCLCPP_WARN(this->get_logger(), "Pose conversion timed out, retrying");
            clear_conversion();
        }
        return;
    }

    auto response = conv_future_->get();
    conv_future_.reset();
    if (!response->tf_success || response->output_poses.empty()) {
        RCLCPP_WARN(this->get_logger(), "Pose conversion failed, retrying");
        conv_stage_ = 0;
        return;
    }
    conv_poses_.insert(conv_poses_.end(), response->output_poses.begin(), response->output_poses.end());

    if (conv_stage_ == 1 && conv_split_) {
        conv_stage_ = send_conversion_request({dense_[waypoints_[conv_wp_end_ - 1]]}, anchor_frame()) ? 2 : 0;
        return;
    }
    conv_stage_ = 0;

    if (conv_poses_.size() != conv_wp_end_ - conv_wp_begin_) {
        RCLCPP_WARN(this->get_logger(), "Pose conversion returned %zu poses for %zu, retrying", conv_poses_.size(),
                    conv_wp_end_ - conv_wp_begin_);
        return;
    }
    // Drop waypoints passed while converting
    const size_t skip = next_wp_ > conv_wp_begin_ ? next_wp_ - conv_wp_begin_ : 0;
    if (skip >= conv_poses_.size()) return;
    conv_poses_.erase(conv_poses_.begin(), conv_poses_.begin() + skip);
    send_locomotion_goal(conv_poses_, conv_wp_begin_ + skip, conv_wp_end_ - 1);
}

void PathTrackerROS2Wrapper::clear_conversion() {
    if (conv_future_) pose_conversion_client_->remove_pending_request(*conv_future_);
    conv_future_.reset();
    conv_stage_ = 0;
}

void PathTrackerROS2Wrapper::send_locomotion_goal(const std::vector<PoseStamped>& poses, size_t wp_begin,
                                                  size_t wp_last) {
    if (!locomotion_action_client_->action_server_is_ready()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Locomotion server not ready");
        return;
    }
    auto goal        = active_->get_goal();
    const bool final = wp_last == waypoints_.size() - 1;

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

    // Intermediate windows succeed as soon as their trajectory ends; only the final window converges
    if (final) {
        loc_goal.forward_tolerance   = goal->forward_tolerance;
        loc_goal.sidemove_tolerance  = goal->sidemove_tolerance;
        loc_goal.heading_tolerance   = goal->heading_tolerance;
        loc_goal.depth_tolerance     = goal->depth_tolerance;
        loc_goal.max_correction_time = goal->stabilize_duration;
    } else {
        loc_goal.max_correction_time = 0;
    }

    clear_locomotion();
    loco_send_future_ = locomotion_action_client_->async_send_goal(loc_goal);
    loco_sent_time_   = now_s();
    loco_owned_       = true;
    sent_final_       = final;
    sent_wp_last_     = wp_last;
    committed_end_    = waypoints_[wp_last];
    has_committed_    = true;
    need_send_        = next_wp_ != wp_begin;

    RCLCPP_INFO(this->get_logger(), "Sent window: waypoints %zu-%zu of %zu%s", wp_begin, wp_last, waypoints_.size(),
                final ? " (final)" : "");
}

PathTrackerROS2Wrapper::LocoEvent PathTrackerROS2Wrapper::poll_locomotion() {
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

void PathTrackerROS2Wrapper::clear_locomotion() {
    loco_send_future_.reset();
    loco_handle_.reset();
    loco_result_future_.reset();
    loco_cancel_sent_ = false;
    loco_owned_       = false;
}

// Validity: committed span is cursor to the end of the sent window, remaining is the rest

void PathTrackerROS2Wrapper::request_validity() {
    if (dense_.empty() || !is_path_valid_client_->service_is_ready()) return;
    next_validity_time_ = now_s() + validity_period_;

    const size_t split = has_committed_ ? std::max(committed_end_, cursor_) : cursor_;
    if (split > cursor_) {
        auto request  = std::make_shared<IsPathValid::Request>();
        request->path = make_path(cursor_, split + 1);
        committed_valid_future_.emplace(is_path_valid_client_->async_send_request(request));
    }
    if (split + 1 < dense_.size()) {
        auto request  = std::make_shared<IsPathValid::Request>();
        request->path = make_path(split, dense_.size());
        remaining_valid_future_.emplace(is_path_valid_client_->async_send_request(request));
    }
    validity_sent_time_ = now_s();
}

void PathTrackerROS2Wrapper::poll_validity() {
    if (!committed_valid_future_ && !remaining_valid_future_) return;
    const bool ready = (!committed_valid_future_ || is_ready(*committed_valid_future_))
                       && (!remaining_valid_future_ || is_ready(*remaining_valid_future_));
    if (!ready) {
        if (now_s() - validity_sent_time_ > validity_period_) clear_validity();
        return;
    }

    const bool committed_ok = !committed_valid_future_ || committed_valid_future_->get()->is_valid;
    const bool remaining_ok = !remaining_valid_future_ || remaining_valid_future_->get()->is_valid;
    clear_validity();

    if (!committed_ok) {
        RCLCPP_WARN(this->get_logger(), "Committed span blocked, stopping to replan");
        begin_stop(AfterStop::REPLAN, TrackPath::Result::STATUS_FAILURE);
    } else if (!remaining_ok && !blocked_beyond_) {
        RCLCPP_WARN(this->get_logger(), "Path ahead blocked, replanning while executing committed span");
        blocked_beyond_ = true;
    }
}

void PathTrackerROS2Wrapper::clear_validity() {
    if (committed_valid_future_) is_path_valid_client_->remove_pending_request(*committed_valid_future_);
    if (remaining_valid_future_) is_path_valid_client_->remove_pending_request(*remaining_valid_future_);
    committed_valid_future_.reset();
    remaining_valid_future_.reset();
}

// Helpers

std::optional<PathTrackerROS2Wrapper::PoseStamped> PathTrackerROS2Wrapper::get_current_pose() {
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

void PathTrackerROS2Wrapper::advance_cursor(const PoseStamped& pose) {
    size_t best      = cursor_;
    double best_dist = planar_distance(dense_[cursor_].pose.position, pose.pose.position);
    for (size_t i = cursor_ + 1; i < dense_.size() && dense_s_[i] - dense_s_[cursor_] <= search_distance_; ++i) {
        const double dist = planar_distance(dense_[i].pose.position, pose.pose.position);
        if (dist < best_dist) {
            best      = i;
            best_dist = dist;
        }
    }
    cursor_ = best;
}

void PathTrackerROS2Wrapper::reset_progress(const PoseStamped& pose) {
    progress_anchor_      = pose.pose.position;
    progress_anchor_time_ = now_s();
}

void PathTrackerROS2Wrapper::publish_feedback() {
    auto feedback                 = std::make_shared<TrackPath::Feedback>();
    const double total            = dense_s_.back();
    const double remaining        = total - dense_s_[cursor_];
    feedback->distance_remaining  = remaining;
    feedback->percentage_complete = total > 0.0 ? 100.0 * (1.0 - remaining / total) : 100.0;
    active_->publish_feedback(feedback);
}

std::string PathTrackerROS2Wrapper::anchor_frame() const {
    const auto& anchor = active_->get_goal()->anchor_frame_name;
    return anchor.empty() ? base_link_frame_id_ : anchor;
}

nav_msgs::msg::Path PathTrackerROS2Wrapper::make_path(size_t begin, size_t end) const {
    nav_msgs::msg::Path path;
    path.header.frame_id = global_frame_id_;
    path.header.stamp    = this->get_clock()->now();
    path.poses.assign(dense_.begin() + begin, dense_.begin() + end);
    return path;
}

double PathTrackerROS2Wrapper::planar_distance(const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

double PathTrackerROS2Wrapper::perpendicular_distance(const PoseStamped& pt, const PoseStamped& line_start,
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
void PathTrackerROS2Wrapper::ramer_douglas_peucker(size_t begin, size_t end, double epsilon,
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
