#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>

namespace path_headings {

// Sets each pose's yaw to the direction of travel to the next pose; planners leave intermediate poses at identity,
// which IsPathValid's footprint check would read as facing +x. The last pose keeps its (goal) orientation, and a pose
// with no distinct position ahead takes the heading before it (the first such takes the next one).
inline void assign_travel_headings(std::vector<geometry_msgs::msg::PoseStamped>& poses, double min_step = 1e-3) {
    if (poses.size() < 2) return;
    const size_t last = poses.size() - 1;

    std::vector<std::optional<double>> yaws(last);
    for (size_t i = 0; i < last; ++i) {
        const double dx = poses[i + 1].pose.position.x - poses[i].pose.position.x;
        const double dy = poses[i + 1].pose.position.y - poses[i].pose.position.y;
        if (std::hypot(dx, dy) >= min_step) yaws[i] = std::atan2(dy, dx);
    }

    std::optional<double> carry;
    for (const auto& yaw : yaws) {
        if (yaw) {
            carry = yaw;
            break;
        }
    }
    if (!carry) return;  // all positions coincide

    for (size_t i = 0; i < last; ++i) {
        if (yaws[i]) carry = yaws[i];
        auto& q = poses[i].pose.orientation;
        q.x     = 0.0;
        q.y     = 0.0;
        q.z     = std::sin(*carry / 2.0);
        q.w     = std::cos(*carry / 2.0);
    }
}

}  // namespace path_headings
