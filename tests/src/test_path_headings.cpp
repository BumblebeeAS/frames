#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "nodes/path_headings.h"

using geometry_msgs::msg::PoseStamped;
using path_headings::assign_travel_headings;

namespace {

std::vector<PoseStamped> path(const std::vector<std::pair<double, double>>& points) {
    std::vector<PoseStamped> poses(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        poses[i].pose.position.x    = points[i].first;
        poses[i].pose.position.y    = points[i].second;
        poses[i].pose.orientation.w = 1.0;
    }
    return poses;
}

double yaw(const PoseStamped& p) {
    const auto& q = p.pose.orientation;
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

double angle_diff(double a, double b) { return std::remainder(a - b, 2.0 * M_PI); }

}  // namespace

TEST(PathHeadings, StraightLine) {
    auto poses = path({{0, 0}, {0, 0.1}, {0, 0.2}, {0, 0.3}});
    assign_travel_headings(poses);
    for (size_t i = 0; i < 3; ++i) EXPECT_NEAR(yaw(poses[i]), M_PI / 2, 1e-9);
}

TEST(PathHeadings, CornerSwitchesAtVertex) {
    auto poses = path({{0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}});
    assign_travel_headings(poses);
    EXPECT_NEAR(yaw(poses[1]), 0.0, 1e-9);
    EXPECT_NEAR(yaw(poses[2]), M_PI / 2, 1e-9);
    EXPECT_NEAR(yaw(poses[3]), M_PI / 2, 1e-9);
}

TEST(PathHeadings, UTurnWrapsThroughPi) {
    auto poses = path({{0, 0}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}});
    assign_travel_headings(poses);
    EXPECT_NEAR(angle_diff(yaw(poses[2]), M_PI), 0.0, 1e-9);
    EXPECT_NEAR(angle_diff(yaw(poses[3]), M_PI), 0.0, 1e-9);
}

TEST(PathHeadings, LastPoseKeepsGoalOrientation) {
    auto poses = path({{0, 0}, {1, 0}, {2, 0}});
    poses.back().pose.orientation.z = std::sin(0.7 / 2);
    poses.back().pose.orientation.w = std::cos(0.7 / 2);
    assign_travel_headings(poses);
    EXPECT_NEAR(yaw(poses.back()), 0.7, 1e-9);
}

TEST(PathHeadings, DuplicatesCarryNeighbouringHeading) {
    auto poses = path({{0, 0}, {0, 0}, {0, 1}, {0, 1}, {1, 1}, {1, 1}});
    assign_travel_headings(poses);
    EXPECT_NEAR(yaw(poses[0]), M_PI / 2, 1e-9);  // leading duplicate takes the first heading ahead
    EXPECT_NEAR(yaw(poses[1]), M_PI / 2, 1e-9);
    EXPECT_NEAR(yaw(poses[2]), M_PI / 2, 1e-9);  // duplicate mid-path keeps the heading before it
    EXPECT_NEAR(yaw(poses[3]), 0.0, 1e-9);
    EXPECT_NEAR(yaw(poses[4]), 0.0, 1e-9);  // duplicate before the goal keeps the heading before it
}

TEST(PathHeadings, DegeneratePathsUnchanged) {
    auto single = path({{3, 4}});
    assign_travel_headings(single);
    EXPECT_DOUBLE_EQ(single[0].pose.orientation.w, 1.0);

    auto coincident = path({{1, 1}, {1, 1}, {1, 1}});
    assign_travel_headings(coincident);
    for (const auto& p : coincident) EXPECT_DOUBLE_EQ(p.pose.orientation.w, 1.0);

    std::vector<PoseStamped> empty;
    assign_travel_headings(empty);
    EXPECT_TRUE(empty.empty());
}
