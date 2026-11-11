#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

// Picks a free goal near a blocked one on a nav2 costmap published as an OccupancyGrid
// (0 free, 1..98 scaled cost, 99 inscribed, 100 lethal, -1 unknown).
namespace goal_sampling {

using Grid  = nav_msgs::msg::OccupancyGrid;
using Point = geometry_msgs::msg::Point;

// Grid value at (x, y) in the grid's frame, nullopt outside it
inline std::optional<int8_t> value_at(const Grid& grid, double x, double y) {
    const auto& info = grid.info;
    if (info.resolution <= 0.0f || info.width == 0 || info.height == 0) return std::nullopt;
    const auto& q    = info.origin.orientation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    const double dx  = x - info.origin.position.x;
    const double dy  = y - info.origin.position.y;
    const double gx  = std::cos(yaw) * dx + std::sin(yaw) * dy;
    const double gy  = -std::sin(yaw) * dx + std::cos(yaw) * dy;
    const auto col   = static_cast<int64_t>(std::floor(gx / info.resolution));
    const auto row   = static_cast<int64_t>(std::floor(gy / info.resolution));
    if (col < 0 || row < 0 || col >= info.width || row >= info.height) return std::nullopt;
    const size_t index = static_cast<size_t>(row) * info.width + static_cast<size_t>(col);
    if (index >= grid.data.size()) return std::nullopt;
    return grid.data[index];
}

// Inside the grid and unknown or at most threshold
inline bool is_free(const Grid& grid, double x, double y, int threshold) {
    const auto value = value_at(grid, x, y);
    return value && (*value < 0 || *value <= threshold);
}

inline bool near_any(double x, double y, const std::vector<Point>& points, double radius) {
    return std::any_of(points.begin(), points.end(),
                       [&](const Point& p) { return std::hypot(p.x - x, p.y - y) < radius; });
}

// Free and away from every tried target
inline bool usable(const Grid& grid, double x, double y, int threshold, const std::vector<Point>& tried,
                   double tried_radius) {
    return is_free(grid, x, y, threshold) && !near_any(x, y, tried, tried_radius);
}

// Candidates on rings spacing apart around nominal out to radius, nominal itself first. The first ring with a usable
// candidate wins, ties to the one nearest boat.
inline std::optional<Point> choose(const Grid& grid, const Point& nominal, double radius, double spacing,
                                   int threshold, const Point& boat, const std::vector<Point>& tried,
                                   double tried_radius) {
    if (usable(grid, nominal.x, nominal.y, threshold, tried, tried_radius)) return nominal;
    if (!(spacing > 0.0) || !(radius > 0.0)) return std::nullopt;

    const int rings = static_cast<int>(std::floor(radius / spacing + 1e-9));
    for (int k = 1; k <= rings; ++k) {
        const double r = k * spacing;
        const int n    = std::max(6, static_cast<int>(std::ceil(2.0 * M_PI * r / spacing)));
        std::optional<Point> best;
        double best_dist = 0.0;
        for (int i = 0; i < n; ++i) {
            const double a = 2.0 * M_PI * i / n;
            Point c        = nominal;
            c.x += r * std::cos(a);
            c.y += r * std::sin(a);
            if (!usable(grid, c.x, c.y, threshold, tried, tried_radius)) continue;
            const double dist = std::hypot(c.x - boat.x, c.y - boat.y);
            if (!best || dist < best_dist) {
                best      = c;
                best_dist = dist;
            }
        }
        if (best) return best;
    }
    return std::nullopt;
}

}  // namespace goal_sampling
