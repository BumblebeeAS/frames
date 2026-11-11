#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "nodes/goal_sampling.h"

using goal_sampling::Grid;
using goal_sampling::Point;

namespace {

// 20 x 20 m at 0.1 m, origin (-10, -10), all free
Grid grid() {
    Grid g;
    g.info.resolution           = 0.1f;
    g.info.width                = 200;
    g.info.height               = 200;
    g.info.origin.position.x    = -10.0;
    g.info.origin.position.y    = -10.0;
    g.info.origin.orientation.w = 1.0;
    g.data.assign(200 * 200, 0);
    return g;
}

// Sets every cell within radius of (x, y)
void fill(Grid& g, double x, double y, double radius, int8_t value) {
    for (uint32_t row = 0; row < g.info.height; ++row) {
        for (uint32_t col = 0; col < g.info.width; ++col) {
            const double cx = g.info.origin.position.x + (col + 0.5) * g.info.resolution;
            const double cy = g.info.origin.position.y + (row + 0.5) * g.info.resolution;
            if (std::hypot(cx - x, cy - y) <= radius) g.data[row * g.info.width + col] = value;
        }
    }
}

Point pt(double x, double y) {
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

constexpr double kSpacing = 0.25;
constexpr int kThreshold  = 17;

}  // namespace

TEST(GoalSampling, ValueAtOutsideIsNullopt) {
    const auto g = grid();
    EXPECT_FALSE(goal_sampling::value_at(g, 10.5, 0.0));
    EXPECT_FALSE(goal_sampling::value_at(g, 0.0, -10.5));
    EXPECT_TRUE(goal_sampling::value_at(g, 0.0, 0.0));
}

TEST(GoalSampling, UnknownIsFreeAboveThresholdIsNot) {
    auto g = grid();
    fill(g, 1.0, 1.0, 0.1, -1);
    fill(g, -1.0, -1.0, 0.1, kThreshold + 1);
    fill(g, 2.0, 2.0, 0.1, kThreshold);
    EXPECT_TRUE(goal_sampling::is_free(g, 1.0, 1.0, kThreshold));
    EXPECT_FALSE(goal_sampling::is_free(g, -1.0, -1.0, kThreshold));
    EXPECT_TRUE(goal_sampling::is_free(g, 2.0, 2.0, kThreshold));
    EXPECT_FALSE(goal_sampling::is_free(g, 50.0, 0.0, kThreshold));  // outside
}

TEST(GoalSampling, FreeNominalIsKept) {
    const auto g = grid();
    const auto p = goal_sampling::choose(g, pt(1.0, 2.0), 3.0, kSpacing, kThreshold, pt(-5.0, 0.0), {}, kSpacing);
    ASSERT_TRUE(p);
    EXPECT_DOUBLE_EQ(p->x, 1.0);
    EXPECT_DOUBLE_EQ(p->y, 2.0);
}

TEST(GoalSampling, BlockedNominalShiftsJustClearOfTheBlob) {
    auto g = grid();
    fill(g, 0.0, 0.0, 1.0, 100);
    const auto p = goal_sampling::choose(g, pt(0.0, 0.0), 3.0, kSpacing, kThreshold, pt(-5.0, 0.0), {}, kSpacing);
    ASSERT_TRUE(p);
    const double r = std::hypot(p->x, p->y);
    EXPECT_GE(r, 1.0 - 1e-9);
    EXPECT_LE(r, 1.25 + 1e-9);  // first ring clear of the blob
    EXPECT_LT(p->x, 0.0);       // on the boat's side
}

TEST(GoalSampling, NothingFreeWithinRadius) {
    auto g = grid();
    fill(g, 0.0, 0.0, 4.0, 100);
    EXPECT_FALSE(goal_sampling::choose(g, pt(0.0, 0.0), 3.0, kSpacing, kThreshold, pt(-5.0, 0.0), {}, kSpacing));
}

TEST(GoalSampling, ZeroRadiusOnlyTriesNominal) {
    auto g = grid();
    fill(g, 0.0, 0.0, 0.5, 100);
    EXPECT_FALSE(goal_sampling::choose(g, pt(0.0, 0.0), 0.0, kSpacing, kThreshold, pt(-5.0, 0.0), {}, kSpacing));
}

TEST(GoalSampling, TriedTargetsAreSkipped) {
    const auto g       = grid();
    const auto nominal = pt(0.0, 0.0);
    const auto boat    = pt(-5.0, 0.0);
    const double tried_radius = kSpacing / 2.0;
    const auto first = goal_sampling::choose(g, nominal, 3.0, kSpacing, kThreshold, boat, {nominal}, tried_radius);
    ASSERT_TRUE(first);
    EXPECT_NEAR(std::hypot(first->x, first->y), kSpacing, 1e-9);  // nominal tried, first ring
    EXPECT_LT(first->x, 0.0);                                      // toward the boat

    const auto second
        = goal_sampling::choose(g, nominal, 3.0, kSpacing, kThreshold, boat, {nominal, *first}, tried_radius);
    ASSERT_TRUE(second);
    EXPECT_GE(std::hypot(second->x - first->x, second->y - first->y), tried_radius);
}

TEST(GoalSampling, OutsideGridIsNotChosen) {
    auto g = grid();
    // Nominal on the grid edge, blocked; everything past the edge is outside
    fill(g, 9.9, 0.0, 1.0, 100);
    const auto p = goal_sampling::choose(g, pt(9.9, 0.0), 3.0, kSpacing, kThreshold, pt(20.0, 0.0), {}, kSpacing);
    ASSERT_TRUE(p);
    EXPECT_LT(p->x, 10.0);
    EXPECT_GE(std::hypot(p->x - 9.9, p->y), 1.0 - 1e-9);
}

TEST(GoalSampling, InKeepoutOnlyInsideAZone) {
    auto mask = grid();
    fill(mask, 2.0, 0.0, 1.0, 100);
    EXPECT_TRUE(goal_sampling::in_keepout(mask, 2.0, 0.0));
    EXPECT_TRUE(goal_sampling::in_keepout(mask, 2.9, 0.0));
    EXPECT_FALSE(goal_sampling::in_keepout(mask, 3.2, 0.0));   // just outside
    EXPECT_FALSE(goal_sampling::in_keepout(mask, 0.0, 0.0));   // free
    EXPECT_FALSE(goal_sampling::in_keepout(mask, 50.0, 0.0));  // off the mask
    fill(mask, -5.0, 0.0, 1.0, 99);
    EXPECT_FALSE(goal_sampling::in_keepout(mask, -5.0, 0.0));  // high cost is not a zone
}
