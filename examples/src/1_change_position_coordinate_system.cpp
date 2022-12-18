#include "frames/frames.h"

/**
 * @brief Example of how to change the coordinate system of a position
 *
 */
auto main() -> int
{
	using namespace frame_utils::coordinates;
	using namespace frame_utils::geometry;

	// Defines a point in ENU coordinates, with odom as origin
	constexpr auto some_point_in_enu = Position<odom> { 1, 2, 3 };

	// Converts the ENU coordinates to NED coordinates
	constexpr auto some_point_in_ned = ENUtoNED * some_point_in_enu;

	// Check that this fits our intuition i.e. swap x and y, and negate z. Note the different coordinate systems
	static_assert(some_point_in_ned == Position<odom_ned> { 2, 1, -3 });

	// Compile-error: Incorrect values
	// static_assert(some_point_in_ned == Position<odom_ned> { 1, 2, 3 });

	// Compile-error: different coordinate_system given; operator== is not defined for different coordinate systems
	// static_assert(some_point_in_ned == geometry::Position<odom> { 2, 1, -3 });

	return 0;
}