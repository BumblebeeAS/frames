#include "frames/frames.h"

/**
 * @brief Example of how to change the coordinate system of an orientation
 *
 * Note that an orientation is really just a rotation from one coordinate system to another.
 *
 * For example, the rotation quaternion in <namespace>/base_link is the rotation of the ENU (East-North-Up) frame to the
 * FLU (Front-Left-Up) frame. Similarly, the rotation quaternion in <namespace>/base_link_ned is the rotation of the NED
 * (North-East-Down) frame to the FRD (Front-Right-Down) frame.
 *
 * Hence, with the correct composition of rotations, we can convert from one coordinate system to another.
 * The following example shows how to convert from ENU to NED for orientation. Results are compared with ROS TF.
 */
auto main() -> int
{
	using namespace frame_utils::coordinates;
	using namespace frame_utils::geometry;

	// Predefined rotation from ENU to NED
	constexpr auto world_to_world_ned = ENUtoNED;

	// Alternatively, we can also get NED by rotating ENU about its z-axis by 90 degrees, followed by about its x-axis
	// by 180 degrees
	constexpr auto world_to_world_ned_custom = make_rotation<ENU, NED>(90, 0, 180).value();

	// We expect the two to be the same
	static_assert(world_to_world_ned == world_to_world_ned_custom);

	// Ground-truth values taken from ROS TF. The two values are taken at the same time, so they should be the same.
	constexpr auto base_link_orientation = make_rotation<ENU, FLU>(-103.123, 4.947, 1.791).value();
	constexpr auto base_link_ned_orientation = make_rotation<NED, FRD>(-166.877, -4.947, 1.792).value();

	// Using frames to compose rotations, we have (NED -> FRD) is equivalent to NED -> (ENU -> FLU) -> FRD
	constexpr auto base_link_to_base_link_ned = make_rotation<FLU, FRD>(0, 0, 180).value();
	constexpr auto world_ned_to_base_link_ned_custom = (NEDtoENU(base_link_orientation))(base_link_to_base_link_ned);

	// Due to loss of precision, the two values are not exactly the same. However, they should be close enough.
	using namespace frame_utils::math;
	static_assert(std::is_same_v<decltype(base_link_ned_orientation), decltype(world_ned_to_base_link_ned_custom)>);
	static_assert(is_close(base_link_ned_orientation.getW(), world_ned_to_base_link_ned_custom.getW(), 0.000001));
	static_assert(is_close(base_link_ned_orientation.getX(), world_ned_to_base_link_ned_custom.getX(), 0.00001));
	static_assert(is_close(base_link_ned_orientation.getY(), world_ned_to_base_link_ned_custom.getY(), 0.00001));
	static_assert(is_close(base_link_ned_orientation.getZ(), world_ned_to_base_link_ned_custom.getZ(), 0.000001));

	return 0;
}