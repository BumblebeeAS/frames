#include <iostream>
#include <limits>

#include "frames/frames.h"

#include "geometry_msgs/Quaternion.h"
#include "tf2/LinearMath/Quaternion.h"

namespace tests
{
using namespace frame_utils::coordinates;
using namespace frame_utils::geometry;
using namespace frame_utils::math;

template<typename R1, typename R2>
using can_sum = decltype(std::declval<R1>() + std::declval<R2>());

constexpr void test_translation()
{
	// Test 1: Position is a translation to some arbitrary point
	constexpr auto x = Position<odom> { 1, 2, 3 };
	constexpr auto y = Translation<Frame<World, ENU>, __ArbitraryOrigin> { 1, 2, 3 };
	static_assert(x == y);

	// Test 2: Translation can be joined together
	constexpr auto a = Translation<Frame<World, ENU>, Vehicle> { 1, 2, 3 };
	constexpr auto b = Translation<Frame<Vehicle, ENU>, __ArbitraryOrigin> { 2, 3, 4 };
	constexpr auto c = Translation<Frame<World, ENU>, __ArbitraryOrigin> { 3, 5, 7 };
	static_assert(a + b == c);

	// Test 3: Translation can change direction
	static_assert(inverse(a) == Translation<Frame<Vehicle, ENU>, World> { -1, -2, -3 });
}

constexpr void test_position()
{
	// Test 1: Addition of position
	constexpr auto p1 = Position<odom> { 1, 2, 3 };
	constexpr auto p2 = Position<odom> { 4, 5, 6 };
	static_assert(p1 + p2 == Position<odom>(5, 7, 9), "Addition of position of the same frame fails!");

	// Test 2: Scaling of position
	constexpr auto temp = Position<odom>(5, 10, 15);
	constexpr auto scaled = 5.0 * p1;
	static_assert(scaled == temp, "Scaling of position fails!");

	// Test 3: Addition of positions of same origin but different frames should not compile!
	constexpr auto p4 = Position<odom_ned> { 1, 3, 5 };
	static_assert(!boost::is_detected_v<can_sum, decltype(p1), decltype(p4)>);

	// Test 4: Addition of positions of different origin but same frames should not compile!
	constexpr auto p5 = Position<base_link> { 1, 3, 5 };
	static_assert(!boost::is_detected_v<can_sum, decltype(p1), decltype(p5)>);

	// Test 5: Adding custom origins
	struct Left_Camera : public Origin
	{ };

	static_assert(is_origin_v<Left_Camera>, "Creation of custom origin fails!");
	using LeftCamOptical = Frame<Left_Camera, RDF>;

	// Defines an object that is 2m to the right and 3m in front of the left
	// camera
	constexpr auto left_cam_object = Position<LeftCamOptical> { 2, 0, 3 };
	static_assert(left_cam_object == left_cam_object);
}

constexpr auto test_rotations() -> void
{
	// Test 1a: Compose with identity
	constexpr auto eye = Rotation<ENU, ENU> { 1, 0, 0, 0 };
	static_assert(eye(ENUtoNED) == ENUtoNED, "Identity rotation fails!");

	// Test 1b: Compose with Equality check
	static_assert(ENUtoNED != NEDtoENU, "Rotations are treated the same despite different frame conventions!");
	static_assert(
		ENUtoNED(NEDtoENU) == static_cast<Rotation<ENU, ENU>>(Identity), "Same rotation is treated differently!"
	);
	static_assert(eye == static_cast<Rotation<ENU, ENU>>(Identity), "Same rotation is treated differently!");

	// Test 2: Norm of quaternion
	static_assert(is_close(eye.norm(), 1.0), "Norm of identity quaternion is not 1!");
	static_assert(is_close(ENUtoNED.norm(), 1.0), "Norm of unit quaternion is not 1!");

	// Test 3: Inverse of quaternion
	static_assert(ENUtoNED.inverse() == static_cast<Rotation<NED, ENU>>(ENUtoNED), "Inverse of unit quaternion fails!");
	static_assert(ENUtoNED.inverse() == NEDtoENU, "Inverse of unit quaternion fails!");

	constexpr auto q2 = Rotation<ENU, __ArbitraryConvention> { 0.9961947, 0, 0, 0.0871557 }; // 10 degrees about z-axis
	static_assert(
		q2.inverse() == Rotation<__ArbitraryConvention, ENU> { -0.9961947, 0, 0, 0.0871557 },
		"Inverse of generic quaternion fails"
	);

	// Test 4: Equality of quaternions
	static_assert(NEDtoENU == NEDtoENU, "Quaternion equality fails!");
	static_assert(NEDtoENU == -1 * NEDtoENU, "Quaternion equality fails");

	// Test 5: Quaternion normalisation
	static_assert(NEDtoENU.normalised() == NEDtoENU, "Quaternion normalisation fails!");
	static_assert(q2.normalised() == q2, "Quaternion normalisation fails!");

	// Test 6: Relative Rotation of quaternions
	constexpr auto q3 = static_cast<Rotation<NED, __ArbitraryConvention>>(q2);
	constexpr auto results = ENUtoNED(q3);

	constexpr auto qr = results(q3.inverse());
	static_assert(qr == ENUtoNED, "Rotation of quaternion fails!");
	static_assert(qr.normalised() == ENUtoNED, "Rotation of quaternion fails!");

	// Test 7a: Factory function from quaternion
	constexpr auto maybeCustom_ENUtoNED1 = make_rotation<ENU, NED>(0., sqrt2 / 2, sqrt2 / 2, 0.);
	static_assert(
		!maybeCustom_ENUtoNED1.has_failure(),
		"Rotation factory function fails! - Pre-condition violation triggered despite no violation"
	);
	static_assert(
		maybeCustom_ENUtoNED1.value() == ENUtoNED,
		"Rotation factory function fails! - Unable to get value despite not an error"
	);

	// Test 7b: Factory function from yaw pitch roll
	constexpr auto maybeCustom_ENUtoNED2 = make_rotation<ENU, NED>(90., 0., 180.);
	static_assert(
		!maybeCustom_ENUtoNED2.has_failure(),
		"Rotation factory function fails! - Pre-condition violation triggered despite no violation"
	);
	static_assert(
		maybeCustom_ENUtoNED2.value() == ENUtoNED,
		"Rotation factory function fails! - YPR conversion to quaternion is incorrect!"
	);

	constexpr auto maybeCustom_NEDtoENU = make_rotation<NED, ENU>(90., 0., 180.);
	static_assert(
		!maybeCustom_NEDtoENU.has_failure(),
		"Rotation factory function fails! - Pre-condition violation triggered despite no violation"
	);
	static_assert(
		maybeCustom_NEDtoENU.value() == NEDtoENU,
		"Rotation factory function fails! - YPR conversion to quaternion is incorrect!"
	);

	// Test 7c: Factory function from yaw pitch roll - Test taken from
	// https://www.mathworks.com/help/aerotbx/ug/angle2quat.html
	constexpr auto check1 = make_rotation<ENU, NED>(rad2deg(0.7854), rad2deg(0.1), rad2deg(0.));
	constexpr auto check2 = make_rotation<ENU, NED>(0.9227, -0.0191, 0.0462, 0.3822);
	static_assert(!check1.has_failure());
	static_assert(is_close(check1.value().getW(), check2.value().getW(), 0.00001));
	static_assert(is_close(check1.value().getX(), check2.value().getX(), 0.0001));
	static_assert(is_close(check1.value().getY(), check2.value().getY(), 0.0001));
	static_assert(is_close(check1.value().getZ(), check2.value().getZ(), 0.00001));

	// TODO: Currently does not compile due to boost::system::error_category not being constexpr (cpp20 feature)
	// Test 8: Non-unit quaternion rejection
	// constexpr auto failed_rotation = make_rotation<ENU, NED>(0, 0, 0, 0);
	// static_assert(failed_rotation.as_failure(), "Rotation from ENU to NED should not compile!");
}

void test_rotation_conversions()
{
	// Test 1: Rotation conversion from quaternion to yaw pitch roll
	{
		constexpr auto yaw = 90.;
		constexpr auto pitch = 0.;
		constexpr auto roll = 180.0;

		constexpr auto rotation = make_rotation<ENU, NED>(yaw, pitch, roll).value();
		constexpr auto results = to_ypr_deg(rotation);
		static_assert(is_close(results.yaw, yaw), "Rotation conversion fails! - Yaw");
		static_assert(is_close(results.pitch, pitch), "Rotation conversion fails! - Pitch");
		static_assert(is_close(results.roll, roll), "Rotation conversion fails! - Roll");

		constexpr auto rotation2 = make_rotation<ENU, NED>(results.yaw, results.pitch, results.roll).value();
		static_assert(rotation == rotation2, "Rotation conversion fails!");
	}

	// Test 2: Rotation conversion from quaternion to yaw pitch roll
	{
		constexpr auto yaw2 = 180.;
		constexpr auto pitch2 = 0.;
		constexpr auto roll2 = 0.;

		constexpr auto r2 = make_rotation<ENU, NED>(yaw2, pitch2, roll2).value();
		constexpr auto results2 = to_ypr_deg(r2);
		static_assert(is_close(results2.yaw, yaw2), "Rotation conversion fails! - Yaw");
		static_assert(is_close(results2.pitch, pitch2), "Rotation conversion fails! - Pitch");
		static_assert(is_close(results2.roll, roll2), "Rotation conversion fails! - Roll");

		constexpr auto r2_b = make_rotation<ENU, NED>(results2.yaw, results2.pitch, results2.roll).value();
		static_assert(r2 == r2_b, "Rotation conversion fails!");
	}

	// Test 3: Rotation conversion from quaternion to yaw pitch roll
	{
		constexpr auto yaw = 179.;
		constexpr auto pitch = 0.;
		constexpr auto roll = 180.0;

		constexpr auto rotation = make_rotation<ENU, NED>(yaw, pitch, roll).value();
		constexpr auto results = to_ypr_deg(rotation);

		static_assert(is_close(results.yaw, yaw, 0.000001), "Rotation conversion fails! - Yaw");
		static_assert(is_close(results.pitch, pitch), "Rotation conversion fails! - Pitch");
		static_assert(is_close(results.roll, roll), "Rotation conversion fails! - Roll");

		constexpr auto rotation2 = make_rotation<ENU, NED>(results.yaw, results.pitch, results.roll).value();
		static_assert(rotation == rotation2, "Rotation conversion fails!");
	}
}

/**
 * @brief Compile-time unit tests for orientations
 *
 */
constexpr void test_orientations()
{
	// Test 1a: Change frame of position
	constexpr auto p1 = Position<odom> { 1, 2, 3 };
	constexpr auto p2 = Position<odom_ned> { 2, 1, -3 };
	constexpr auto rotation_result = ENUtoNED * p1;
	static_assert(rotation_result == p2, "Rotation of position fails!");

	constexpr auto rotation_result2 = FRDtoRDF * (NEDtoFRD * p2);
	static_assert(rotation_result2 == Position<Frame<World, RDF>> { 1, -3, 2 });

	// Test 2: Change frame of orientation
	constexpr auto ENU_Quaternion = make_rotation<ENU, FLU>(-103.123, 4.947, 1.791).value();
	constexpr auto NED_Quaternion = make_rotation<NED, FRD>(-166.877, -4.947, 1.792).value();

	constexpr auto NED_Quaternion_custom = NEDtoENU * ENU_Quaternion * FLUtoFRD;
	static_assert(std::is_same_v<decltype(NED_Quaternion), decltype(NED_Quaternion_custom)>);
	static_assert(is_close(NED_Quaternion.getW(), NED_Quaternion_custom.getW(), 0.000001));
	static_assert(is_close(NED_Quaternion.getX(), NED_Quaternion_custom.getX(), 0.00001));
	static_assert(is_close(NED_Quaternion.getY(), NED_Quaternion_custom.getY(), 0.00001));
	static_assert(is_close(NED_Quaternion.getZ(), NED_Quaternion_custom.getZ(), 0.000001));

	// TODO: The following doesn't compile due to lack of precision. Need to find a way to fix this.
	// static_assert(NED_Quaternion == NED_Quaternion_custom);
}

auto main() -> int
{
	// Run tests at compile time
	test_translation();
	test_position();
	test_rotations();
	test_orientations();

	// Run examples at compile-time

	std::cout << "ALL TESTS PASSED!" << std::endl;
	return 0;
}
}