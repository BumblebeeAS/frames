#include "frames/conversions/rotation.h"
#include "frames/coordinate_system/origin.h"
#include "frames/frames.h"
#include "frames/geometry/position.h"
#include "frames/geometry/rotation.h"
#include "frames/geometry/translation.h"

/**
 * @brief A simple example to demonstrate how to define custom origins and frames
 *
 * @return int
 */
auto main() -> int
{
	using namespace frame_utils::coordinates;
	using namespace frame_utils::geometry;

	// Define a new origin and frames, corresponding to the right camera
	struct RightCameraOrigin : Origin
	{ };

	struct RightCameraFrame : Frame<RightCameraOrigin, FLU>
	{ };

	struct RightCameraOpticalFrame : Frame<RightCameraOrigin, RDF>
	{ };

	// Define another origin and frames, corresponding to the left camera
	struct LeftCameraOrigin : Origin
	{ };

	struct LeftCameraFrame : Frame<LeftCameraOrigin, FLU>
	{ };

	struct LeftCameraOpticalFrame : Frame<LeftCameraOrigin, RDF>
	{ };

	// Define relationships between the two cameras:
	// Right camera is located 0.1m in front of, 0.2m to the right, and 0.3m above the left camera
	constexpr auto left_to_right = Translation<Frame<LeftCameraOrigin, FLU>, RightCameraOrigin> { 0.1, -0.2, 0.3 };

	// Suppose an object is 2m to the left and 3m in front of the left camera
	constexpr auto some_object = Position<Frame<LeftCameraOrigin, FLU>> { 3, 2, 0 };

	// The object is thus 2.9m in front of, 2.2m to the right, and -0.3m above the right camera
	constexpr auto expected = Position<Frame<RightCameraOrigin, FLU>> { 2.9, 2.2, -0.3 };

	// Using the type system to guide us, we can easily compute the position of the object in the right camera frame
	constexpr auto from_right_cam = inverse(left_to_right) + some_object;
	static_assert(from_right_cam == expected);

	std::cout << "The object is " << from_right_cam << " from the right camera" << std::endl;
	return 0;
}