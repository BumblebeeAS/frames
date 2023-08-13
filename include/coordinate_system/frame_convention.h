#ifndef FRAME_UTIlS_COORDINATE_SYSTEM_FRAME_CONVENTION_H
#define FRAME_UTIlS_COORDINATE_SYSTEM_FRAME_CONVENTION_H
#include <type_traits>

namespace frame_utils::coordinates
{

/**
 * @brief Concept: FrameConvention
 * A frame convention defines the semantics of the axis of a coordinate
 * system. Supported conventions currently are
 *  1. ENU: East-North-Up
 *  2. NED: North-East-Down
 *  3. FRD: Front-Right-Down
 *  4. FLU: Front-Left-Up
 *  5. RDF: Right-Down-Front
 *
 * Note that ROS conventions are ENU and FLU
 * (https://www.ros.org/reps/rep-0103.html#coordinate-frame-conventions)
 *
 * Gazebo conventions is FLU
 * (https://answers.gazebosim.org//question/8497/standard-units-of-measure-and-coordinate-conventions-in-gazebo/)
 *
 * OpenCV conventions is RDF
 * (https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
 */
struct FrameConvention
{ };

struct ENU : FrameConvention
{ };

struct NED : FrameConvention
{ };

struct FRD : FrameConvention
{ };

struct FLU : FrameConvention
{ };

struct RDF : FrameConvention
{ };

struct ECEF : FrameConvention
{ };

struct __ArbitraryConvention : FrameConvention
{ };

template<typename T>
struct is_convention : std::is_base_of<FrameConvention, T>
{ };

template<typename T>
constexpr bool is_convention_v = is_convention<T>::value;

}
#endif