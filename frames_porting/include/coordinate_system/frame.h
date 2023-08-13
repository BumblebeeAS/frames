#ifndef FRAME_UTIlS_COORDINATE_SYSTEM_FRAME_H
#define FRAME_UTIlS_COORDINATE_SYSTEM_FRAME_H

#include "coordinate_system/frame_convention.h"
#include "coordinate_system/origin.h"

namespace frame_utils::coordinates {
/**
 * @brief Concept: Frame
 * A coordinate frame is defined by an origin and a frame convention. Frame
 * thus takes in two template arguments, corresponding to origin and frame
 * convention respectively.
 *
 * ROS REP 105 (https://www.ros.org/reps/rep-0105.html#coordinate-frames)
 * defines several frames:
 * 1. odom: A frame with World as its origin and ENU as its convention
 * 2. base_link: A frame with Vehicle as its origin and FLU as its
 *               convention
 *
 * Furthermore, ROS REP 103
 * (https://www.ros.org/reps/rep-0103.html#coordinate-frame-conventions)
 * provides standard namings for certain special frames
 * 1. odom_ned: A frame with World as its origin and NED as its convention
 * 2. base_link_ned: A frame with Vehicle as its origin and FRD as its
 *                   convention
 * 3. <camera>_optical: A frame with <camera> as its origin and RDF as its
 *                      convention
 * @tparam Origin
 * @tparam FrameConvention
 */
template <typename Origin, typename FrameConvention>
struct Frame {
    static_assert(is_origin_v<Origin>, "Frame must be created with a valid origin!");
    static_assert(is_convention_v<FrameConvention>, "Frame must be created with a valid frame convention!");
};

using odom = Frame<World, ENU>;
using odom_ned = Frame<World, NED>;
using base_link = Frame<Vehicle, FLU>;
using base_link_ned = Frame<Vehicle, FRD>;
using ecef = Frame<Earth, ECEF>;

template <typename>
struct is_frame : std::false_type {};

template <typename Origin, typename FrameConvention>
struct is_frame<Frame<Origin, FrameConvention>> : std::true_type {};

template <typename T>
constexpr bool is_frame_v = is_frame<T>::value;
}  // namespace frame_utils::coordinates
#endif