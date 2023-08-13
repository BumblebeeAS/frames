#ifndef FRAME_UTILS_COORDINATE_SYSTEM_ORIGIN_H
#define FRAME_UTILS_COORDINATE_SYSTEM_ORIGIN_H

#include <type_traits>

namespace frame_utils::coordinates
{

/**
 * @brief Concept: Origin
 * An origin defines the zero coordinates of a frame. ROS REP 105
 * (https://www.ros.org/reps/rep-0105.html#coordinate-frames) implies the
 * following origins:
 * 1. Vehicle: to be used by base_link frame
 * 2. World: to be used by odom frame and ecef frame
 * 3. Map: to be used by map frame
 */
struct Origin
{ };

struct Map : Origin
{ };

struct World : Origin
{ };

struct Earth : Origin
{ };

struct Vehicle : Origin
{ };

struct __ArbitraryOrigin : coordinates::Origin
{ };

template<typename T>
struct is_origin : std::is_base_of<Origin, T>

{ };

template<typename T>
constexpr bool is_origin_v = is_origin<T>::value;
}

#endif // FRAME_UTILS_COORDINATE_SYSTEM_ORIGIN_H