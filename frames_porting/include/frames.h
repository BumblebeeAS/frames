#ifndef FRAME_UTILS_H
#define FRAME_UTILS_H

#include "conversions/rotation.h"
#include "conversions/unsafe_conversions.h"
#include "coordinate_system/frame.h"
#include "coordinate_system/frame_convention.h"
#include "coordinate_system/geodetic/geodetic.h"
#include "geometry/position.h"
#include "geometry/rotation.h"
#include "math.h"

namespace frame_utils::coordinates {
using geometry::Position;
using geometry::Rotation;

/**
 * @brief Rotate position using a quaternion
 *
 *
 * @param rot Quaternion to rotate position by
 * @param pos The position to rotate
 * @return Position<Frame2>
 */
template <
    typename Origin, typename FC1, typename FC2, typename = std::enable_if_t<is_origin_v<Origin>>,
    typename = std::enable_if_t<is_convention_v<FC1>>, typename = std::enable_if_t<is_convention_v<FC2>>>
[[nodiscard]] constexpr auto operator*(Rotation<FC1, FC2> rot, Position<Frame<Origin, FC1>> pos) noexcept
    -> Position<Frame<Origin, FC2>> {
    const auto results = (static_cast<Rotation<__ArbitraryConvention, __ArbitraryConvention>>(rot.conjugate())(Rotation<__ArbitraryConvention, __ArbitraryConvention>{0, pos.getX(), pos.getY(), pos.getZ()}))(static_cast<Rotation<__ArbitraryConvention, __ArbitraryConvention>>(rot));

    return Position<Frame<Origin, FC2>>{results.getX(), results.getY(), results.getZ()};
}
}  // namespace frame_utils::coordinates

namespace frame_utils::coordinates::geodetic {

/**
 * @brief Functions relating to coordinate system changes
 *
 * Note: Functions going from Geodetic -> ENU/NED are not recommended due to
 * high error rate (>= 0.2m) from lost of precision of Geodetic coordinates
 */

constexpr auto toNed(const Position<odom>& enu_coords) -> Position<odom_ned> {
    return geometry::ENUtoNED * enu_coords;
};

constexpr auto toEnu(const Position<odom_ned>& ned_coords) -> Position<odom> {
    return geometry::NEDtoENU * ned_coords;
}

constexpr auto toGeodetic(const Position<odom>& enu, const Geodetic& reference) -> Geodetic {
    return geodetic::details::toGeodetic(geodetic::details::toEcef(enu, reference));
}

constexpr auto toGeodetic(const Position<odom_ned>& ned, const Geodetic& reference) -> Geodetic {
    return toGeodetic(toEnu(ned), reference);
}

constexpr auto toEnu(const Geodetic& geodetic, const Geodetic& reference) -> Position<odom> {
    return details::toEnu(details::toEcef(geodetic), reference);
}

constexpr auto toNed(const Geodetic& geodetic, const Geodetic& reference) -> Position<odom_ned> {
    return toNed(toEnu(geodetic, reference));
}
}  // namespace frame_utils::coordinates::geodetic

#endif
