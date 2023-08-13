#ifndef FRAME_UTIlS_COORDINATE_SYSTEM_GEODETIC_CONSTANTS_H
#define FRAME_UTIlS_COORDINATE_SYSTEM_GEODETIC_CONSTANTS_H

namespace frame_utils::coordinates::geodetic::constants::wgs84
{
constexpr double a = 6378137.0; // WGS-84 Earth semimajor axis (m)

constexpr double b = 6356752.314245; // Derived Earth semiminor axis (m)
constexpr double f = (a - b) / a;    // Ellipsoid Flatness
constexpr double f_inv = 1.0 / f;    // Inverse flattening

constexpr double a_sq = a * a;
constexpr double b_sq = b * b;
constexpr double e_sq = f * (2 - f);

constexpr double R = a;
} // namespace frame_utils::coordinates::geodetic::constants::wgs84
#endif