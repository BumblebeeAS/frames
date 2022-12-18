#ifndef FRAME_UTIlS_COORDINATE_SYSTEM_GEODETIC_GEODETIC_H
#define FRAME_UTIlS_COORDINATE_SYSTEM_GEODETIC_GEODETIC_H

#include <type_traits>

#include "frames/coordinate_system/frame.h"
#include "frames/coordinate_system/geodetic/constants.h"
#include "frames/coordinate_system/origin.h"
#include "frames/geometry/position.h"
#include "frames/math.h"

namespace frame_utils::coordinates::geodetic
{
/**
 * @brief Geodetic coordinates. Everything is in degrees
 *
 */
class Geodetic
{
private:
	double lat;
	double lon;
	double alt;

public:
	constexpr Geodetic(double lat, double lon, double alt) :
		lat(lat),
		lon(lon),
		alt(alt)
	{ }

	[[nodiscard]] constexpr auto getLat() const noexcept
	{
		return lat;
	}

	[[nodiscard]] constexpr auto getLon() const noexcept
	{
		return lon;
	}

	[[nodiscard]] constexpr auto getAlt() const noexcept
	{
		return alt;
	}
};
} // namespace frame_utils::coordinates::geodetic

/**
 * @brief Functions to convert between Geodetic, ECEF, and ENU coordinate system
 *
 */
namespace frame_utils::coordinates::geodetic::details
{
using namespace math;

static constexpr auto computePrimeVerticalRadiusOfCurvature(const double lat)
{
	using namespace constants::wgs84;
	return a / math::sqrt(1 - e_sq * math::sin(lat) * math::sin(lat));
}

/**
 * @brief Convert from geodetic to ECEF coordinates
 * Source: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
 *
 * @param geodetic
 * @return Position<ecef>
 */
constexpr auto toEcef(Geodetic geodetic) -> geometry::Position<coordinates::ecef>
{
	using namespace constants::wgs84;
	const auto phi = deg2rad(geodetic.getLat());
	const auto lambda = deg2rad(geodetic.getLon());
	const auto alt = geodetic.getAlt();

	const auto N = computePrimeVerticalRadiusOfCurvature(phi);

	return { (N + alt) * cos(phi) * cos(lambda), (N + alt) * cos(phi) * sin(lambda),
		     ((1 - e_sq) * N + alt) * sin(phi) };
}

/**
 * @brief Convert from ECEF to geodetic coordinates
 *
 * Source of this approach is lost through time... But it seems to be a simplification of the Newton-Raphson
 * approach with one iteration
 * @param ecef
 * @return Geodetic
 */
constexpr auto toGeodetic(geometry::Position<ecef> ecef) -> Geodetic
{
	using namespace constants::wgs84;
	const auto x = ecef.getX();
	const auto y = ecef.getY();
	const auto z = ecef.getZ();

	constexpr auto eps = e_sq / (1.0 - e_sq);
	const auto p = sqrt(x * x + y * y);
	const auto q = math::atan2((z * a), (p * b));
	const auto sin_q = sin(q);
	const auto cos_q = cos(q);
	const auto sin_q_3 = sin_q * sin_q * sin_q;
	const auto cos_q_3 = cos_q * cos_q * cos_q;
	const auto phi = atan2((z + eps * b * sin_q_3), (p - e_sq * a * cos_q_3));
	const auto lambda = atan2(y, x);
	const auto N = computePrimeVerticalRadiusOfCurvature(phi);

	return { rad2deg(phi), rad2deg(lambda), (p / cos(phi)) - N };
}

/**
 * @brief Convert from odom to ecef coordinates
 *
 * @param coords in Odom frame
 * @param reference
 * @return Position<ecef>
 */
constexpr auto toEcef(geometry::Position<odom> coords, Geodetic reference) -> geometry::Position<ecef>
{
	const auto xEast = coords.getX();
	const auto yNorth = coords.getY();
	const auto zUp = coords.getZ();

	const auto phi = deg2rad(reference.getLat());
	const auto lambda = deg2rad(reference.getLon());

	const auto reference_ecef = toEcef(reference);
	const auto x0 = reference_ecef.getX();
	const auto y0 = reference_ecef.getY();
	const auto z0 = reference_ecef.getZ();

	const auto sin_lambda = sin(lambda);
	const auto cos_lambda = cos(lambda);
	const auto cos_phi = cos(phi);
	const auto sin_phi = sin(phi);

	const auto xd = -sin_lambda * xEast - sin_phi * cos_lambda * yNorth + cos_phi * cos_lambda * zUp;
	const auto yd = cos_lambda * xEast - sin_phi * sin_lambda * yNorth + cos_phi * sin_lambda * zUp;
	const auto zd = cos_phi * yNorth + sin_phi * zUp;

	return { xd + x0, yd + y0, zd + z0 };
}

/**
 * @brief Convert from ecef to odom coordinates
 *
 * Source: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
 * @param coords in ECEF frame
 * @param reference
 * @return Position<Odom>
 */
constexpr auto toEnu(geometry::Position<ecef> coords, Geodetic reference) -> geometry::Position<odom>
{
	const auto x = coords.getX();
	const auto y = coords.getY();
	const auto z = coords.getZ();

	const auto reference_ecef = toEcef(reference);
	const auto x0 = reference_ecef.getX();
	const auto y0 = reference_ecef.getY();
	const auto z0 = reference_ecef.getZ();

	const auto phi = deg2rad(reference.getLat());
	const auto lambda = deg2rad(reference.getLon());

	const auto sin_lambda = sin(lambda);
	const auto cos_lambda = cos(lambda);
	const auto cos_phi = cos(phi);
	const auto sin_phi = sin(phi);

	const auto xd = x - x0;
	const auto yd = y - y0;
	const auto zd = z - z0;

	const auto xEast = -sin_lambda * xd + cos_lambda * yd;
	const auto yNorth = -sin_phi * cos_lambda * xd + -sin_phi * sin_lambda * yd + cos_phi * zd;
	const auto zUp = cos_phi * cos_lambda * xd + cos_phi * sin_lambda * yd + sin_phi * zd;

	return { xEast, yNorth, zUp };
}
} // namespace frame_utils::coordinates::geodetic::details

#endif