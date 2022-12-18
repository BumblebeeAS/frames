#ifndef FRAME_UTIlS_GEOMETRY_ROTATION_H
#define FRAME_UTIlS_GEOMETRY_ROTATION_H

#include <type_traits>

#include "frames/coordinate_system/frame_convention.h"
#include "frames/geometry/error_code.h"
#include "frames/math.h"

namespace frame_utils::geometry
{

/**
 * @brief 3D Rotation with known frame conventions semantics
 *
 * The rotation defines conversion from one frame to another frame. Operations such as rotation composition are defined
 * if and only if the frame conventions matches.
 *
 * For example, ENUtoNED(NEDtoABC) is valid [returns ENUtoABC], but ENUtoNED(ENUtoABC) is not.
 *
 * @tparam FrameConvention1
 * @tparam FrameConvention2
 */
template<typename FrameConvention1, typename FrameConvention2>
class Rotation

{
	static_assert(
		coordinates::is_convention_v<FrameConvention1>, "Rotation must be created with a valid frame convention!"
	);
	static_assert(
		coordinates::is_convention_v<FrameConvention2>, "Rotation must be created with a valid frame convention!"
	);

private:
	double w = 1.0;
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;

public:
	constexpr Rotation(double w, double x, double y, double z) noexcept :
		w(w),
		x(x),
		y(y),
		z(z)
	{ }

	template<
		typename FC3, typename FC4, typename = std::enable_if_t<coordinates::is_convention_v<FC3>>,
		typename = std::enable_if_t<coordinates::is_convention_v<FC4>>>
	[[nodiscard]] explicit constexpr operator Rotation<FC3, FC4>() const noexcept
	{
		return { w, x, y, z };
	}

	[[nodiscard]] constexpr auto getW() const noexcept
	{
		return w;
	}

	[[nodiscard]] constexpr auto getX() const noexcept
	{
		return x;
	}

	[[nodiscard]] constexpr auto getY() const noexcept
	{
		return y;
	}

	[[nodiscard]] constexpr auto getZ() const noexcept
	{
		return z;
	}

	/**
	 * @brief Returns the squared norm of the quaternion
	 *
	 * @return double
	 */
	[[nodiscard]] constexpr auto norm2() const noexcept -> double
	{
		return w * w + x * x + y * y + z * z;
	}

	/**
	 * @brief Returns the norm of the quaternion
	 *
	 * @return double
	 */
	[[nodiscard]] constexpr auto norm() const noexcept -> double
	{
		return math::sqrt(this->norm2());
	}

	/**
	 * @brief Returns a normalised quaternion
	 *
	 * @return Rotation<FrameConvention1, FrameConvention2>
	 */
	[[nodiscard]] constexpr auto normalised() const noexcept -> Rotation<FrameConvention1, FrameConvention2>
	{
		return *this / norm();
	}

	/**
	 * @brief Returns the conjugate of the quaternion
	 *
	 * @return Rotation<FrameConvention1, FrameConvention2>
	 */
	[[nodiscard]] constexpr auto conjugate() const noexcept -> Rotation<FrameConvention1, FrameConvention2>
	{
		return { w, -x, -y, -z };
	}

	/**
	 * @brief Returns the inverse of the quaternion
	 *
	 * @return Rotation<FrameConvention2, FrameConvention1>
	 */
	[[nodiscard]] constexpr auto inverse() const noexcept -> Rotation<FrameConvention2, FrameConvention1>
	{
		return { -w, x, y, z };
	};

	/**
	 * @brief Returns the rotation that is the result of the product of the two given rotations
	 *
	 * Note that this uses the postfix notation. In particular, if R1 rotates from frame A to frame B, and R2 rotates
	 * from frame B to frame C, then R1(R2) rotates from frame A to frame C.
	 *
	 * @tparam FrameConvention3
	 * @tparam typename
	 * @param rot
	 * @return Rotation<FrameConvention1, FrameConvention3>
	 */
	template<typename FrameConvention3, typename = std::enable_if_t<coordinates::is_convention_v<FrameConvention3>>>
	[[nodiscard]] constexpr auto operator*(Rotation<FrameConvention2, FrameConvention3> other) const noexcept
		-> Rotation<FrameConvention1, FrameConvention3>
	{
		return { w * other.getW() - x * other.getX() - y * other.getY() - z * other.getZ(),
			     w * other.getX() + x * other.getW() + y * other.getZ() - z * other.getY(),
			     w * other.getY() - x * other.getZ() + y * other.getW() + z * other.getX(),
			     w * other.getZ() + x * other.getY() - y * other.getX() + z * other.getW() };
	}

	/**
	 * @brief Returns the rotation that is the result of the composition of the two given rotations
	 *
	 * Note that this uses the postfix notation. In particular, if R1 rotates from frame A to frame B, and R2 rotates
	 * from frame B to frame C, then R1(R2) rotates from frame A to frame C.
	 *
	 * @tparam FrameConvention3
	 * @tparam typename
	 * @param rot
	 * @return Rotation<FrameConvention1, FrameConvention3>
	 */
	template<typename FrameConvention3, typename = std::enable_if_t<coordinates::is_convention_v<FrameConvention3>>>
	[[nodiscard]] constexpr auto operator()(Rotation<FrameConvention2, FrameConvention3> rot) const noexcept
		-> Rotation<FrameConvention1, FrameConvention3>
	{
		return *this * rot;
	}

	/**
	 * @brief Returns the scaled quaternion
	 *
	 * @param scale
	 * @return Rotation
	 */
	[[nodiscard]] constexpr auto operator/(double scale) const noexcept -> Rotation
	{
		return { (1.0 / scale) * w, (1.0 / scale) * x, (1.0 / scale) * y, (1.0 / scale) * z };
	}

	/**
	 * @brief Returns true if the two rotations are equal
	 *
	 * @tparam FC3
	 * @tparam FC4
	 * @param other
	 * @return bool
	 */
	template<typename FC3, typename FC4>
	[[nodiscard]] constexpr auto operator==(Rotation<FC3, FC4> other) const noexcept -> bool
	{
		return (std::is_same_v<FrameConvention1, FC3> && std::is_same_v<FrameConvention2, FC4>) &&(
			(math::is_close(w, other.getW(), std::numeric_limits<float>::epsilon()) &&
		     math::is_close(x, other.getX(), std::numeric_limits<float>::epsilon()) &&
		     math::is_close(y, other.getY(), std::numeric_limits<float>::epsilon()) &&
		     math::is_close(z, other.getZ(), std::numeric_limits<float>::epsilon())) ||
			(math::is_close(w, -other.getW(), std::numeric_limits<float>::epsilon()) &&
		     math::is_close(x, -other.getX(), std::numeric_limits<float>::epsilon()) &&
		     math::is_close(y, -other.getY(), std::numeric_limits<float>::epsilon()) &&
		     math::is_close(z, -other.getZ(), std::numeric_limits<float>::epsilon()))
		);
	}

	/**
	 * @brief Returns true if the two rotations are not equal
	 *
	 * @tparam FC3
	 * @tparam FC4
	 * @param other
	 * @return bool
	 */
	template<typename FC3, typename FC4>
	[[nodiscard]] constexpr auto operator!=(Rotation<FC3, FC4> other) const noexcept -> bool
	{
		return !(*this == other);
	}

	friend auto operator<<(std::ostream& os, Rotation rot) -> std::ostream&
	{
		return os << "Rotation(" << rot.w << ", " << rot.x << ", " << rot.y << ", " << rot.z << ")";
	}
};

/**
 * @brief Returns the scaled quaternion
 *
 * @tparam FrameConvention1
 * @tparam FrameConvention2
 * @param scale
 * @param rot
 * @return Rotation<FrameConvention1, FrameConvention2>
 */
template<typename FrameConvention1, typename FrameConvention2>
[[nodiscard]] constexpr auto operator*(double scale, Rotation<FrameConvention1, FrameConvention2> rot) noexcept
	-> Rotation<FrameConvention1, FrameConvention2>
{
	return { scale * rot.getW(), scale * rot.getX(), scale * rot.getY(), scale * rot.getZ() };
}

static constexpr auto sqrt2 = math::sqrt(2);

/**
 * @brief Pre-defined conversions. Use this when you want to convert between two standard frames.
 *
 */
constexpr auto Identity =
	Rotation<coordinates::__ArbitraryConvention, coordinates::__ArbitraryConvention> { 1, 0, 0, 0 };
constexpr auto ENUtoNED = Rotation<coordinates::ENU, coordinates::NED> { 0, sqrt2 / 2, sqrt2 / 2, 0 };
constexpr auto NEDtoENU = ENUtoNED.inverse();

constexpr auto ENUtoFLU = Rotation<coordinates::ENU, coordinates::FLU> { 1, 0, 0, 0 };
constexpr auto FLUtoENU = ENUtoFLU.inverse();

constexpr auto NEDtoFRD = Rotation<coordinates::NED, coordinates::FRD> { 1, 0, 0, 0 };
constexpr auto FRDtoNED = NEDtoFRD.inverse();

constexpr auto FLUtoFRD = Rotation<coordinates::FLU, coordinates::FRD> { 0, 1, 0, 0 };
constexpr auto FRDtoFLU = FLUtoFRD.inverse();

constexpr auto FRDtoRDF = Rotation<coordinates::FRD, coordinates::RDF> { 0.5, 0.5, 0.5, 0.5 };
constexpr auto RDFtoFRD = FRDtoRDF.inverse();
}

#endif // FRAME_UTILS_COORDINATES_GEOMETRY_ROTATION_HPP