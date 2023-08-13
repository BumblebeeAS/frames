#pragma once

#include <boost/type_traits/is_detected.hpp>
#include <ostream>
#include <type_traits>

#include "frames/coordinate_system/frame.h"
#include "frames/math.hpp"

namespace frame_utils::geometry {
/**
 * @brief 3D Vector with a known reference frame semantics
 * The vector has an origin, a frame convention, and a final point. Operations such as
 * addition are defined if and only if they have the same origin, frame convention, and final point.
 *
 * @tparam Frame
 */
template <typename Frame1, typename Origin2>
class Translation {
    static_assert(coordinates::is_frame_v<Frame1>, "Translation must be created with a valid frame!");
    static_assert(coordinates::is_origin_v<Origin2>, "Translation must be created with a valid origin!");

   private:
    double x{};
    double y{};
    double z{};

   public:
    constexpr Translation(double x, double y, double z) : x(x),
                                                          y(y),
                                                          z(z){};

    [[nodiscard]] constexpr auto getX() const noexcept {
        return x;
    }

    [[nodiscard]] constexpr auto getY() const noexcept {
        return y;
    }

    [[nodiscard]] constexpr auto getZ() const noexcept {
        return z;
    }

    /**
     * @brief Addition of two translations
     * @param other
     * @return
     */
    [[nodiscard]] constexpr auto operator+(const Translation& other) const -> Translation {
        return {this->getX() + other.getX(), this->getY() + other.getY(), this->getZ() + other.getZ()};
    }

    /**
     * @brief Subtraction of two translations
     * @param other
     * @return
     */
    [[nodiscard]] constexpr auto operator-(const Translation& other) const -> Translation {
        return {this->getX() - other.getX(), this->getY() - other.getY(), this->getZ() - other.getZ()};
    }

    /**
     * @brief Equality operator
     *
     */
    template <typename Frame2, typename Origin3>
    [[nodiscard]] constexpr auto operator==(const Translation<Frame2, Origin3>& other) const noexcept -> bool {
        return std::is_same_v<Frame1, Frame2> && std::is_same_v<Origin2, Origin3> &&
               (math::is_close(x, other.x, std::numeric_limits<float>::epsilon()) &&
                math::is_close(y, other.y, std::numeric_limits<float>::epsilon()) &&
                math::is_close(z, other.z, std::numeric_limits<float>::epsilon()));
    }

    /**
     * @brief Inequality operator
     *
     */
    template <typename Frame2, typename Origin3>
    [[nodiscard]] constexpr auto operator!=(const Translation& other) const noexcept -> bool {
        return !(*this == other);
    }

    friend auto operator<<(std::ostream& os, Translation t) -> std::ostream& {
        return os << "Translation(" << t.x << ", " << t.y << ", " << t.z << ")";
    }

    constexpr auto inverse();
};

/**
 * @brief Scaling of a translation
 *
 */
template <typename Frame, typename Origin>
[[nodiscard]] constexpr auto operator*(double scale, const Translation<Frame, Origin>& t) -> Translation<Frame, Origin> {
    return {t.getX() * scale, t.getY() * scale, t.getZ() * scale};
}

/**
 * @brief Addition of two translations with different origins
 *
 */
template <typename Origin1, typename Origin2, typename Origin3, typename FrameConvention>
[[nodiscard]] constexpr auto operator+(
    const Translation<coordinates::Frame<Origin1, FrameConvention>, Origin2>& t1,
    const Translation<coordinates::Frame<Origin2, FrameConvention>, Origin3>& t2) -> Translation<coordinates::Frame<Origin1, FrameConvention>, Origin3> {
    static_assert(coordinates::is_origin_v<Origin1>);
    static_assert(coordinates::is_origin_v<Origin2>);
    static_assert(coordinates::is_origin_v<Origin3>);
    static_assert(coordinates::is_convention_v<FrameConvention>);

    return {t1.getX() + t2.getX(), t1.getY() + t2.getY(), t1.getZ() + t2.getZ()};
}

/**
 * @brief Inverse of a translation changes the direction.
 *
 */
template <typename OriginA, typename OriginB, typename FrameConvention>
[[nodiscard]] constexpr auto inverse(const Translation<coordinates::Frame<OriginA, FrameConvention>, OriginB>& t)
    -> Translation<coordinates::Frame<OriginB, FrameConvention>, OriginA> {
    return {-t.getX(), -t.getY(), -t.getZ()};
}

}  // namespace frame_utils::geometry