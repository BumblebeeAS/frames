#ifndef FRAME_UTILS_MATH_H
#define FRAME_UTILS_MATH_H

#include <limits>
#include <type_traits>

/**
 * @brief Compile-time math because <cmath> doesn't provide this.
 * Note: Not all basic arithmetic operations are implemented
 *
 * Inspired by:
 * https://github.com/elbeno/constexpr/blob/master/src/include/cx_math.h
 *
 * Modifications made:
 * 1. Stylistic changes
 * 2. Remove exceptions
 * 3. Approximations when necessary to avoid recursion depth limit
 */

namespace frame_utils::math::detail
{
constexpr auto EULER_NUM = 2.71828182845904523536l;
constexpr auto PI = 3.1415926535897932385l;

template<typename T>
constexpr auto pi() -> T
{
	return static_cast<T>(PI);
}

template<typename T>
auto constexpr e() -> T
{
	return static_cast<T>(EULER_NUM);
}

auto constexpr sqrtNewtonRaphson(double x, double curr, double prev) -> double;

template<typename T>
constexpr auto exp(T x, T sum, T n, int i, T t) -> T;

template<typename T>
constexpr auto trig_series(T x, T sum, T n, int i, int s, T t) -> T;

template<typename T>
constexpr auto asin_series(T x, T sum, int n, T t) -> T;

template<typename T>
constexpr auto atan_sum(T x, T sum, int n) -> T;
} // namespace frame_utils::math::detail

namespace frame_utils::math
{
/**
 * @brief Floating-point radian to degree
 *
 */
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr auto rad2deg(T x) -> T
{
	return x * 180 / detail::pi<T>();
}

/**
 * @brief Integral radian to degree
 *
 */
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr auto rad2deg(T x) -> double
{
	return x * 180.0 / detail::pi<double>();
}

/**
 * @brief Floating-point degree to radian
 *
 * @tparam T
 * @tparam typename
 * @param x
 * @return T
 */
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr auto deg2rad(T x) -> T
{
	return x * detail::pi<T>() / 180;
}

/**
 * @brief Integral degree to radian
 *
 */
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr auto deg2rad(T x) -> double
{
	return x * detail::pi<double>() / 180.0;
}

/**
 * @brief Absolute function
 *
 */
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto constexpr abs(T x) -> T
{
	return x >= 0 ? x : x < 0 ? -x : x; // Last case is for nan
}

/**
 * @brief Comparison function
 *
 */
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto constexpr is_close(T x, T y) -> bool
{
	return abs(x - y) <= std::numeric_limits<T>::epsilon();
}

/**
 * @brief Comparison function with user-provided precision
 *
 */
template<typename T, typename Q, typename = std::enable_if_t<std::is_floating_point_v<T> && std::is_floating_point_v<Q>>>
auto constexpr is_close(T x, T y, Q epsilon) -> bool
{
	return abs(x - y) <= epsilon;
}

/**
 * @brief Floating-point sqrt function
 *
 * @param x
 * @return double
 */

template<typename T>
auto constexpr sqrt(T x) -> double
{
	return x >= 0 && x < std::numeric_limits<double>::infinity() ? detail::sqrtNewtonRaphson(x, x, 0)
	                                                             : std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief Floating-point exponent function
 *
 * @param x
 * @return T
 */
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr auto exp(T x) -> T
{
	return detail::exp(x, T { 1 }, T { 1 }, 2, x);
}

/**
 * @brief Integral exponent function
 *
 * @param x
 * @return double
 */
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr auto exp(T x) -> double
{
	return detail::exp<double>(x, 1.0, 1.0, 2, x);
}

/**
 * @brief Floating-point sine function
 *
 * @param x
 * @return constexpr T
 */
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr auto sin(T x) -> T
{
	return detail::trig_series(x, x, T { 6 }, 4, -1, x * x * x);
}

/**
 * @brief Integral sine function
 *
 * @param x
 * @return constexpr double
 */
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr auto sin(T x) -> double
{
	return sin<double>(x);
}

/**
 * @brief Floating-point cosine function
 *
 * @param x
 * @return constexpr T
 */
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr auto cos(T x) -> T
{
	return detail::trig_series(x, T { 1 }, T { 2 }, 3, -1, x * x);
}

/**
 * @brief Integral cosine function
 *
 * @param x
 * @return constexpr double
 */
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr auto cos(T x) -> double
{
	return detail::trig_series<double>(x, 1.0, 2.0, 3, -1, static_cast<double>(x) * static_cast<double>(x));
}

/**
 * @brief Floating-point tangent function
 *
 * Note that this function does not check for domain errors e.g. if cos(x) = 0
 * @param x
 * @return constexpr T
 */
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr auto tan(T x) -> T
{
	return sin(x) / cos(x);
}

/**
 * @brief Integral tangent function
 *
 * Note that this function does not check for domain errors e.g. if cos(x) = 0
 * @param x
 * @return constexpr double
 */
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr auto tan(T x) -> double
{
	return sin(x) / cos(x);
}

/**
 * @brief Floating-point sine inverse function
 *
 * Note that this function does not check for domain errors e.g. UB if asin(x), x > 1
 *
 * @param x
 * @return T
 */
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr auto asin(T x) -> T
{
	return x == T { -1 }  ? detail::pi<double>() / T { -2 }
	       : x == T { 1 } ? detail::pi<double>() / T { 2 }
	                      : detail::asin_series(x, x, 1, x * x * x / T { 2 });
}

/**
 * @brief Integral sine inverse function
 *
 * Note that this function does not check for domain errors e.g. UB if asin(x), x > 1
 *
 * @param x
 * @return double
 */
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr auto asin(T x) -> double
{
	return asin<double>(x);
}

/**
 * @brief Floating-point cosine inverse function
 *
 * Note that this function does not check for domain errors e.g. UB if acos(x), x > 1
 *
 * @param x
 * @return T
 */
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr auto acos(T x) -> T
{
	return x == T { -1 }  ? static_cast<T>(detail::pi<double>())
	       : x == T { 1 } ? 0
	                      : detail::pi<double>() / T { 2 } - asin(x);
}

/**
 * @brief Integral cosine inverse function
 *
 * Note that this function does not check for domain errors e.g. UB if acos(x), x > 1
 *
 * @param x
 * @return double
 */
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr auto acos(T x) -> double
{
	return acos<double>(x);
}

/**
 * @brief Floating-point tangent inverse function
 *
 * Note that this function implements an approximation-based algorithm
 * Source: https://blasingame.engr.tamu.edu/z_zCourse_Archive/P620_18C/P620_zReference/PDF_Txt_Hst_Apr_Cmp_(1955).pdf
 *
 * @param x
 * @return T
 */
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr auto atan(T x) -> T
{
	constexpr auto a1 = 0.9999993329;
	constexpr auto a3 = -0.3332985605;
	constexpr auto a5 = 0.1994653599;
	constexpr auto a7 = -0.1390853351;
	constexpr auto a9 = 0.0964200441;
	constexpr auto a11 = -0.0559098861;
	constexpr auto a13 = 0.0218612288;
	constexpr auto a15 = -0.0040540580;

	// Approximation-based
	const auto x1 = x;
	const auto x3 = x * x * x;
	const auto x5 = x * x * x3;
	const auto x7 = x * x * x5;
	const auto x9 = x * x * x7;
	const auto x11 = x * x * x9;
	const auto x13 = x * x * x11;
	const auto x15 = x * x * x13;

	return a1 * x1 + a3 * x3 + a5 * x5 + a7 * x7 + a9 * x9 + a11 * x11 + a13 * x13 + a15 * x15;

	// Original implementation:
	// return x / (T { 1 } + x * x) * detail::atan_sum(x, T { 1 }, 1);
}

/**
 * @brief Integral tangent inverse function
 *
 * @param x
 * @return T
 */
template<typename Integral>
constexpr auto atan(Integral x, typename std::enable_if<std::is_integral<Integral>::value>::type* = nullptr) -> double
{
	return atan<double>(x);
}

/**
 * @brief Floating-point atan2 function
 *
 * @param x
 * @param y
 * @return T
 */
template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
constexpr auto atan2(T numerator, T denominator) -> T
{
	const auto y = numerator;
	const auto x = denominator;
	return is_close(x, 0.0) ? (is_close(y, 0.0) ? std::numeric_limits<double>::signaling_NaN()
	                           : y > 0.0        ? static_cast<T>(detail::pi<T>() / 2.0l)
	                                            : -static_cast<T>(detail::pi<T>() / 2.0l))
	                        : (x > 0.0                        ? atan(y / x)
	                           : (is_close(y, 0.0) || y >= 0) ? atan(y / x) + detail::pi<T>()
	                                                          : atan(y / x) - detail::pi<T>());
}

/**
 * @brief Integral atan2 function
 *
 * @param x
 * @param y
 * @return T
 */
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr auto atan2(T numerator, T denominator) -> double
{
	return atan2(static_cast<double>(numerator), static_cast<double>(denominator));
}

} // namespace frame_utils::math

namespace frame_utils::math::detail
{
auto constexpr sqrtNewtonRaphson(double x, double curr, double prev) -> double
{
	return is_close(curr, prev) ? curr : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
}

template<typename T>
constexpr auto exp(T x, T sum, T n, int i, T t) -> T
{
	return is_close(sum, sum + t / n) ? sum : exp(x, sum + t / n, n * i, i + 1, t * x);
}

template<typename T>
constexpr auto trig_series(T x, T sum, T n, int i, int s, T t) -> T
{
	return is_close(sum, sum + t * s / n) ? sum
	                                      : trig_series(x, sum + t * s / n, n * i * (i + 1), i + 2, -s, t * x * x);
}

template<typename T>
constexpr auto atan_term(T x2, int k) -> T
{
	return (T { 2 } * static_cast<T>(k) * x2) / ((T { 2 } * static_cast<T>(k) + T { 1 }) * (T { 1 } + x2));
}

template<typename T>
constexpr auto atan_product(T x, int k) -> T
{
	return k == 1 ? atan_term(x * x, k) : atan_term(x * x, k) * atan_product(x, k - 1);
}

template<typename T>
constexpr auto atan_sum(T x, T sum, int n) -> T
{
	return sum + atan_product(x, n) == sum ? sum : atan_sum(x, sum + atan_product(x, n), n + 1);
}

template<typename T>
constexpr auto asin_series(T x, T sum, int n, T t) -> T
{
	return is_close(sum, sum + t * static_cast<T>(n) / (n + 2), std::numeric_limits<T>::epsilon())
	           ? sum
	           : asin_series(x, sum + t * static_cast<T>(n) / (n + 2), n + 2, t * x * x * static_cast<T>(n) / (n + 3));
}

} // namespace frame_utils::math::detail

namespace frame_utils::math::test
{
constexpr auto E_D = frame_utils::math::detail::e<double>();
constexpr auto E2_D = E_D * E_D;
constexpr auto TWO_I = 2;
constexpr auto TWO_D = 2.0;
constexpr auto TWO_F = static_cast<float>(2.0);

// Test 1: Exponent function
static_assert(
	is_close(exp(1), E_D, std::numeric_limits<float>::epsilon()), "exp(1) differs from pre-computed euler number!"
);

static_assert(
	is_close(exp(TWO_I), E2_D, std::numeric_limits<float>::epsilon()),
	"exp(2) differs from pre-computed euler number squared!"
);
static_assert(
	is_close(exp(TWO_D), E2_D, std::numeric_limits<float>::epsilon()),
	"exp(2.0) differs from pre-computed euler number squared!"
);

// Test 2: sqrt function
static_assert(is_close(sqrt(TWO_D), sqrt(TWO_I)), "sqrt((double) 2) is not equal to sqrt((int) 2)");
static_assert(is_close(sqrt(TWO_F), sqrt(TWO_I)), "sqrt((float) 2) is not equal to sqrt((int) 2)");
static_assert(is_close(sqrt(TWO_D), sqrt(TWO_F)), "sqrt((double) 2) is not equal to sqrt((float) 2)");
static_assert(
	is_close(sqrt(exp(TWO_D)), E_D, std::numeric_limits<float>::epsilon()),
	"sqrt(exp(2.0)) differs from pre-computed euler number!"
);

} // namespace frame_utils::math::test

#endif // FRAME_UTILS_MATH_H