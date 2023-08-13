#pragma once

#include <type_traits>
#include <boost/system/error_code.hpp>

namespace frame_utils::geometry::error_code
{
enum class RotationConstructionErrc
{
	Success = 0,
	NonUnitQuaternion = 1
};

namespace detail
{
	class RotationConstructionErrc_category : public boost::system::error_category
	{
	public:
		[[nodiscard]] auto name() const noexcept -> const char* final
		{
			return "RotationConstructionError";
		}

		[[nodiscard]] auto message(int c) const -> std::string final
		{
			switch (static_cast<RotationConstructionErrc>(c)) {
			case RotationConstructionErrc::Success:
				return "rotation construction successful";
			case RotationConstructionErrc::NonUnitQuaternion:
				return "given quaternion is not a unit quaternion";
			default:
				return "unknown";
			}
		}
	};
}
}

namespace frame_utils::geometry::error_code
{
inline constexpr auto make_error_code(frame_utils::geometry::error_code::RotationConstructionErrc e)
{
	return boost::system::error_code(
		static_cast<int>(e), frame_utils::geometry::error_code::detail::RotationConstructionErrc_category()
	);
}
}

namespace boost::system
{
// Tell the C++ 11 STL metaprogramming that enum RotationConstructionErrc is registered with the boost error code
// system
template<>
struct is_error_code_enum<frame_utils::geometry::error_code::RotationConstructionErrc> : std::true_type
{ };
}