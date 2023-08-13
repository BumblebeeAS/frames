#pragma once

// #include <ostream>
// #include <type_traits>
// #include <boost/type_traits/is_detected.hpp>

#include "frames/geometry/translation.h"

namespace frame_utils::geometry {

// An alias: a position is a translation with an arbitrary final point
template <typename Frame>
using Position = Translation<Frame, coordinates::__ArbitraryOrigin>;

}  // namespace frame_utils::geometry