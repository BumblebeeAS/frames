# `frames` - A Frames Library for C++

`frames` is a templated **C++17** compile-time library that provides compile-time analysis of origins, frame conventions, and frames.

Features:

- Compile-time checks of frames
- Tested against ROS TF (`rosrun tf tf_echo`)
- Easy to use

This library is *technically* a header-only library i.e. the `include` folder is sufficient for all the functionalities. However, compile-time unit tests have been added for correctness assurance.

## Table of Contents

- [`frames` - A Frames Library for C++](#frames---a-frames-library-for-c)
  - [Table of Contents](#table-of-contents)
  - [A Small Example](#a-small-example)
  - [Coordinate System Constructs](#coordinate-system-constructs)
    - [FrameConvention](#frameconvention)
    - [Origin](#origin)
    - [Frame](#frame)
  - [Geometric Constructs](#geometric-constructs)
    - [Rotation](#rotation)
    - [Translation](#translation)
    - [Position](#position)
  - [Rotation Convention](#rotation-convention)
  - [Other Utilities](#other-utilities)

## A Small Example

Here is a small example of a possible operation

```cpp
void example_1()
{
 // Example 1: Change coordinate system of a position vector

 using namespace frame_utils::coordinates;
 using namespace frame_utils::geometry;

 // Defines a point in ENU coordinates, with odom as origin
 constexpr auto some_point_in_enu = Position<odom> { 1, 2, 3 };

 // Converts the ENU coordinates to NED coordinates
 constexpr auto some_point_in_ned = ENUtoNED * some_point_in_enu;

 // Check that this fits our intuition i.e. swap x and y, and negate z. Note the different coordinate systems
 static_assert(some_point_in_ned == Position<odom_ned> { 2, 1, -3 });

 // Compile-error: Incorrect values
 // static_assert(some_point_in_ned == Position<odom_ned> { 1, 2, 3 });

 // Compile-error: different coordinate_system given; operator== is not defined for different coordinate systems
 // static_assert(some_point_in_ned == Position<odom> { 2, 1, -3 });

 return 0;
}
```

The `examples` folder contains more information as to how this library can be used.

## Coordinate System Constructs

To properly define coordinate systems, the following concepts are defined:

1. `FrameConvention`
2. `Origin`
3. `Frame`

### FrameConvention

A `FrameConvention` provides semantics for `x`, `y`, `z` e.g. under North-East-Down (NED), the `x`-axis points to north, `y` to east, and `z` to down. The following `FrameConvention` are supported and ready-for-use:

1. `ENU`: East-North-Up. For use with world-fixed frames
2. `NED`: North-East-Down. For use with world-fixed frames
3. `FLU`: Front-Left-Up, the body frame counter-part of ENU
4. `FRD`: Front-Right-Down, the body frame counter-part of NED
5. `RDF`: Right-Down-Front. For use with optical frames

The above are in line with [ROS REP 105](https://www.ros.org/reps/rep-0105.html#coordinate-frames).

Additional `FrameConvention` can be defined for your application:

```cpp
struct SomeNewConvention : FrameConvention 
{ }
```

### Origin

An `Origin` represents the mathematical origin of an arbitrary coordinate system. It defines the meaning of `(0, 0, 0)`. The following `Origin` are supported and ready-for-use:

1. `World`: The origin used for `ENU` and `NED` frames
2. `Vehicle`: The origin used by `base_link` and `base_link_ned` frames
3. `Earth`: The origin used by `ecef` frame

The above are in line with [ROS REP 105](https://www.ros.org/reps/rep-0105.html#coordinate-frames).

Additional `Origin` can be defined for your application:

```cpp
struct RightCameraOrigin : Origin
{} 
```

### Frame

A `Frame` is defined with respect to an origin and a frame convention. The following `Origin` are supported and ready-for-use:

1. `odom`: A frame with `woWorldrld` as `Origin` and `ENU` as `FrameConvention`
2. `odom_ned`: A frame with `World` as `Origin` and `NED` as `FrameConvention`
3. `base_link`:  A frame with `Vehicle` as `Origin` and `ENU` as `FrameConvention`
4. `base_link_ned`:  A frame with `Vehicle` as `Origin` and `NED` as `FrameConvention`
5. `ecef`:  A frame with `Earth` as `Origin` and `ECEF` as `FrameConvention`

The above are in line with [ROS REP 105](https://www.ros.org/reps/rep-0105.html#coordinate-frames).

To assist with conversions, some pre-defined constants are available:

```cpp
const auto in_enu = Position<odom>{1, 2, 3};
const auto in_ned = ENUtoNED * in_enu; 
```

The constants defined are `ENUtoNED`, `NEDtoENU`, `ENUtoFLU`, `FLUtoENU`, `NEDtoFRD`, `FRDtoNED`, `FLUtoFRD`, `FRDtoFLU`,  `FLUtoRDF`, `RDFtoFRD` and `FRDtoRDF`.

Note that `Geodetic` i.e. Latitude, Longitude, Altitude, is technically not a `Position` as it does not have any `x`, `y`, `z` components. Thus, a `Geodetic` class is provided, with conversion functions to assist:

```cpp
constexpr auto toGeodetic(const Position<odom>& enu, const Geodetic& reference) -> Geodetic;
constexpr auto toGeodetic(const Position<odom_ned>& ned, const Geodetic& reference) -> Geodetic;
constexpr auto toEnu(const Geodetic& geodetic, const Geodetic& reference) -> Position<odom>;
constexpr auto toNed(const Geodetic& geodetic, const Geodetic& reference) -> Position<odom_ned>;
```

## Geometric Constructs

There are 3 geometric constructs currently provided: `Rotation`, `Translation`, and `Position`.

### Rotation

A `Rotation` defines the rotation from one frame to another. The quaternion representation is used due to it being compact and the absence of singularities.

Note that only unit quaternions truly represent orientation. Hence, please use the `make_rotation` function as it will return a unit quaternion if possible, and a failure message otherwise.

```cpp
constexpr auto make_rotation(double w, double x, double y, double z) noexcept
 -> boost::outcome_v2::result<Rotation<FrameConvention1, FrameConvention2>>;
constexpr auto make_rotation(double yaw, double pitch, double roll)
 -> boost::outcome_v2::result<Rotation<FrameConvention1, FrameConvention2>>;
constexpr auto make_rotation(const YPR_Deg& rpy)
 -> boost::outcome_v2::result<Rotation<FrameConvention1, FrameConvention2>>;
constexpr auto make_rotation(const YPR_Rad& rpy)
 -> boost::outcome_v2::result<Rotation<FrameConvention1, FrameConvention2>>;

constexpr auto to_ypr_deg(const Rotation<FrameConvention1, FrameConvention2>& rotation)
 -> YPR_Deg;
constexpr auto to_ypr_rad(const Rotation<FrameConvention1, FrameConvention2>& rotation)
-> YPR_Rad;
```

Notes:

1. For `make_rotation`, the return type is `boost::outcome_v2::result<Rotation<FrameConvention1, FrameConvention2>>`. Refer to `example_4` for how to use it.
2. Overload 2 of `make_rotation` which takes 3 parameters takes them in degrees.
3. Yaw Pitch Roll intrinsic rotation is used, contrary to [ROS REP 103](https://www.ros.org/reps/rep-0103.html#rotation-representation) preference. Refer to the [Rotation Convention](#rotation-convention) section for the explanation.

### Translation

A `Translation` defines how to move a `Frame` to another position as its new `Origin`, with no change in `FrameConvention`. Because of these restrictions, most operations are defined only among `Translation` of the same `Frame` and target `Origin`. The exceptions are the `inverse` free function, which changes the direction of translation, and an overload of `operator+` which enables joining translations of different origins:

```cpp
 // Joining translations together
 constexpr auto a = Translation<Frame<World, ENU>, Vehicle> { 1, 2, 3 };
 constexpr auto b = Translation<Frame<Vehicle, ENU>, __ArbitraryOrigin> { 2, 3, 4 };
 constexpr auto c = Translation<Frame<World, ENU>, __ArbitraryOrigin> { 3, 5, 7 };
 static_assert(a + b == c);

// Changing direction
 static_assert(inverse(a) == Translation<Frame<Vehicle, ENU>, World> { -1, -2, -3 });
```

### Position

A `Position` is just a translation towards an unknown origin `__ArbitraryOrigin`. This essentially removes type check for the destination. Thus `Position` should be used only when the target origin of the translation is not needed. This is usually the case when describing position of objects (hence the name `Position`).

## Rotation Convention

[ROS REP 103](https://www.ros.org/reps/rep-0103.html#rotation-representation) defines the preferred rotation representation in ROS:

1. Quaternion (Most Preferred)
2. Rotation Matrix
3. Fixed-axis roll, pitch, yaw [Extrinsic Rotation]
4. Euler angles yaw, pitch, roll [Intrinsic Rotation]

The difference between the fixed-axis (extrinsic) rotation against the euler angles (intrinsic) rotations is in the axis used: for fixed-axis, the original axes before the rotation are used, whereas for euler angles, the intermediate axes are used.

Note that this means the rotation from ENU to NED can be described as follows:

1. Extrinsic rotation RPY: 180, 0, 90
2. Intrinsic rotation RPY: 180, 0, -90
3. Intrinsic rotation YPR: 90, 0, 180

All three conventions are correct rotations transforming ENU to NED, but only 1. and 3. have the same values. Since intrinsic rotation is easier to reason about, thus option 3 is chosen i.e. yaw first, followed by pitch, followed by roll, using intrinsic rotations. This is reflected in the ordering used across the code base i.e. `YPR`.

For a more detailed explanation, please refer to [this](https://dominicplein.medium.com/extrinsic-intrinsic-rotation-do-i-multiply-from-right-or-left-357c38c1abfd).

## Other Utilities

Due to the need for compile-time checks and the lack of `constexpr` support from the C++ Standard Library, several `constexpr` math functions are thus implemented:

1. Angle converions: `rad2deg`, `deg2rad`
2. Exponents: `exp`, `sqrt`
3. Trigonometric functions: `sin`, `cos`, `tan`
4. Inverse trig. functions: `asin`, `acos`, `atan`, `atan2`
5. Miscellaneous: `abs`, `is_close`

This can be separated out into a separate compile-time math library if need be. For now, however, the purpose is only to support the implementation of `frames`. All the math utility functions can be found in the `frame_utils::math` namespace.
