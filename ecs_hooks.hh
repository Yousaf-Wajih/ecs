#pragma once

// You can change these functions to do something upon various events in the
// `Ecs`.

namespace ecs {
class Ecs;

namespace hooks {

template <typename T> inline void on_register_component(Ecs &) {}
template <typename T> inline void on_add_resource(Ecs &) {}

} // namespace hooks
} // namespace ecs
