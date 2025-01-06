/// \file ecs.hh
/// The whole implementation of the ECS in a single header file.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <functional>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <queue>
#include <ranges>
#include <stack>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "ecs_hooks.hh"

#ifndef ECS_COMPONENT_TYPE
#define ECS_COMPONENT_TYPE uint8_t
#endif

#ifndef ECS_MAX_COMPONENTS
#define ECS_MAX_COMPONENTS 128
#endif

namespace ecs {

/// Indicates that a query must have the component `T`. The query will only
/// contain entities that have `T`. This is better than using a reference to `T`
/// in the query if access to the component data is not required.
/// \tparam T The component that must be present.
template <typename T> struct With {};

/// Indicates that a query must not have the component `T`. The query will not
/// contain any entities that have `T`.
/// \tparam T The component that must not be present.
template <typename T> struct Without {};

/// Used in a query to check for the presence of a component `T`. It can be cast
/// to a boolean value. This is better than using a pointer to `T` in the query
/// if access to the component data is not required.
/// \tparam T The component to check.
template <typename T> struct Has {
  explicit Has(bool value) : value(value) {}
  explicit operator bool() const { return value; }

  bool value;
};

/// An `Entity` is just an ID.
using Entity = uint64_t;

/// The type for a component type ID. It can be changed by defining
/// `ECS_COMPONENT_TYPE`. By default, it is `uint8_t`.
using ComponentType = ECS_COMPONENT_TYPE;

/// THe maximum number of types of components that can be registered. It can be
/// changed by defining `ECS_MAX_COMPONENTS`. By default, it is 128.
constexpr ComponentType MAX_COMPONENTS = ECS_MAX_COMPONENTS;

/// The priority of a message logged by the ECS.
enum class LogPriority : uint8_t { Verbose, Info, Warning, Error };

/// The type of function callback used by the ECS for logging.
using LogFn = std::function<void(LogPriority, const std::string &)>;

class Ecs;

namespace detail {

#ifdef __GNUC__
#include <cxxabi.h>
inline std::string demangle_typename(const char *mangled) {
  char *name = abi::__cxa_demangle(mangled, nullptr, nullptr, nullptr);
  if (!name) return mangled;
  std::string ret(name);
  free(name);
  return ret;
}
#else
inline std::string demangle_typename(const char *mangled) { return mangled; }
#endif

template <typename T> inline std::string type_name() {
  return demangle_typename(typeid(T).name());
}

inline size_t thread_count() {
  return std::max(std::thread::hardware_concurrency(), 1u);
}

template <typename... Ts> struct is_contained;

template <typename... Ts>
inline constexpr bool is_contained_v = is_contained<Ts...>::value;

template <typename T> struct is_contained<T> : std::false_type {};

template <typename T, typename Head, typename... Tail>
struct is_contained<T, Head, Tail...>
    : std::bool_constant<std::is_same_v<T, Head> ||
                         (is_contained_v<T, Tail> || ...)> {};

template <typename... Ts> struct is_unique;

template <typename... Ts> constexpr bool is_unique_v = is_unique<Ts...>::value;

template <> struct is_unique<> : std::true_type {};

template <typename Head, typename... Tail>
struct is_unique<Head, Tail...>
    : std::bool_constant<!is_contained_v<Head, Tail...> &&
                         is_unique_v<Tail...>> {};

template <typename T, template <typename...> typename U>
struct is_specialization_of : std::false_type {};

template <typename... Ts, template <typename...> typename U>
struct is_specialization_of<U<Ts...>, U> : std::true_type {};

template <typename T, template <typename...> typename U>
constexpr bool is_specialization_of_v = is_specialization_of<T, U>::value;

template <typename T, template <typename...> typename U>
struct remove_specialization_of {
  using type = T;
};

template <typename T, template <typename...> typename U>
struct remove_specialization_of<U<T>, U> {
  using type = T;
};

template <typename T, template <typename...> typename U>
using remove_specialization_of_t = remove_specialization_of<T, U>::type;

using unique_void_ptr = std::unique_ptr<void, void (*)(const void *)>;

template <typename T> auto unique_void(T *ptr) {
  return unique_void_ptr(
      ptr, [](const void *data) { delete static_cast<const T *>(data); });
}

template <typename T, typename... Args>
requires std::constructible_from<T, Args...>
auto make_unique_void(Args &&...args) {
  return unique_void(new T(std::forward<Args>(args)...));
}

class Log {
public:
  Log(const LogFn &fn) : fn(fn ? fn : [](auto, auto) {}) {}

  template <typename... Ts>
  void verbose(fmt::format_string<Ts...> str, Ts &&...args) const {
    fn(LogPriority::Verbose, fmt::format(str, std::forward<Ts>(args)...));
  }

  template <typename... Ts>
  void info(fmt::format_string<Ts...> str, Ts &&...args) const {
    fn(LogPriority::Info, fmt::format(str, std::forward<Ts>(args)...));
  }

  template <typename... Ts>
  void warn(fmt::format_string<Ts...> str, Ts &&...args) const {
    fn(LogPriority::Warning, fmt::format(str, std::forward<Ts>(args)...));
  }

  template <typename... Ts>
  void error(fmt::format_string<Ts...> str, Ts &&...args) const {
    fn(LogPriority::Error, fmt::format(str, std::forward<Ts>(args)...));
  }

  template <typename E = std::logic_error, typename... Ts>
  void fatal(fmt::format_string<Ts...> str, Ts &&...args) const {
    auto msg = fmt::format(str, std::forward<Ts>(args)...);
    fn(LogPriority::Error, msg);
    throw E(msg);
  }

private:
  LogFn fn;
};

struct ComponentInterface {
  void (*move)(uint8_t *, const uint8_t *);
  void (*destruct)(uint8_t *);
  size_t size;
  std::string name;
};

using ArchetypeId = std::bitset<MAX_COMPONENTS>;
struct Archetype {
  ArchetypeId id;
  std::vector<Entity> entities;
  std::array<std::unique_ptr<uint8_t[]>, MAX_COMPONENTS> component_data;
  std::array<std::size_t, MAX_COMPONENTS> component_data_sizes;
};

struct Record {
  Archetype *archetype;
  size_t index;
};

template <typename T, typename... Args>
concept Package = requires(Ecs &ecs, Args &&...args) {
  T::setup(ecs, std::forward<Args>(args)...);
};

template <typename T>
concept QueryArg = std::is_same_v<T, Entity> || std::is_reference_v<T> ||
                   std::is_pointer_v<T> || is_specialization_of_v<T, With> ||
                   is_specialization_of_v<T, Without> ||
                   is_specialization_of_v<T, Has>;

template <typename... Ts>
concept QueryArgs = is_unique_v<Ts...> && (QueryArg<Ts> && ...);

template <QueryArg T> struct SetIdValue;
template <QueryArg T> struct QueryArgValueType;
template <QueryArg T> struct GetQueryArg;
template <QueryArg T> struct GetQueryValue;

} // namespace detail

template <detail::QueryArgs... Ts> class Query;
template <typename T> struct Res;
class Commands;

namespace detail {

template <typename T> class DependencyGraph {
public:
  void add_node(const T &node) {
    if (!adj_list.contains(node)) {
      adj_list[node] = {};
      in_degree[node] = 0;
    }
  }

  void add_edge(const T &src, const T &dst) {
    add_node(src);
    add_node(dst);

    if (std::ranges::find(adj_list[src], dst) != adj_list[src].end()) {
      return;
    }

    adj_list[src].push_back(dst);
    in_degree[dst]++;
  }

  const auto &get_adj_list() const { return adj_list; }
  const auto &get_in_degree() const { return in_degree; }

  bool are_connected(const T &a, const T &b) const {
    if (!adj_list.contains(a) || !adj_list.contains(b)) return false;

    std::unordered_set<T> visited;
    std::stack<T> stack;
    stack.push(a);

    while (!stack.empty()) {
      T current = stack.top();
      stack.pop();

      if (current == b) return true;

      if (visited.contains(current)) continue;
      visited.insert(current);

      if (adj_list.contains(current)) {
        for (const auto &neighbor : adj_list.at(current)) {
          stack.push(neighbor);
        }
      }
    }

    return false;
  }

  bool is_acyclic() const {
    std::queue<T> to_visit;
    auto in_degree_copy = in_degree;

    for (const auto &[node, degree] : in_degree_copy) {
      if (degree == 0) to_visit.push(node);
    }

    size_t visited_count = 0;
    while (!to_visit.empty()) {
      T current = to_visit.front();
      to_visit.pop();
      visited_count++;

      for (const auto &neighbor : adj_list.at(current)) {
        if (--in_degree_copy[neighbor] == 0) to_visit.push(neighbor);
      }
    }

    return visited_count == adj_list.size();
  }

private:
  std::unordered_map<T, std::vector<T>> adj_list;
  std::unordered_map<T, size_t> in_degree;
};

using SystemDependencyGraph = DependencyGraph<uint64_t>;

class ISystem {
public:
  virtual ~ISystem() = default;
  virtual void run(Ecs &ecs) = 0;
  virtual void setup(Ecs &ecs) = 0;

  bool on_main_thread{};
  std::vector<size_t> dependents{};
  std::vector<std::function<bool(Ecs &)>> conditions;
};

template <typename T>
concept SystemArg =
    (std::is_reference_v<T> && std::is_const_v<std::remove_reference_t<T>> &&
     is_specialization_of_v<std::remove_cvref_t<T>, Query>) ||
    is_specialization_of_v<T, Res> || std::is_same_v<T, Commands>;

template <typename... Ts>
concept SystemArgs = is_unique_v<Ts...> && (SystemArg<Ts> && ...);

template <typename Fn>
concept SystemDependencyRegisterer = std::invocable<Fn, std::type_index, bool>;

template <typename Fn> concept SystemCondition = std::invocable<Fn, Ecs &>;

template <SystemArg T> struct GetSystemDependency;

template <SystemArg T> struct GetSystemArg {
  static auto get(Ecs &ecs) { return T{ecs}; }
};

template <SystemArgs... Args> class System;
class ExclusiveSystem;

template <typename Fn>
concept SystemFn = requires(Fn fn) {
  System(std::function(fn), [](auto, auto) {});
};

template <typename Fn>
concept ExclusiveSystemFn = requires(Fn fn) {
  ExclusiveSystem(std::function(fn), [](auto, auto) {});
};

template <typename Fn>
concept AnySystemFn = SystemFn<Fn> || ExclusiveSystemFn<Fn>;

template <AnySystemFn T> inline uint64_t get_system_id(T &&v);

struct SystemLayer {
  struct ResourceDependees {
    std::unordered_set<uint64_t> readers, writers;
  };

  std::vector<std::unique_ptr<ISystem>> systems;
  std::unordered_map<uint64_t, size_t> system_indices;

  std::unordered_map<size_t, std::vector<uint64_t>> system_groups;
  std::unordered_map<std::type_index, ResourceDependees> resource_dependencies;
  struct {
    std::unordered_map<uint64_t, std::vector<size_t>> after, before;
  } group_dependencies;

  SystemDependencyGraph dependency_graph;
  std::vector<size_t> dependency_counts;
  std::vector<std::atomic_size_t> current_dependency_counts;
  std::vector<bool> conditions_fulfilled;
};

template <SystemArgs... Args> class System : public ISystem {
public:
  System(std::function<void(Args...)> &&fn,
         SystemDependencyRegisterer auto &&dep_registerer);

  virtual void run(Ecs &ecs) override { fn(GetSystemArg<Args>::get(ecs)...); }

  virtual void setup(Ecs &ecs) override { (system_setup_arg<Args>(ecs), ...); }

  template <SystemArg T> void system_setup_arg(Ecs &ecs);

private:
  std::function<void(Args...)> fn;
};

class ExclusiveSystem : public ISystem {
public:
  ExclusiveSystem(std::function<void(Ecs &)> &&fn,
                  SystemDependencyRegisterer auto &&dep_registerer);

  virtual void run(Ecs &ecs) override { fn(ecs); }

  virtual void setup(Ecs &) override { on_main_thread = true; }

private:
  std::function<void(Ecs &)> fn;
};

struct Resource {
  explicit Resource(unique_void_ptr ptr) : ptr(std::move(ptr)) {}

  unique_void_ptr ptr;
  bool read_main_thread_only{}, write_main_thread_only{};
};

} // namespace detail

template <bool Layered = true> class SystemHandle {
  friend class ecs::Ecs;

public:
  /// Specifies whether the systems must be run on the main thread.
  /// \param value Whether to run on the main thread or not.
  SystemHandle &on_main_thread(bool value = true) {
    for (auto [id, sys] : systems) { sys->on_main_thread = value; }
    return *this;
  }

  /// Adds a condition to the systems. A condition is a callable which takes a
  /// reference to `Ecs` and returns a boolean.
  /// \param fn The condition function.
  SystemHandle &run_if(detail::SystemCondition auto &&fn) {
    for (auto [id, sys] : systems) { sys->conditions.push_back(fn); }
    return *this;
  }

  /// Makes the system a part of the given group.
  /// \param group The group index.
  SystemHandle &in_group(size_t group) requires Layered {
    for (auto [id, sys] : systems) { layer.system_groups[group].push_back(id); }
    return *this;
  }

  /// Adds a dependency to run the systems represented by the handle after the
  /// given system.
  /// \param fn The system to run after.
  SystemHandle &after(detail::SystemFn auto &&fn) requires Layered {
    for (auto [id, sys] : systems) {
      layer.dependency_graph.add_edge(detail::get_system_id(fn), id);
    }

    return *this;
  }

  /// Adds a dependency to run the systems represented by the handle before the
  /// given system.
  /// \param fn The system to run before.
  SystemHandle &before(detail::SystemFn auto &&fn) requires Layered {
    for (auto [id, sys] : systems) {
      layer.dependency_graph.add_edge(id, get_system_id(fn));
    }

    return *this;
  }

  /// Adds a dependency to run the systems represented by the handle after the
  /// given group.
  /// \param group The group to run after.
  SystemHandle &after(size_t group) requires Layered {
    for (auto [id, sys] : systems) {
      layer.group_dependencies.after[id].push_back(group);
    }

    return *this;
  }

  /// Adds a dependency to run the systems represented by the handle before the
  /// given group.
  /// \param group The group to run before.
  SystemHandle &before(size_t group) requires Layered {
    for (auto [id, sys] : systems) {
      layer.group_dependencies.before[id].push_back(group);
    }

    return *this;
  }

private:
  using LayerType =
      std::conditional_t<Layered, detail::SystemLayer &, std::monostate>;

  SystemHandle(LayerType layer) requires Layered : layer(layer) {}
  SystemHandle() requires(!Layered) {}

  SystemHandle(const SystemHandle &) = delete;
  SystemHandle(SystemHandle &&) = default;

  LayerType layer;
  std::vector<std::pair<uint64_t, detail::ISystem *>> systems{};
};

template <typename T> class ResourceHandle {
  friend class ecs::Ecs;

public:
  /// Gives a reference to the resource value.
  /// \return A reference to the stored resource.
  T &get_value() { return *static_cast<T *>(resource.ptr.get()); }

  /// Indicates that a resource may only be accessed on the main thread. Any
  /// systems that access this will be forced to run on the main thread.
  /// \param read Whether the resource must be read on the main thread (false by
  /// default).
  /// \param write Whether the resource must be written to on the main thread
  /// (true by default).
  ResourceHandle &main_thread_only(bool read = false, bool write = true) {
    resource.read_main_thread_only = read;
    resource.write_main_thread_only = write;
    return *this;
  }

private:
  ResourceHandle(detail::Resource &resource) : resource(resource) {}

  detail::Resource &resource;
};

class EntityHandle {
  friend class EntityRange;

public:
  /// Returns the ID of the entity.
  /// \return The entity ID.
  auto id() { return entity; }

  /// Gives a view to all of the components of this entity in a type-erased
  /// manner. The view consists of tuples for each component in the form:
  /// `(type_index, name, void_ptr_to_component_data)`
  auto get_components();

private:
  explicit EntityHandle(Ecs &ecs, Entity e) : ecs(ecs), entity(e) {}

  Ecs &ecs;
  Entity entity;
};

class EntityRange {
  friend class Ecs;

  struct Iterator {
    using iterator_category = std::input_iterator_tag;
    using value_type = EntityHandle;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type *;
    using reference = value_type &;

    const EntityRange &range;
    size_t archetype_index, entity_index;

    Iterator(EntityRange &range, size_t archetype_index,
             size_t entity_index = 0)
        : range(range), archetype_index(archetype_index),
          entity_index(entity_index) {}

    value_type operator*() const {
      return EntityHandle(
          range.ecs, range.entity_arrays[archetype_index].second[entity_index]);
    }

    Iterator &operator++() {
      if (++entity_index >= range.entity_arrays[archetype_index].first) {
        entity_index = 0;
        archetype_index++;
      }

      return *this;
    }

    Iterator operator++(int) {
      Iterator temp = *this;
      ++(*this);
      return temp;
    }

    bool operator==(const Iterator &other) const {
      return &range == &other.range &&
             archetype_index == other.archetype_index &&
             entity_index == other.entity_index;
    }
  };

public:
  Iterator begin();
  Iterator end();

private:
  EntityRange(Ecs &ecs);

  Ecs &ecs;
  std::vector<std::pair<size_t, Entity *>> entity_arrays;
  std::vector<detail::Archetype *> archetypes;
};

/// The main class for the ECS.
class Ecs {
  template <detail::SystemArgs...> friend class detail::System;
  template <detail::SystemArg> friend struct detail::GetSystemArg;
  template <detail::QueryArgs...> friend class Query;
  friend class Commands;
  friend class EntityRange;
  friend class EntityHandle;

public:
  /// The constructor takes an optional logging callback. If none is provided,
  /// an empty no-op one will be used. The ECS thread pool is also created in
  /// the constructor.
  /// \param log_fn The logging callback.
  Ecs(const LogFn &log_fn = {}) : log(log_fn) {
    thread_pool.reserve(detail::thread_count());
    for (size_t i = 0; i < detail::thread_count(); i++) {
      thread_pool.push_back(std::thread([this] {
        while (!done) system_worker(false);
      }));
    }
  }

  /// The destructor cleans up the thread pool.
  ~Ecs() {
    {
      std::lock_guard lock(layer_mutex);
      done = true;
      layer_cv.notify_all();
    }

    for (auto &thread : thread_pool) thread.join();
  }

  /// Adds a "package" to the `Ecs`. A package is any `struct` containing a
  /// static function `setup` that takes a reference to an `Ecs` instance along
  /// with any number of custom arguments. `add_package<P>(args...)` is
  /// basically syntax sugar for `P::setup(ecs, args...)`.
  /// \tparam P The type of the package to add.
  /// \tparam Args... The types of the additional arguments to be given to the
  /// package setup function.
  /// \param args The additional arguments to be given to the package setup
  /// function.
  template <typename P, typename... Args> requires detail::Package<P, Args...>
  void add_package(Args &&...args) {
    P::setup(*this, std::forward<Args>(args)...);
  }

  /// Returns the current state of the ECS.
  /// \return The current state.
  const std::string &get_state() const { return current_state; }

  /// Sets the names of all possible states in the ECS. This can only be called
  /// once to initialize all states. It should not be called with no arguments,
  /// as that has no effect.
  /// \param state_names The names of all of the states as values convertible
  /// to `std::string`. None of these should be empty or "Null".
  void
  set_states(std::convertible_to<const std::string &> auto &&...state_names) {
    if constexpr (sizeof...(state_names) == 0) {
      log.warn("Setting empty states has no effect");
      return;
    }

    constexpr auto is_valid = [](const std::string &s) {
      return !s.empty() && s != "Null";
    };

    if ((!is_valid(state_names) && ...)) {
      log.fatal("Invalid state names");
      return;
    }

    if (!states.empty()) {
      log.error("Setting states multiple times is not supported!");
      return;
    }

    states = {
        "Null",
        std::string(std::forward<decltype(state_names)>(state_names))...,
    };

    next_state = std::get<0>(std::tuple(state_names...));
    current_state = "Null";
    state_changed = true;

    log.verbose("Starting in state {}", next_state);

    for (const auto &a : states) {
      for (const auto &b : states) {
        if (a == b) continue;

        register_layer("TransitionFrom" + a + "To" + b);
      }
    }
  }

  /// Queues a state change for the ECS. The actual state change occurs on the
  /// next `Ecs::run_systems()` call. Note that if more than one state change
  /// occurs before they get a chance to actually be executed, only the last one
  /// will be considered.
  /// \param state The state to change to.
  void change_state(const std::string &state) {
    if (state == current_state || state == next_state) return;

    if (state_changed && current_state != "Null") {
      log.warn("{} to {} change overriding previous state change ({} to {})",
               current_state, state, current_state, next_state);
    }

    state_changed = true;
    next_state = state;
  }

  /// Tells if a component is registered in the ECS.
  /// \tparam T The component to check.
  /// \return Whether the component is registered.
  template <typename T> bool is_component_registered() const {
    return component_types.contains(typeid(T));
  }

  /// Returns the `ComponentType` ID of a component.
  /// \tparam T The component whose type ID is needed.
  /// \return The type ID of the component.
  template <typename T> ComponentType get_component_type() const {
    if (!is_component_registered<T>()) {
      log.fatal("Attempted to access unregistered component {}",
                detail::type_name<T>());
    }

    return component_types.at(typeid(T));
  }

  /// Returns a pair of the name and type index of the component indicated by
  /// the given `ComponentType` ID.
  /// \param type The type ID of the component.
  /// \return The name and type index of the component.
  std::pair<std::string, std::type_index>
  get_component_type_info(ComponentType type) const {
    if (!component_type_indices.contains(type)) {
      log.fatal("Attempted to access unregistered component {}", type);
    }

    return {components[type].name, component_type_indices.at(type)};
  }

  /// Registers `T` as a component for use in the ECS.
  /// \tparam T The component to register.
  template <typename T> void register_component() {
    if (next_component_type + 1 >= MAX_COMPONENTS) {
      log.fatal("Attempted to register too many components");
    }

    if (is_component_registered<T>()) {
      log.error("Attempted to register already registered component {}",
                detail::type_name<T>());
      return;
    }

    ComponentType type = next_component_type++;
    component_types.emplace(typeid(T), type);
    component_type_indices.emplace(type, typeid(T));

    components[type] = detail::ComponentInterface{
        .move =
            [](uint8_t *dst, const uint8_t *src) {
              new (dst) T(std::move(*reinterpret_cast<const T *>(src)));
            },
        .destruct =
            [](uint8_t *data) {
              std::launder(reinterpret_cast<T *>(data))->~T();
            },
        .size = sizeof(T),
        .name = detail::type_name<T>(),
    };

    log.verbose("Registered component {}", detail::type_name<T>());

    hooks::on_register_component<T>(*this);
  }

  /// Registers `Ts...` as components for use in the ECS.
  /// \tparam Ts... The components to register.
  template <typename... Ts> void register_components() {
    (register_component<Ts>(), ...);
  }

  /// Tells if a given resource is present in the ECS.
  /// \tparam T The resource to check.
  /// \return Whether the resource is present.
  template <typename T> bool has_resource() const {
    return resources.contains(typeid(T));
  }

  /// Adds a resource to the ECS.
  /// \tparam T The type of the resource to add.
  /// \tparam Args... The types of the arguments to pass to the constructor of
  /// `T`
  /// \param args The arguments to pass to the constructor of `T`
  template <typename T, typename... Args>
  requires std::constructible_from<T, Args...>
  ResourceHandle<T> add_resource(Args &&...args) {
    if (has_resource<T>()) {
      log.error("Attempted to register already registered resource {}",
                detail::type_name<T>());

      return ResourceHandle<T>{resources.at(typeid(T))};
    }

    resources.emplace(typeid(T), detail::Resource{detail::make_unique_void<T>(
                                     std::forward<Args>(args)...)});

    log.verbose("Registered resource {}", detail::type_name<T>());

    hooks::on_add_resource<T>(*this);
    return ResourceHandle<T>{resources.at(typeid(T))};
  }

  /// Returns a pointer to the resource of type `T`.
  /// \tparam T The resource type to get.
  /// \return A pointer to the resource.
  template <typename T> T *get_resource() {
    return static_cast<T *>(get_resource_internal<T>().ptr.get());
  }

  /// Spawns an entity with the given components.
  /// \tparam Ts... The types of the components to add to the entity.
  /// \param components The values of the components to add to the entity.
  /// \return The ID of the entity.
  template <typename... Ts> Entity spawn(Ts &&...components) {
    Entity entity = next_entity++;
    records[entity] = detail::Record{};
    add_components(entity, std::move(components)...);
    return entity;
  }

  /// Removes all entities given by the provided range.
  /// \param entities A range of entity IDs.
  void remove_entities(std::ranges::range auto &&entities) {
    for (auto e : entities) remove_entity(e);
  }

  /// Removes the given entity from the ECS.
  /// \param entity The ID of the entity to remove.
  void remove_entity(Entity entity) {
    if (!records.contains(entity)) {
      log.error("Attempted to remove invalid entity {}", entity);
      return;
    }

    detail::Record &record = records[entity];
    detail::Archetype *archetype = record.archetype;
    if (!archetype) return;

    for (size_t i = 0; i < MAX_COMPONENTS; i++) {
      if (!archetype->id[i]) continue;

      detail::ComponentInterface *comp = &components[i];
      comp->destruct(&archetype->component_data[i][record.index * comp->size]);
    }

    remove_record(record, -1);
    records.erase(entity);
  }

  /// Adds the given components to the provided entity.
  /// \tparam Ts... The types of the components to add.
  /// \param entity The ID of the entity to which the components will be added.
  /// \param components The values of the components to add.
  template <typename... Ts>
  void add_components(Entity entity, Ts &&...components) {
    (add_component<Ts>(entity, std::move(components)), ...);
  }

  /// Adds the given component to the provided entity.
  /// \tparam T The type of the component to add.
  /// \tparam Args... The types of the arguments to `T`'s constructor.
  /// \param entity The ID of the entity to which the component will be added.
  /// \param args The arguments to `T`'s constructor.
  template <typename T, typename... Args>
  requires std::constructible_from<T, Args...>
  void add_component(Entity entity, Args &&...args) {
    if (!is_component_registered<T>()) {
      log.fatal("Attempted to add unregistered component {}",
                detail::type_name<T>());
    }

    ComponentType type = component_types.at(typeid(T));
    detail::Record &record = records[entity];
    detail::Archetype *old_archetype = record.archetype;
    detail::Archetype *new_archetype = nullptr;

    if (old_archetype) {
      if (old_archetype->id[type]) {
        log.error("Attempted to add component {} to entity {} but it is "
                  "already present",
                  detail::type_name<T>(), entity);
      }

      detail::ArchetypeId new_archetype_id = old_archetype->id;
      new_archetype_id.set(type);
      new_archetype = get_archetype(new_archetype_id);

      for (ComponentType i = 0; i < MAX_COMPONENTS; i++) {
        if (!new_archetype_id[i]) continue;

        auto [old_size, move_c] = move_archetype(new_archetype, old_archetype,
                                                 i, record.index);

        if (move_c) {
          new (&new_archetype->component_data[i][old_size])
              T(std::forward<Args>(args)...);
        }
      }

      remove_record(record, type);
    } else {
      detail::ArchetypeId new_archetype_id = {};
      new_archetype_id.set(type);
      new_archetype = get_archetype(new_archetype_id);

      size_t old_size = realloc_archetype_component_data(new_archetype, type);

      new (&new_archetype->component_data[type][old_size])
          T(std::forward<Args>(args)...);
    }

    record.archetype = new_archetype;
    record.index = new_archetype->entities.size();
    new_archetype->entities.push_back(entity);
  }

  /// Removes the given components from the provided entity.
  /// \tparam Ts... The types of the components to remove.
  /// \param entity The ID of the entity from which the components will be
  /// removed.
  template <typename... Ts> void remove_components(Entity entity) {
    (remove_component<Ts>(entity), ...);
  }

  /// Removes the given component from the provided entity.
  /// \tparam T The type of the component to remove.
  /// \param entity The ID of the entity from which the components will be
  /// removed.
  template <typename T> void remove_component(Entity entity) {
    remove_component(entity, typeid(T));
  }

  /// Removes the given component from the provided entity.
  /// \param entity The ID of the entity from which the components will be
  /// removed.
  /// \param type_id The type index of the component to remove.
  void remove_component(Entity entity, std::type_index type_id) {
    if (!component_types.contains(type_id)) {
      log.fatal("Attempted to remove unregistered component {}",
                detail::demangle_typename(type_id.name()));
    }

    ComponentType type = component_types.at(type_id);
    detail::Record &record = records[entity];
    detail::Archetype *old_archetype = record.archetype;
    detail::Archetype *new_archetype = nullptr;

    if (!old_archetype || !old_archetype->id[type]) {
      log.warn("Ignoring attempt to remove non-existent component {} from "
               "entity {}",
               detail::demangle_typename(type_id.name()), entity);

      return;
    }

    detail::ArchetypeId new_archetype_id = old_archetype->id;
    new_archetype_id.reset(type);

    remove_record(record, type);

    if (new_archetype_id == 0) {
      records[entity] = {};
      return;
    }

    new_archetype = get_archetype(new_archetype_id);

    for (size_t i = 0; i < MAX_COMPONENTS; i++) {
      if (!new_archetype_id[i]) continue;
      move_archetype(new_archetype, old_archetype, i, record.index);
    }

    record.archetype = new_archetype;
    record.index = new_archetype->entities.size();
    new_archetype->entities.push_back(entity);
  }

  /// Registers layers with the given names for use with the ECS systems.
  /// \param layer_names The names of the layers to register.
  void register_layers(
      std::convertible_to<const std::string &> auto &&...layer_names) {
    (register_layer(std::forward<decltype(layer_names)>(layer_names)), ...);
  }

  /// Registers a layer with the given name for use with the ECS systems.
  /// \param layer_name The name of the layer to register.
  void register_layer(const std::string &layer_name) {
    if (system_layers.contains(layer_name)) {
      log.warn("Attempted to register already-registered layer \"\"",
               layer_name);
      return;
    }

    log.verbose("Registered layer {}", layer_name);

    system_layers.emplace(layer_name, detail::SystemLayer{});
  }

  /// Registers the given systems to run on a state transition. See
  /// `Ecs::register_systems()` for details of a system.
  /// \param from The state from which a transition should occur to run the
  /// given systems, or `nullopt` to allow running on a transition from any
  /// state.
  /// \param to The state to which a transition should occur to run the given
  /// systems, or `nullopt` to allow running on a transition to any state.
  /// \param fn The systems to register.
  /// \return An unlayered handle to modify some properties of the system.
  auto register_systems_on_transition(std::optional<std::string> from,
                                      std::optional<std::string> to,
                                      detail::AnySystemFn auto &&...fn) {
    std::vector<std::string> layers;

    if (!from && !to) {
      for (const auto &a : states) {
        for (const auto &b : states) {
          if (a == b) continue;

          layers.push_back("TransitionFrom" + a + "To" + b);
        }
      }
    } else if (!from) {
      for (const auto &a : states) {
        if (a != *to) layers.push_back("TransitionFrom" + a + "To" + *to);
      }
    } else if (!to) {
      for (const auto &a : states) {
        if (a != *from) layers.push_back("TransitionFrom" + *from + "To" + a);
      }
    } else {
      layers.push_back("TransitionFrom" + *from + "To" + *to);
    }

    SystemHandle<false> handle;
    for (const auto &layer : layers) {
      do_register_systems(layer, handle, std::forward<decltype(fn)>(fn)...);
    }

    return handle;
  }

  /// Registers the given systems to run on the specified layer. A system is a
  /// function that matches one of the following:
  /// - An exclusive system: A function taking a reference to the `Ecs` as an
  /// argument and nothing else. It is never run in parallel with any other
  /// systems.
  /// - A normal system: A function taking any unique combination of valid
  /// system arguments. The system arguments may be the following:
  ///   - `const Query<Ts...> &`
  ///   - `Res<T>`
  ///   - `Commands`
  /// \param layer_name The name of the layer to register the systems on.
  /// \param fn The systems to register.
  /// \return A layered handle to modify properties of the system.
  auto register_systems(const std::string &layer_name,
                        detail::AnySystemFn auto &&...fn) {
    if (!system_layers.contains(layer_name)) {
      log.fatal("Attempted to access invalid layer \"{}\"", layer_name);
    }

    SystemHandle handle{system_layers.at(layer_name)};
    do_register_systems(layer_name, handle, fn...);
    return handle;
  }

  /// Prepare all systems for running. This MUST be called before any call to
  /// `Ecs::run_systems()` and no systems should be registered after a call to
  /// this.
  void prepare_systems() {
    auto start = std::chrono::high_resolution_clock::now();

    // OPTIMIZE: This could probably be made more efficient
    for (auto &[_, layer] : system_layers) {
      for (const auto &sys : layer.systems) sys->setup(*this);

      // User dependencies must be added before automatic dependencies.
      for (const auto &[id, groups] : layer.group_dependencies.after) {
        for (const auto group : groups) {
          for (const auto &other : layer.system_groups[group]) {
            layer.dependency_graph.add_edge(other, id);
          }
        }
      }

      for (const auto &[id, groups] : layer.group_dependencies.before) {
        for (const auto group : groups) {
          for (const auto &other : layer.system_groups[group]) {
            layer.dependency_graph.add_edge(id, other);
          }
        }
      }

      for (const auto &[res_id, dependees] : layer.resource_dependencies) {
        for (const auto &writer : dependees.writers) {
          for (const auto &other_writer : dependees.writers) {
            if (!layer.dependency_graph.are_connected(other_writer, writer)) {
              layer.dependency_graph.add_edge(writer, other_writer);
            }
          }

          for (const auto &reader : dependees.readers) {
            if (!layer.dependency_graph.are_connected(reader, writer)) {
              layer.dependency_graph.add_edge(writer, reader);
            }
          }
        }
      }

      if (!layer.dependency_graph.is_acyclic()) {
        log.fatal("Dependency graph has a cycle");
      }

      for (const auto &[id, deps] : layer.dependency_graph.get_adj_list()) {
        auto &system = layer.systems[layer.system_indices.at(id)];

        system->dependents.reserve(deps.size());
        for (const auto &id : deps) {
          system->dependents.push_back(layer.system_indices.at(id));
        }
      }

      layer.dependency_counts.resize(layer.systems.size());
      for (const auto &[id, count] : layer.dependency_graph.get_in_degree()) {
        layer.dependency_counts[layer.system_indices.at(id)] = count;
      }

      layer.current_dependency_counts = std::vector<std::atomic_size_t>(
          layer.systems.size());
      layer.conditions_fulfilled = std::vector<bool>(layer.systems.size());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start);

    log.info("prepare_systems done in {}", time);
  }

  /// Runs the systems on the specified layer. If there is a state transition
  /// queued, this also executes that first. That is, it recursively calls
  /// itself with "TransitionFromAToB" if a state change is queued from A to B.
  /// It also executes any queued commands at the end of the function. The
  /// systems are run in parallel using the thread pool where possible.
  /// \param layer_name The name of the layer to run.
  void run_systems(std::string layer_name) {
    if (state_changed && current_state != next_state) {
      state_changed = false;
      run_systems("TransitionFrom" + current_state + "To" + next_state);
      current_state = next_state;
    }

    if (!system_layers.contains(layer_name)) {
      log.fatal("Attempted to access invalid layer \"{}\"", layer_name);
    }

    auto &layer = system_layers[layer_name];

    size_t num_systems = layer.systems.size();
    for (size_t i = 0; i < layer.systems.size(); i++) {
      layer.conditions_fulfilled[i] = true;
      for (const auto &cond : layer.systems[i]->conditions) {
        if (!cond(*this)) {
          layer.conditions_fulfilled[i] = false;
          break;
        }
      }

      if (!layer.conditions_fulfilled[i]) {
        num_systems--;
        continue;
      }

      layer.current_dependency_counts[i] = layer.dependency_counts[i];
      if (layer.dependency_counts[i] == 0) ready_queue.push(i);
    }

    if (num_systems == 0) return;

    remaining_systems = num_systems;

    {
      std::lock_guard lock(layer_mutex);
      current_layer = &layer;
    }
    layer_cv.notify_all();

    while (remaining_systems.load() > 0) { system_worker(true); }

    {
      std::lock_guard lock(layer_mutex);
      current_layer = nullptr;
    }
    layer_cv.notify_all();

    run_commands();
  }

  /// Runs any queued commands by the ECS. It is also automatically called at
  /// the end of `Ecs::run_systems()`.
  void run_commands() {
    for (const auto &cmd : commands) { cmd(*this); }
    commands.clear();
  }

  /// Constructs a query with the specified args.
  /// \tparam QueryArgs... The arguments of the query.
  /// \return The constructed query.
  template <detail::QueryArgs... Args> auto query() {
    return Query<Args...>{*this};
  }

  /// Returns a range which allows looping over all entities and also accessing
  /// their components in a type-erased manner.
  /// \return The entity range.
  auto get_all_entities() { return EntityRange(*this); }

  /// Returns a range which allows looping over all resources.
  /// \return The resource range.
  auto get_all_resources() {
    return resources | std::views::transform([](const auto &pair) {
             return std::make_pair(pair.first, pair.second.ptr.get());
           });
  }

private:
  template <bool Layered>
  void do_register_systems(const std::string &layer_name,
                           SystemHandle<Layered> &handle,
                           detail::AnySystemFn auto &&...fn) {
    if (!system_layers.contains(layer_name)) {
      log.fatal("Attempted to access invalid layer \"{}\"", layer_name);
    }

    auto &layer = system_layers.at(layer_name);
    (do_register_system(layer, std::function(fn), detail::get_system_id(fn),
                        handle),
     ...);
  }

  template <bool Layered, typename... Args>
  void do_register_system(detail::SystemLayer &layer,
                          std::function<void(Args...)> &&fn, uint64_t id,
                          SystemHandle<Layered> &handle) {
    size_t index = layer.systems.size();

    auto dep_registerer = [&layer, id](std::type_index res_id, bool constant) {
      auto &dependees = layer.resource_dependencies[res_id];
      auto &set = constant ? dependees.readers : dependees.writers;
      set.insert(id);
    };

    std::unique_ptr<detail::ISystem> sys;
    if constexpr (detail::SystemArgs<Args...>) {
      sys = std::make_unique<detail::System<Args...>>(std::move(fn),
                                                      dep_registerer);
    } else {
      sys = std::make_unique<detail::ExclusiveSystem>(std::move(fn),
                                                      dep_registerer);
    }

    auto ptr = sys.get();

    layer.dependency_graph.add_node(id);
    layer.system_indices[id] = index;
    layer.systems.push_back(std::move(sys));

    handle.systems.push_back({id, ptr});
  }

  void system_worker(bool main_thread) {
    std::unique_lock lock(layer_mutex);
    layer_cv.wait(lock, [this] { return current_layer != nullptr || done; });

    auto &layer = *current_layer;
    size_t current_system;

    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (ready_queue.empty()) return;

      current_system = ready_queue.front();

      if (layer.systems[current_system]->on_main_thread && !main_thread) {
        return;
      }

      ready_queue.pop();
    }

    layer.systems[current_system]->run(*this);
    remaining_systems--;

    for (size_t dependent : layer.systems[current_system]->dependents) {
      if (!layer.conditions_fulfilled[dependent]) continue;
      if (--layer.current_dependency_counts[dependent] == 0) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        ready_queue.push(dependent);
      }
    }
  }

  template <typename T> detail::Resource &get_resource_internal() {
    if (!has_resource<T>()) {
      log.fatal("Attempted to access unregistered resource {}",
                detail::type_name<T>());
    }

    return resources.at(typeid(T));
  }

  detail::Archetype *get_archetype(detail::ArchetypeId id) {
    for (const auto &archetype : archetypes) {
      if (id == archetype->id) { return archetype.get(); }
    }

    auto archetype = std::make_unique<detail::Archetype>();
    archetype->id = id;

    auto res = archetype.get();
    archetypes.push_back(std::move(archetype));

    return res;
  }

  size_t realloc_archetype_component_data(detail::Archetype *archetype,
                                          ComponentType type) {
    detail::ComponentInterface *comp = &components[type];
    size_t current_size = archetype->entities.size() * comp->size;
    size_t new_size = current_size + comp->size;

    if (new_size > archetype->component_data_sizes[type]) {
      archetype->component_data_sizes[type] *= 2;
      archetype->component_data_sizes[type] += comp->size;

      auto new_data = std::make_unique<uint8_t[]>(
          archetype->component_data_sizes[type]);

      for (size_t e = 0; e < archetype->entities.size(); e++) {
        comp->move(new_data.get() + e * comp->size,
                   archetype->component_data[type].get() + e * comp->size);
        comp->destruct(archetype->component_data[type].get() + e * comp->size);
      }

      archetype->component_data[type] = std::move(new_data);
    }

    return current_size;
  }

  std::pair<size_t, bool> move_archetype(detail::Archetype *new_archetype,
                                         detail::Archetype *old_archetype,
                                         ComponentType type, size_t old_index) {
    detail::ComponentInterface *comp = &components[type];
    size_t old_size = realloc_archetype_component_data(new_archetype, type);

    if (old_archetype->id[type]) {
      comp->move(new_archetype->component_data[type].get() + old_size,
                 old_archetype->component_data[type].get() +
                     old_index * comp->size);

      comp->destruct(old_archetype->component_data[type].get() +
                     old_index * comp->size);

      return {old_size, false};
    } else {
      return {old_size, true};
    }
  }

  void remove_record(const detail::Record &record, ComponentType type) {
    auto [archetype, index] = record;
    if (archetype->entities.empty()) return;

    for (ComponentType i = 0; i < MAX_COMPONENTS; i++) {
      if (!archetype->id[i]) continue;

      detail::ComponentInterface *comp = &components[i];
      if (i == type) {
        comp->destruct(&archetype->component_data[i][index * comp->size]);
      }

      if (index != archetype->entities.size() - 1) {
        comp->move(
            &archetype->component_data[i][index * comp->size],
            &archetype->component_data[i][(archetype->entities.size() - 1) *
                                          comp->size]);
      }
    }

    Entity last = archetype->entities[archetype->entities.size() - 1];
    records[last].index = index;
    archetype->entities[index] = last;
    archetype->entities.resize(archetype->entities.size() - 1);
  }

  detail::Log log;
  Entity next_entity{1};
  ComponentType next_component_type{};

  std::array<detail::ComponentInterface, MAX_COMPONENTS> components;
  std::unordered_map<std::type_index, ComponentType> component_types;
  std::unordered_map<ComponentType, std::type_index> component_type_indices;

  std::vector<std::unique_ptr<detail::Archetype>> archetypes;
  std::unordered_map<Entity, detail::Record> records;

  std::unordered_map<std::type_index, detail::Resource> resources;
  std::vector<std::function<void(Ecs &)>> commands;

  std::unordered_map<std::string, detail::SystemLayer> system_layers;
  std::unordered_set<std::string> states;
  std::string next_state, current_state;
  bool state_changed;

  std::vector<std::thread> thread_pool;
  std::atomic_bool done{};
  std::atomic_size_t remaining_systems{};

  std::queue<size_t> ready_queue;
  std::mutex queue_mutex;

  detail::SystemLayer *current_layer{};
  std::condition_variable layer_cv;
  std::mutex layer_mutex;
};

/// A `Query` is used to access any entities and requested component data
/// represented by the template arguments of the query. The query arguments may
/// be a unique mixture of the following:
/// - `Entity` to get the entity ID of any matched entity in the query.
/// - `T &` to require a component (`T` can and should be `const` where
/// applicable).
/// - `T *` to optionally get a component (`T` can and should be `const` where
/// applicable).
/// - `Filter<T>` where `Filter` may be one of `With`, 'Without', or `Has`.
/// \tparam Ts... The arguments of the query.
template <detail::QueryArgs... Ts> class Query {
  template <detail::QueryArg T> friend struct detail::GetQueryValue;
  friend struct Iterator;

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::tuple<Ts...>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type *;
    using reference = value_type &;

    const Query<Ts...> *query;
    size_t archetype_index, value_index;

    Iterator() = default;
    Iterator(const Query<Ts...> &query, size_t archetype_index,
             size_t value_index)
        : query(&query), archetype_index(archetype_index),
          value_index(value_index) {}

    value_type operator*() const {
      return std::tuple<Ts...>{
          detail::GetQueryValue<Ts>::template get<Ts...>(
              *this, std::get<typename detail::QueryArgValueType<Ts>::Type>(
                         query->values))...,
      };
    }

    Iterator &operator++() {
      if (++value_index >= query->value_counts[archetype_index]) {
        value_index = 0;
        archetype_index++;
      }

      return *this;
    }

    Iterator operator++(int) {
      Iterator temp = *this;
      ++(*this);
      return temp;
    }

    bool operator==(const Iterator &other) const {
      return query == other.query && archetype_index == other.archetype_index &&
             value_index == other.value_index;
    }
  };

public:
  /// Constructs a query with the given `Ecs` reference. It automatically gets
  /// all data from the `Ecs` based on the query arguments.
  Query(Ecs &ecs) {
    detail::ArchetypeId include, exclude;
    (detail::SetIdValue<Ts>::set(ecs, include, exclude), ...);

    for (const auto &archetype : ecs.archetypes) {
      if (archetype->entities.empty() || (archetype->id & include) != include ||
          (archetype->id & exclude) != 0) {
        continue;
      }

      archetypes_count++;
      total_count += archetype->entities.size();
      value_counts.push_back(archetype->entities.size());
      (detail::GetQueryArg<Ts>::get(
           ecs, archetype.get(),
           std::get<typename detail::QueryArgValueType<Ts>::Type>(values)),
       ...);
    }
  }

  /// Gives an iterator to the start of the query data.
  /// \return The begin iterator.
  Iterator begin() const { return Iterator{*this, 0, 0}; }

  /// Gives an iterator to the one past the end of the query data.
  /// \return The end iterator.
  Iterator end() const { return Iterator{*this, archetypes_count, 0}; }

  /// Gives the number of entities matched by the query.
  /// \return The total count of the query.
  size_t size() const { return total_count; }

  /// Returns `*begin()` if the query has a single element, and throws a
  /// `logic_error` otherwise.
  decltype(auto) single() const {
    if (total_count != 1) {
      throw std::logic_error(
          fmt::format("{}::single called on a query with {} values",
                      detail::type_name<decltype(*this)>(), total_count));
    }

    return *begin();
  }

  /// Executes a function for each element of the query, where the function
  /// takes each query argument as a parameter.
  /// \param f The function to invoke for each element.
  void for_each(std::invocable<Ts...> auto &&f) const {
    for (const auto &t : *this) { std::apply(f, t); }
  }

  /// Executes a function for each element of the query, where the function
  /// takes an index and each query argument as a parameter.
  /// \param f The function to invoke for each element.
  void for_each(std::invocable<size_t, Ts...> auto &&f) const {
    for (size_t i = 0; const auto &t : *this) {
      std::apply(std::bind_front(f, i++), t);
    }
  }

  /// Executes in parallel a function for each element of the query, where the
  /// function takes an optional index and each query argument as a parameter.
  /// \param f The function to invoke for each element.
  void parallel_for_each(auto &&f) const
      requires std::invocable<decltype(f), Ts...> ||
               std::invocable<decltype(f), size_t, Ts...> {
    if (total_count == 0) return;

    size_t num_threads = detail::thread_count();
    size_t chunk_size = (total_count + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < total_count; i += chunk_size) {
      futures.push_back(std::async(
          [this, &f, start = i, count = std::min(chunk_size, total_count - i)] {
            auto it = at(start);
            for (size_t i = 0; i < count; i++, ++it) {
              if constexpr (std::invocable<decltype(f), Ts...>) {
                std::apply(f, *it);
              } else {
                std::apply(std::bind_front(f, start + i), *it);
              }
            }
          }));
    }

    for (auto &future : futures) future.wait();
  }

private:
  Iterator at(size_t flat_index) const {
    size_t archetype_idx = 0;
    size_t local_index = flat_index;

    while (archetype_idx < archetypes_count) {
      if (local_index < value_counts[archetype_idx]) { break; }
      local_index -= value_counts[archetype_idx];
      ++archetype_idx;
    }

    return Iterator{*this, archetype_idx, local_index};
  }

  size_t total_count{};
  size_t archetypes_count{};
  std::vector<size_t> value_counts;
  std::tuple<typename detail::QueryArgValueType<Ts>::Type...> values;
};

/// `Res` is used to access a resource in a system.
/// \tparam T The resource to access. This can (and should) be `const` where
/// applicable.
template <typename T> struct Res {
  Res(Ecs &ecs) : value(ecs.get_resource<T>()) {}

  operator T *() { return value; }
  T *operator->() { return value; }
  T &operator*() { return *value; }
  T *value;
};

/// `Commands` can be used in (or out of) a system to defer certain operations
/// on the `Ecs`, which will be executed when `Ecs::run_commands()` or
/// `Ecs::run_systems()` is called.
/// Note that since this uses function callbacks, it is not very efficient. If a
/// large number of such commands need to be done in a system, consider an
/// exclusive system. See `Ecs::run_systems()` for detail.
class Commands {
public:
  Commands(Ecs &ecs) : ecs(ecs) {}

  template <typename... Ts> Entity spawn(Ts &&...components) {
    Entity entity = ecs.next_entity++;
    ecs.commands.push_back([=](Ecs &ecs) {
      ecs.records[entity] = detail::Record{};
      ecs.add_components(entity, std::move(components)...);
    });

    return entity;
  }

  template <typename... Ts>
  void add_components(Entity entity, Ts &&...components) {
    ecs.commands.push_back([=](Ecs &ecs) {
      ecs.add_components(entity, std::move(components)...);
    });
  }

  template <typename T, typename... Args>
  void add_component(Entity entity, Args &&...args) {
    ecs.commands.push_back(
        [=](Ecs &ecs) { ecs.add_component<T>(entity, args...); });
  }

  template <typename T> void remove_component(Entity entity) {
    ecs.commands.push_back([=](Ecs &ecs) { ecs.remove_component<T>(entity); });
  }

  void remove_component(Entity entity, std::type_index type) {
    ecs.commands.push_back(
        [=](Ecs &ecs) { ecs.remove_component(entity, type); });
  }

  void remove_entity(Entity entity) {
    ecs.commands.push_back([=](Ecs &ecs) { ecs.remove_entity(entity); });
  }

  void change_state(const std::string &state) { ecs.change_state(state); }

private:
  Ecs &ecs;
};

/// A condition to only run a system if in the given state.
/// \param state The required state.
/// \return The appropriate condition function.
constexpr auto in_state(const std::string &state) {
  return [state](Ecs &ecs) { return ecs.get_state() == state; };
};

inline EntityRange::EntityRange(Ecs &ecs) : ecs(ecs) {
  for (const auto &archetype : ecs.archetypes) {
    if (archetype->entities.empty()) continue;

    archetypes.push_back(archetype.get());
    entity_arrays.push_back(
        {archetype->entities.size(), archetype->entities.data()});
  }
}

inline auto EntityRange::begin() -> Iterator { return Iterator(*this, 0); }
inline auto EntityRange::end() -> Iterator {
  return Iterator(*this, entity_arrays.size());
}

inline auto EntityHandle::get_components() {
  auto [archetype, index] = ecs.records.at(entity);

  auto v = std::views::iota((ComponentType)0, MAX_COMPONENTS) |
           std::views::filter(
               [=](auto i) { return archetype ? archetype->id[i] : false; }) |
           std::views::transform([=, this](auto i) {
             return std::make_tuple(
                 ecs.component_type_indices.at(i), ecs.components[i].name,
                 &archetype->component_data[i][index * ecs.components[i].size]);
           });

  return v;
}

namespace detail {

template <SystemArgs... Args>
inline System<Args...>::System(std::function<void(Args...)> &&fn,
                               SystemDependencyRegisterer auto &&dep_registerer)
    : fn(fn) {
  dep_registerer(typeid(Ecs &), true);
  (GetSystemDependency<Args>::get(dep_registerer), ...);
}

inline ExclusiveSystem::ExclusiveSystem(
    std::function<void(Ecs &)> &&fn,
    SystemDependencyRegisterer auto &&dep_registerer)
    : fn(fn) {
  dep_registerer(typeid(Ecs &), false);
}

template <SystemArgs... Args>
template <SystemArg T>
inline void System<Args...>::system_setup_arg(Ecs &ecs) {
  if constexpr (detail::is_specialization_of_v<T, Res>) {
    using U = detail::remove_specialization_of_t<T, Res>;
    auto &res = ecs.get_resource_internal<std::remove_const_t<U>>();

    bool main_thread_only = res.read_main_thread_only;
    if (!std::is_const_v<U>) {
      main_thread_only = main_thread_only || res.write_main_thread_only;
    }

    if (main_thread_only) on_main_thread = true;
  }
}

template <AnySystemFn T> inline uint64_t get_system_id(T &&v) {
  using F = std::decay_t<T>;
  if constexpr (std::is_function_v<std::remove_pointer_t<F>> ||
                std::is_member_function_pointer_v<F>) {
    return reinterpret_cast<uintptr_t>(v);
  } else {
    return typeid(F).hash_code();
  }
}

template <> struct GetSystemDependency<Commands> {
  static void get(SystemDependencyRegisterer auto &&registerer) {
    registerer(typeid(Commands), false);
  }
};

template <typename T> struct GetSystemDependency<Res<T>> {
  static void get(SystemDependencyRegisterer auto &&registerer) {
    registerer(typeid(Res<std::remove_const_t<T>>), std::is_const_v<T>);
  }
};

template <QueryArgs... Ts> struct GetSystemDependency<const Query<Ts...> &> {
  template <QueryArg T>
  static void
  get_query_dependency(SystemDependencyRegisterer auto &&registerer) {
    if constexpr (std::is_reference_v<T> || std::is_pointer_v<T>) {
      using Base = std::remove_reference_t<std::remove_pointer_t<T>>;
      registerer(typeid(std::remove_const_t<Base>), std::is_const_v<Base>);
    }
  }

  static void get(SystemDependencyRegisterer auto &&registerer) {
    (get_query_dependency<Ts>(registerer), ...);
  }
};

template <typename T> struct QueryArgValueType<T &> {
  using Type = std::vector<T *>;
};

template <typename T> struct QueryArgValueType<T *> {
  using Type = std::vector<std::optional<T *>>;
};

template <> struct QueryArgValueType<Entity> {
  using Type = std::vector<Entity *>;
};

template <typename T> struct QueryArgValueType<With<T>> {
  using Type = std::monostate;
};

template <typename T> struct QueryArgValueType<Without<T>> {
  using Type = std::monostate;
};

template <typename T> struct QueryArgValueType<Has<T>> {
  using Type = std::vector<bool>;
};

template <typename T> struct GetQueryArg<T &> {
  static void get(Ecs &ecs, detail::Archetype *archetype,
                  detail::QueryArgValueType<T &>::Type &to) {
    to.push_back(reinterpret_cast<T *>(
        archetype->component_data[ecs.get_component_type<T>()].get()));
  }
};

template <typename T> struct GetQueryArg<T *> {
  static void get(Ecs &ecs, detail::Archetype *archetype,
                  detail::QueryArgValueType<T *>::Type &to) {
    auto type = ecs.get_component_type<T>();
    to.push_back(
        archetype->id[type]
            ? reinterpret_cast<T *>(archetype->component_data[type].get())
            : std::optional<T *>{});
  }
};

template <> struct GetQueryArg<Entity> {
  static void get(Ecs &, detail::Archetype *archetype,
                  detail::QueryArgValueType<Entity>::Type &to) {
    to.push_back(archetype->entities.data());
  }
};

template <typename T> struct GetQueryArg<With<T>> {
  static void get(Ecs &, detail::Archetype *,
                  detail::QueryArgValueType<With<T>>::Type &) {}
};

template <typename T> struct GetQueryArg<Without<T>> {
  static void get(Ecs &, detail::Archetype *,
                  detail::QueryArgValueType<Without<T>>::Type &) {}
};

template <typename T> struct GetQueryArg<Has<T>> {
  static void get(Ecs &ecs, detail::Archetype *archetype,
                  detail::QueryArgValueType<Has<T>>::Type &to) {
    to.push_back(archetype->id[ecs.get_component_type<T>()]);
  }
};

template <QueryArg T> struct GetQueryValue {
  template <QueryArgs... Ts>
  static T get(const Query<Ts...>::Iterator &it,
               const detail::QueryArgValueType<T>::Type &value) {
    return value[it.archetype_index][it.value_index];
  }
};

template <typename T> struct GetQueryValue<T *> {
  template <QueryArgs... Ts>
  static T *get(const Query<Ts...>::Iterator &it,
                const detail::QueryArgValueType<T *>::Type &value) {
    return value[it.archetype_index]
               ? &(*value[it.archetype_index])[it.value_index]
               : static_cast<T *>(nullptr);
  }
};

template <typename T> struct GetQueryValue<With<T>> {
  template <QueryArgs... Ts>
  static auto get(const Query<Ts...>::Iterator &,
                  const detail::QueryArgValueType<With<T>>::Type &) {
    return With<T>{};
  }
};

template <typename T> struct GetQueryValue<Without<T>> {
  template <QueryArgs... Ts>
  static auto get(const Query<Ts...>::Iterator &,
                  const detail::QueryArgValueType<Without<T>>::Type &) {
    return Without<T>{};
  }
};

template <typename T> struct GetQueryValue<Has<T>> {
  template <QueryArgs... Ts>
  static auto get(const Query<Ts...>::Iterator &it,
                  const detail::QueryArgValueType<Has<T>>::Type &value) {
    return value[it.archetype_index];
  }
};

template <typename T> struct SetIdValue<T &> {
  static void set(Ecs &ecs, ArchetypeId &include, ArchetypeId &) {
    include.set(ecs.get_component_type<T>());
  }
};

template <typename T> struct SetIdValue<T *> {
  static void set(Ecs &, ArchetypeId &, ArchetypeId &) {}
};

template <> struct SetIdValue<Entity> {
  static void set(Ecs &, ArchetypeId &, ArchetypeId &) {}
};

template <typename T> struct SetIdValue<With<T>> {
  static void set(Ecs &ecs, ArchetypeId &include, ArchetypeId &) {
    include.set(ecs.get_component_type<T>());
  }
};

template <typename T> struct SetIdValue<Without<T>> {
  static void set(Ecs &ecs, ArchetypeId &, ArchetypeId &exclude) {
    exclude.set(ecs.get_component_type<T>());
  }
};

template <typename T> struct SetIdValue<Has<T>> {
  static void set(Ecs &, ArchetypeId &, ArchetypeId &) {}
};

} // namespace detail
} // namespace ecs
