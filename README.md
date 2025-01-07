# ECS
This is a small ECS I've been working on in C++.

## How to Use
Just copy and paste `ecs.hh` and `ecs_hooks.hh` into your project. `ecs_hooks.hh` has empty functions by default. You need a C++20-compatible compiler.

## Core concepts
### Entities
An entity is just an ID which is defined as a `uint64_t`. That is:
```cpp
using Entity = uint64_t;
```

An important thing to note is that entity IDs are *not* reused. This means that when an entity is deleted, its ID is not used again for future entities. This is usually not an issue, because with 64 bits of space, there are a total of 2^64 IDs which should be more than enough for any use case. Just to put this into perspective, if a million entities are spawned every second, it would take around 58494 decades to run out of the 64-bit space.

To spawn an entity, `Ecs::spawn()` can be used with any number of components to add to any entities.

### Components
Components are just plain structs, like:
```cpp
struct Position {
  float x, y;
};
```

No annotations are needed on the struct itself. However, it needs to be registered for use with the ECS like so:
```cpp
ecs.register_component<Position>();
```

It is also possible to use `Ecs::register_components()` to register multiple components at once. The maximum number of components that can be registered is limited. By default, it is 128. It is possible to modify this by defining `ECS_MAX_COMPONENTS`. Do note that internally, the ECS uses IDs to represent the component types. This is defined as:
```cpp
using ComponentType = ECS_COMPONENT_TYPE;
```

By default, `ECS_COMPONENT_TYPE` is `uint8_t`. If you need more than 256 types of components, change this to a larger type.

Components can be accessed in systems using queries. Functions exist to add and remove components from entities as well.

### Systems
A system is any callable (i.e., a function, lambda, or object with `operator()`). There are two types of systems:
- **Exclusive System:** An exclusive system takes a single argument: a reference to the `Ecs` instance. This way, it can do pretty much anything with the ECS.
- **"Normal" System:** Any system which is not an exclusive system is a normal system. Such a system can contain any unique combination of valid system arguments. A valid system argument may be `Query<Ts...>`, `Res<T>`, or `Commands`.

Systems can be registered using `Ecs::register_systems()` or `Ecs::register_systems_on_transition()`. These functions return a "handle" that can be used to modify certain aspects of the registered system(s). For example,
```cpp
ecs.register_systems("TestLayer", [](...) { ... }).run_if(in_state("True"));
```

### System Layers
Systems are divided into "layers". For a game, you might have an "Update" layer, a "Render" layer, a "Start" layer, etc. When registering systems, you have to specify which layer to register it on. And to run systems, you must also specify which system layer to run. Also note that inter-dependency connections can only be added between systems of the same layer.
Layers must be registered using `Ecs::register_layer()` or `Ecs::register_layers()`. Some layers for state transitions are automatically registered by the ECS.

### System Execution
Before executing any systems, `Ecs::prepare_systems()` must be called. And after a call to this function, no new systems may be registered.
To run systems, `Ecs::run_systems()` can be used with the appropriate state name. Normal systems are run in parallel where possible. If any two systems cannot be run in parallel (for example, at least one of them mutably accesses a shared resource), then the order in which they are executed is usually undefined, except when:
- An explicit ordering is done using `SystemHandle::before()` or `SystemHandle::after()`. It is also possible to group systems using the `SystemHandle::in_group()` function, and then add ordering with a whole group.
- One system accesses a resource mutably, while the other accesses it as `const`. In this case, the system which accesses mutably is run first, unless explicitly specified otherwise using the `before()` or `after()` functions.

Exclusive systems are never run in parallel with any other systems and are always run on the main thread.

The parallel execution is done by creating a thread pool (with the number of threads given by `std::thread::hardware_concurrency()` when the `Ecs` is constructed, and signalling the worker threads during `run_systems()`. The pool is destroyed when the `Ecs` is destructed. Sometimes, a system needs to be run only on the main thread. In such a case, `SystemHandle::on_main_thread()` can be used.

### States
The ECS can be in different "states". The semantic meaning of these is up to the user to decide. It is also possible not to use this feature at all. The ECS provides functions to:
- Set all possible states of the ECS using `Ecs::set_states()`
- Change the current state using `Ecs::change_state()` or `Commands::change_state()`.
- Add conditions to run a system only in a specific state using `in_state()` and `SystemHandle::run_if()`.

When the state is changed, a "transition" automatically occurs on the next `Ecs::run_systems()` call. That is, the systems on the appropriate "transition layer" are executed. Systems can be registered to run on a transition using `Ecs::register_systems_on_transition()`.

### Resources
Resources are global data storage. The memory is owned and managed by the ECS. They can be added using `Ecs::add_resource<T>()`. They can be accessed with `Res<T>` in systems, and `Ecs::get_resource<T>()` otherwise. `Ecs::add_resource<T>()` returns a handle, which can be used to set some properties of the resource. For example, it is possible to set a resource to be only accessed on the main thread. This way, any system using that resource will always be run on the main thread.

### Hooks
In the `ecs_hooks.h` file, there are some functions which are called when the corresponding functions are called in the `Ecs`. You can modify these functions to do whatever you like.

### Access to the Whole `Ecs` State
Sometimes, it is necessary to get access to all entities and their components, or to get access to all resources. For this, the `Ecs` provides `Ecs::get_all_entities()` and `Ecs::get_all_resources()` to access all entities (along with their components) and all resources in a type-erased manner.
