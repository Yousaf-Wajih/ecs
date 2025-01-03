// This is an incomplete example program. It does not yet demonstrate all
// features of the ECS.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <string>

#include "ecs.hh"

using namespace ecs;

int main() {
  srand(time(NULL));

  auto log = [](ecs::LogPriority priority, const std::string &str) {
    static constexpr std::string colors[] = {
        "0;90m",
        "0;32m",
        "1;33m",
        "1;31m",
    };

    static constexpr std::string names[] = {
        "VERBOSE",
        "INFO",
        "WARN",
        "ERROR",
    };

    uint8_t p = static_cast<uint8_t>(priority);
    fmt::print("\033[{}[{:7}] {}\033[0;0m\n", colors[p], names[p], str);
  };

  struct Test {
    char c;
  };

  Ecs ecs{log};
  ecs.register_components<int, float, Test>();
  ecs.add_resource<double>(1.0);
  ecs.register_layers("A", "B", "C");

  ecs.register_systems("A", [](Ecs &ecs) {
    ecs.spawn(5, 3.5f);
    ecs.spawn(Test('c'));
    ecs.spawn(7, Test('a'));
    ecs.spawn(9, 1.23f, Test('j'));
  });

  ecs.register_systems("B", [](const Query<int &, With<Test>> &q) {
    for (auto [i, _] : q) i++;
  });

  ecs.register_systems("B",
                       [](Res<double> r, const Query<float &, Has<Test>> &q) {
                         for (auto [f, has_test] : q) {
                           *r += f;
                           f += has_test ? 2 : 1;
                         }
                       });

  ecs.register_systems(
      "C",
      [](Res<const double> r,
         const Query<Entity, const int *, const float *, const Test *> &q) {
        for (auto [e, ip, fp, tp] : q) {
          fmt::print("{}: ", e);
          if (ip) fmt::print("{} ", *ip);
          if (fp) fmt::print("{:.2f} ", *fp);
          if (tp) fmt::print("{} ", tp->c);
          fmt::print("\n");
        }

        fmt::print("Resource = {:.2f}\n", *r);
      });

  ecs.prepare_systems();
  ecs.run_systems("A");
  ecs.run_systems("B");
  ecs.run_systems("C");

  // Expected output:
  // 0: 5, 4.50
  // 1: c
  // 2: 8 a
  // 3: 10 3.23 j
  // Resource = 5.73
}
