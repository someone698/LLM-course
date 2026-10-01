#ifndef BOOST_OPTIONAL_HPP
#define BOOST_OPTIONAL_HPP
// Boost 兼容垫片。基于 std::optional，额外补上 KenLM 可能调用的 .get()。
// 继承而非别名，是为了同时提供 *p / p-> / (bool)p / p.get() 四种访问方式。
#include <optional>

namespace boost {

using std::nullopt;
using std::nullopt_t;

template <class T>
class optional : public std::optional<T> {
 public:
  optional() : std::optional<T>() {}
  optional(nullopt_t) : std::optional<T>(std::nullopt) {}
  optional(const T &v) : std::optional<T>(v) {}
  optional(const optional &) = default;
  optional(optional &&) = default;

  optional &operator=(const T &v) {
    std::optional<T>::operator=(v);
    return *this;
  }
  optional &operator=(nullopt_t) {
    std::optional<T>::operator=(std::nullopt);
    return *this;
  }

  T &get() { return **this; }
  const T &get() const { return **this; }
};

} // namespace boost
#endif
