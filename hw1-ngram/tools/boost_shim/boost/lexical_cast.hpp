#ifndef BOOST_LEXICAL_CAST_HPP
#define BOOST_LEXICAL_CAST_HPP
// Boost 兼容垫片。KenLM 用在 lm/common/model_buffer.cc（int -> string）
// 和 lm/builder/debug_print.hh（int -> string）。
#include <sstream>
#include <stdexcept>
#include <string>

namespace boost {

class bad_lexical_cast : public std::runtime_error {
 public:
  bad_lexical_cast() : std::runtime_error("bad lexical cast") {}
};

namespace detail {

template <class T>
inline T LexicalCastFrom(const std::string &s) {
  std::istringstream in(s);
  T out;
  in >> out;
  if (in.fail()) throw bad_lexical_cast();
  return out;
}

template <>
inline std::string LexicalCastFrom<std::string>(const std::string &s) { return s; }

} // namespace detail

template <class T, class Source>
inline T lexical_cast(const Source &src) {
  std::ostringstream out;
  out << src;
  return detail::LexicalCastFrom<T>(out.str());
}

} // namespace boost
#endif
