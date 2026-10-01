#ifndef BOOST_UTILITY_HPP
#define BOOST_UTILITY_HPP
// Boost 兼容垫片：KenLM 只用到 boost::noncopyable 与 boost::ref/cref。
#include <boost/noncopyable.hpp>

#include <functional>

namespace boost {

template <class T>
using reference_wrapper = std::reference_wrapper<T>;

template <class T>
inline reference_wrapper<T> ref(T &t) { return std::ref(t); }

template <class T>
inline reference_wrapper<const T> cref(const T &t) { return std::cref(t); }

} // namespace boost
#endif
