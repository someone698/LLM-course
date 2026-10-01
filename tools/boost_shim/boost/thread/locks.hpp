#ifndef BOOST_THREAD_LOCKS_HPP
#define BOOST_THREAD_LOCKS_HPP
// Boost 兼容垫片：boost::unique_lock / lock_guard -> std 对应物。
#include <mutex>

namespace boost {

template <class Mutex>
using unique_lock = std::unique_lock<Mutex>;

template <class Mutex>
using lock_guard = std::lock_guard<Mutex>;

using std::defer_lock;
using std::adopt_lock;
using std::try_to_lock;

} // namespace boost
#endif
