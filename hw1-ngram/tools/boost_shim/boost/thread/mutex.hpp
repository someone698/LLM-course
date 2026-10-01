#ifndef BOOST_THREAD_MUTEX_HPP
#define BOOST_THREAD_MUTEX_HPP
// Boost 兼容垫片：boost::mutex -> std::mutex。
//
// 真 Boost 的 boost/thread/mutex.hpp 会连带提供 boost::unique_lock
// （经 lock_types.hpp）。KenLM 的 util/pcqueue.hh 与
// util/stream/multi_progress.cc 只用 <boost/thread/mutex.hpp> 就写了
// boost::unique_lock<boost::mutex>，依赖的正是这一传递关系，故一并引入。
#include <boost/thread/locks.hpp>

#include <mutex>

namespace boost {

using std::mutex;
using std::recursive_mutex;
using std::timed_mutex;
using std::try_to_lock_t;
using std::try_to_lock;

} // namespace boost
#endif
