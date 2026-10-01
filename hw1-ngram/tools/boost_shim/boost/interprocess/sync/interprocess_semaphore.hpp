#ifndef BOOST_INTERPROCESS_SYNC_INTERPROCESS_SEMAPHORE_HPP
#define BOOST_INTERPROCESS_SYNC_INTERPROCESS_SEMAPHORE_HPP
// Boost 兼容垫片：用 POSIX 无名信号量实现。
// KenLM 在 util/pcqueue.hh 里作为生产者/消费者队列的计数信号量使用。
#include <cerrno>
#include <stdexcept>
#include <string>

#include <semaphore.h>

namespace boost {
namespace interprocess {

class interprocess_exception : public std::runtime_error {
 public:
  // native_ 在构造时捕获 errno：KenLM 的 util/pcqueue.hh 会
  // catch 它并判断 e.get_native_error() != EINTR 来决定是否重试。
  interprocess_exception(const std::string &what, int native)
      : std::runtime_error(what), native_(native) {}
  explicit interprocess_exception(const std::string &what)
      : std::runtime_error(what), native_(0) {}

  int get_native_error() const { return native_; }

 private:
  int native_;
};

class interprocess_semaphore {
 public:
  explicit interprocess_semaphore(unsigned int initial) {
    if (sem_init(&sem_, 0, initial) != 0) {
      const int e = errno;
      throw interprocess_exception("boost_shim: sem_init failed", e);
    }
  }
  ~interprocess_semaphore() { sem_destroy(&sem_); }

  void post() {
    if (sem_post(&sem_) != 0) {
      const int e = errno;
      throw interprocess_exception("boost_shim: sem_post failed", e);
    }
  }

  void wait() {
    if (sem_wait(&sem_) != 0) {
      const int e = errno;
      throw interprocess_exception("boost_shim: sem_wait failed", e);
    }
  }

  bool try_wait() {
    if (sem_trywait(&sem_) == 0) return true;
    if (errno == EAGAIN || errno == EINTR) return false;
    const int e = errno;
    throw interprocess_exception("boost_shim: sem_trywait failed", e);
  }

 private:
  sem_t sem_;
  interprocess_semaphore(const interprocess_semaphore &) = delete;
  interprocess_semaphore &operator=(const interprocess_semaphore &) = delete;
};

} // namespace interprocess
} // namespace boost
#endif
