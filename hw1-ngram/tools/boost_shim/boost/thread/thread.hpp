#ifndef BOOST_THREAD_THREAD_HPP
#define BOOST_THREAD_THREAD_HPP
// Boost 兼容垫片：boost::thread。
//
// 这里不能简单 `using std::thread`。二者参数绑定语义不同：
//   boost::thread 把实参 decay 保存后，以 **左值引用** 交给可调用对象；
//   std::thread   以 **右值** 传递，要求可调用对象能接受右值。
// KenLM util/stream/chain.hh 的签名是
//     void operator()(const Position &position, Worker &worker)
// （worker 是非 const 左值引用），用 std::thread 会触发
//     static assertion failed: std::thread arguments must be invocable
//     after conversion to rvalues
// 因此这里实现一个保持 boost 语义的薄包装：decay 保存 + 调用时以左值传入。
#include <boost/utility.hpp>

#include <functional>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

namespace boost {

class thread {
 public:
  thread() noexcept = default;

  template <class F, class... Args>
  explicit thread(F &&f, Args &&...args) {
    std::decay_t<F> callable(std::forward<F>(f));
    auto bound = std::make_tuple(std::decay_t<Args>(std::forward<Args>(args))...);
    // mutable 是必须的：否则 lambda 的 operator() 为 const，bound 里的元素
    // 会以 const 左值传给 callable，仍然绑不上 Worker&。
    t_ = std::thread([callable, bound]() mutable {
      std::apply([&callable](auto &...a) { std::invoke(callable, a...); }, bound);
    });
  }

  thread(thread &&o) noexcept : t_(std::move(o.t_)) {}
  thread &operator=(thread &&o) noexcept {
    t_ = std::move(o.t_);
    return *this;
  }
  // 与 boost 一致：析构时不 join，直接 detach（KenLM 自己会显式 join）
  ~thread() {
    if (t_.joinable()) t_.detach();
  }

  void join() { t_.join(); }
  void detach() { t_.detach(); }
  bool joinable() const { return t_.joinable(); }

  static unsigned hardware_concurrency() { return std::thread::hardware_concurrency(); }

 private:
  std::thread t_;
  thread(const thread &) = delete;
  thread &operator=(const thread &) = delete;
};

} // namespace boost
#endif
