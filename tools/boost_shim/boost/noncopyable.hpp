#ifndef BOOST_NONCOPYABLE_HPP
#define BOOST_NONCOPYABLE_HPP
// Boost 兼容垫片：仅覆盖 KenLM 用到的语义（protected 构造/析构，禁止拷贝）。
namespace boost {
struct noncopyable {
 protected:
  noncopyable() = default;
  ~noncopyable() = default;

 private:
  noncopyable(const noncopyable &) = delete;
  noncopyable &operator=(const noncopyable &) = delete;
};
} // namespace boost
#endif
