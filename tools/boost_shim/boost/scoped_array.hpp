#ifndef BOOST_SCOPED_ARRAY_HPP
#define BOOST_SCOPED_ARRAY_HPP
// Boost 兼容垫片。KenLM 用在 lm/builder/corpus_count.cc：
//   boost::scoped_array<WordIndex> buffer_;  buffer_.reset(new WordIndex[n]);
#include <cstddef>
#include <memory>

namespace boost {

template <class T>
class scoped_array {
 public:
  scoped_array() : p_() {}
  explicit scoped_array(T *p) : p_(p) {}
  scoped_array(scoped_array &&o) : p_(std::move(o.p_)) {}

  void reset(T *p = 0) { p_.reset(p); }
  T &operator[](std::ptrdiff_t i) const { return p_[i]; }
  T *get() const { return p_.get(); }
  explicit operator bool() const { return static_cast<bool>(p_); }
  void swap(scoped_array &o) { p_.swap(o.p_); }

 private:
  std::unique_ptr<T[]> p_;
  scoped_array(const scoped_array &) = delete;
  scoped_array &operator=(const scoped_array &) = delete;
};

} // namespace boost
#endif
