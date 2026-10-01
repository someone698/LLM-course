#ifndef BOOST_PTR_CONTAINER_PTR_VECTOR_HPP
#define BOOST_PTR_CONTAINER_PTR_VECTOR_HPP
// Boost 兼容垫片：只实现 KenLM 用到的那部分 boost::ptr_vector 接口。
// 语义：容器持有指针的所有权；迭代器解引用得到 T&（而不是 T*）。
// KenLM 各处用法：push_back(new T(...)) / empty() / clear() / size() /
//                 [i] -> T& / begin()..end() 且 *it 为 T&。
// 真 Boost 的 ptr_container 建立在 smart_ptr 之上，会连带引入 scoped_array。
// KenLM 依赖了这一点：lm/builder/corpus_count.cc 使用了 boost::scoped_array
// 却没直接 include 它，而是经由 util/stream/chain.hh -> 本头文件 拿到。
// 这里补上同一传递关系，保持与上游一致的可见性。
#include <boost/scoped_array.hpp>

#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

namespace boost {
namespace ptr_container_detail {

template <class T, class Base>
class IndirectIterator {
 public:
  typedef std::random_access_iterator_tag iterator_category;
  typedef T value_type;
  typedef std::ptrdiff_t difference_type;
  typedef T *pointer;
  typedef T &reference;

  IndirectIterator() : b_() {}
  explicit IndirectIterator(Base b) : b_(b) {}

  reference operator*() const { return **b_; }
  pointer operator->() const { return b_->get(); }
  reference operator[](difference_type n) const { return *b_[n]; }

  IndirectIterator &operator++() { ++b_; return *this; }
  IndirectIterator operator++(int) { IndirectIterator t(*this); ++b_; return t; }
  IndirectIterator &operator--() { --b_; return *this; }
  IndirectIterator operator--(int) { IndirectIterator t(*this); --b_; return t; }
  IndirectIterator &operator+=(difference_type n) { b_ += n; return *this; }
  IndirectIterator &operator-=(difference_type n) { b_ -= n; return *this; }
  IndirectIterator operator+(difference_type n) const { return IndirectIterator(b_ + n); }
  IndirectIterator operator-(difference_type n) const { return IndirectIterator(b_ - n); }
  difference_type operator-(const IndirectIterator &o) const { return b_ - o.b_; }
  bool operator==(const IndirectIterator &o) const { return b_ == o.b_; }
  bool operator!=(const IndirectIterator &o) const { return b_ != o.b_; }
  bool operator<(const IndirectIterator &o) const { return b_ < o.b_; }
  bool operator>(const IndirectIterator &o) const { return b_ > o.b_; }
  bool operator<=(const IndirectIterator &o) const { return b_ <= o.b_; }
  bool operator>=(const IndirectIterator &o) const { return b_ >= o.b_; }

 private:
  Base b_;
};

} // namespace ptr_container_detail

template <class T, class CloneAllocator = void>
class ptr_vector {
 public:
  typedef ptr_container_detail::IndirectIterator<
      T, typename std::vector<std::unique_ptr<T> >::iterator>
      iterator;
  typedef ptr_container_detail::IndirectIterator<
      const T, typename std::vector<std::unique_ptr<T> >::const_iterator>
      const_iterator;

  ptr_vector() {}
  ~ptr_vector() { v_.clear(); }

  void push_back(T *p) { v_.push_back(std::unique_ptr<T>(p)); }
  void pop_back() { v_.pop_back(); }
  void clear() { v_.clear(); }
  void reserve(std::size_t n) { v_.reserve(n); }
  void resize(std::size_t n) { v_.resize(n); }

  bool empty() const { return v_.empty(); }
  std::size_t size() const { return v_.size(); }

  T &operator[](std::size_t i) { return *v_[i]; }
  const T &operator[](std::size_t i) const { return *v_[i]; }
  T &at(std::size_t i) { return *v_.at(i); }
  const T &at(std::size_t i) const { return *v_.at(i); }
  T &front() { return *v_.front(); }
  T &back() { return *v_.back(); }

  iterator begin() { return iterator(v_.begin()); }
  iterator end() { return iterator(v_.end()); }
  const_iterator begin() const { return const_iterator(v_.begin()); }
  const_iterator end() const { return const_iterator(v_.end()); }

 private:
  std::vector<std::unique_ptr<T> > v_;
  ptr_vector(const ptr_vector &) = delete;
  ptr_vector &operator=(const ptr_vector &) = delete;
};

} // namespace boost
#endif
