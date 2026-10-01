#ifndef BOOST_RANGE_ITERATOR_RANGE_HPP
#define BOOST_RANGE_ITERATOR_RANGE_HPP
// Boost 兼容垫片：只实现 KenLM util/multi_intersection.hh 用到的接口
// （empty/size/begin/end/front/back/advance_begin）。
#include <cstddef>
#include <iterator>

namespace boost {

template <class Iterator>
class iterator_range {
 public:
  typedef Iterator iterator;
  typedef typename std::iterator_traits<Iterator>::reference reference;

  iterator_range() : b_(), e_() {}
  iterator_range(Iterator b, Iterator e) : b_(b), e_(e) {}

  iterator begin() const { return b_; }
  iterator end() const { return e_; }
  bool empty() const { return b_ == e_; }
  std::size_t size() const { return static_cast<std::size_t>(std::distance(b_, e_)); }
  reference front() const { return *b_; }
  reference back() const {
    iterator t = e_;
    --t;
    return *t;
  }

  void advance_begin(std::size_t n) { std::advance(b_, n); }
  void advance_end(std::size_t n) { std::advance(e_, n); }

 private:
  Iterator b_, e_;
};

template <class Iterator>
inline iterator_range<Iterator> make_iterator_range(Iterator b, Iterator e) {
  return iterator_range<Iterator>(b, e);
}

} // namespace boost
#endif
