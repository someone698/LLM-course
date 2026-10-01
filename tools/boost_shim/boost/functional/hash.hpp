#ifndef BOOST_FUNCTIONAL_HASH_HPP
#define BOOST_FUNCTIONAL_HASH_HPP
// Boost 兼容垫片。KenLM 只在 util/string_piece_hash.hh 里调用 hash_range
// 来给 StringPiece 做哈希（用于 probing hash table 的桶分布）。
// 哈希值与真 Boost 不同，但只影响哈希表的内部布局，不影响任何计算结果。
#include <cstddef>
#include <functional>
#include <string>

namespace boost {

template <class T>
struct hash : public std::hash<T> {};

template <class It>
inline std::size_t hash_range(It first, It last) {
  std::size_t seed = 0;
  for (; first != last; ++first) {
    seed ^= static_cast<std::size_t>(*first) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
  }
  return seed;
}

template <class It>
inline void hash_range(std::size_t &seed, It first, It last) {
  for (; first != last; ++first) {
    seed ^= static_cast<std::size_t>(*first) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
  }
}

} // namespace boost
#endif
