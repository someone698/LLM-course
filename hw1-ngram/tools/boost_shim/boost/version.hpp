#ifndef BOOST_VERSION_HPP
#define BOOST_VERSION_HPP
// Boost 兼容垫片。KenLM 只在 lmplz_main.cc 等处用 BOOST_VERSION 做特性判断；
// 报一个足够新的版本号，使其走 "已有 required()" 的分支。
#define BOOST_VERSION 108800
#define BOOST_LIB_VERSION "1_88"
#endif
