#!/usr/bin/env bash
# 编译 KenLM 的 lmplz / build_binary，绕开 Boost。
#
# 做法：官方 cmake 构建强制要求 Boost（program_options/system/thread/unit_test），
# 而 Boost 只被 CLI 外壳和单元测试用到，训练引擎本身零依赖。所以这里：
#   * 用 tools/lmplz_main_noboost.cc 替代 lm/builder/lmplz_main.cc（唯一被替换的文件）
#   * 不用 lm/common/size_option.cc（boost 包装，功能已并入替代外壳）
#   * 其余源文件原样编译，训练算法与官方完全一致
# build_binary 用的是 getopt，本来就无 Boost，直接编。
#
# 用法：bash tools/build_lmplz.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/tools/kenlm"
BUILD="$ROOT/tools/build"
BIN="$ROOT/tools/bin"
CXX="${CXX:-g++}"
MAX_ORDER="${MAX_ORDER:-10}"
STD="${STD:-}"                      # 留空用编译器默认标准
JOBS="${JOBS:-$(nproc)}"

[[ -d "$SRC/lm/builder" ]] || { echo "找不到 $SRC，先 clone kpu/kenlm" >&2; exit 1; }

# boost_shim 放最前：本机没有真 Boost，用垫片满足 <boost/...> 的 include
CXXFLAGS="-I$ROOT/tools/boost_shim -I$SRC -O3 -DNDEBUG -DKENLM_MAX_ORDER=$MAX_ORDER -pthread"
[[ -n "$STD" ]] && CXXFLAGS="$CXXFLAGS -std=$STD"

rm -rf "$BUILD"; mkdir -p "$BUILD" "$BIN"

# 收集源文件：排除 *_main.cc（各自单独链接）与 *_test.cc（要 boost::test）
mapfile -t SOURCES < <(
  find "$SRC/util" "$SRC/lm" -name '*.cc' \
    ! -name '*_main.cc' ! -name '*_test.cc' ! -name 'size_option.cc' \
    ! -path '*/interpolate/*' ! -path '*/filter/*' ! -path '*/wrappers/*' | sort
)
echo "编译 ${#SOURCES[@]} 个源文件（-j$JOBS）..."

compile_one() {
  local src="$1" rel obj
  rel="${src#$SRC/}"
  obj="$BUILD/${rel//\//_}.o"
  $CXX $CXXFLAGS -c "$src" -o "$obj" || { echo "FAILED: $rel" >&2; return 1; }
}
export -f compile_one
export CXX CXXFLAGS SRC BUILD

printf '%s\n' "${SOURCES[@]}" | xargs -P "$JOBS" -I{} bash -c 'compile_one "$@"' _ {}

echo "编译外壳与 build_binary ..."
$CXX $CXXFLAGS -c "$ROOT/tools/lmplz_main_noboost.cc" -o "$BUILD/_lmplz_main.o"
$CXX $CXXFLAGS -c "$SRC/lm/build_binary_main.cc" -o "$BUILD/_build_binary_main.o"

echo "链接 ..."
# 两个可执行文件各有自己的 main，链接时要互斥
COMMON=$(ls "$BUILD"/*.o | grep -v '/_lmplz_main\.o$' | grep -v '/_build_binary_main\.o$')
$CXX $COMMON "$BUILD/_lmplz_main.o"        -o "$BIN/lmplz"        $CXXFLAGS -lrt -pthread
$CXX $COMMON "$BUILD/_build_binary_main.o" -o "$BIN/build_binary" $CXXFLAGS -lrt -pthread

echo "完成："
ls -la "$BIN"
"$BIN/lmplz" --help 2>&1 | head -3 || true
