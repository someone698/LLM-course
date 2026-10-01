#!/usr/bin/env bash
# P125 第 2 步：用 KenLM lmplz 训练 1..N 阶 n-gram 模型。
# 记录：每阶训练耗时、模型体积、各阶 n-gram 条数、lmplz 的训练过程输出。
#
# 依赖 lmplz / build_binary（KenLM 的 C++ 可执行文件）。
# 可用环境变量覆盖（换语料/换输出目录即可复用，06 对照实验就是这么调它的）：
#   LMPLZ=/path/to/lmplz  ORDERS="1 2 3 4 5"  TRAIN=data/split/train.txt
#   OUT_DIR=out/models  LOG_DIR=out/logs  RESULT=out/train_bench.tsv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LMPLZ="${LMPLZ:-$ROOT/tools/bin/lmplz}"
BUILD_BINARY="${BUILD_BINARY:-$(dirname "$LMPLZ")/build_binary}"
ORDERS="${ORDERS:-1 2 3 4 5}"
TRAIN="${TRAIN:-$ROOT/data/split/train.txt}"
OUT="${OUT_DIR:-$ROOT/out/models}"
LOG="${LOG_DIR:-$ROOT/out/logs}"
TMP="${TMP_DIR:-$ROOT/out/tmp}"
RESULT="${RESULT:-$ROOT/out/train_bench.tsv}"

if [[ ! -x "$LMPLZ" ]]; then
  cat >&2 <<EOF
找不到可执行的 lmplz: $LMPLZ

先编译（本机无 Boost 且无 sudo，官方 cmake 构建走不通，
改用 tools/build_lmplz.sh：保留 KenLM 训练引擎，替换掉依赖 Boost 的 CLI 外壳）：
  git clone https://github.com/kpu/kenlm.git tools/kenlm   # 课程 PPT 第 124 页给出的地址
  bash tools/build_lmplz.sh

或用 LMPLZ=/path/to/lmplz 指定已有的可执行文件。
EOF
  exit 1
fi

mkdir -p "$OUT" "$LOG" "$TMP" "$(dirname "$RESULT")"
printf 'order\ttrain_sec\tarpa_bytes\tbin_bytes\tngram_1\tngram_2\tngram_3\tngram_4\tngram_5\n' > "$RESULT"

for o in $ORDERS; do
  arpa="$OUT/renmin.${o}gram.arpa"
  bin="$OUT/renmin.${o}gram.bin"
  echo "=== 训练 ${o}-gram ==="

  t0=$(date +%s.%N)
  "$LMPLZ" -o "$o" \
      --text "$TRAIN" \
      --arpa "$arpa" \
      --discount_fallback \
      --temp_prefix "$TMP" \
      -S 1G \
      > "$LOG/lmplz.${o}.stdout" 2> "$LOG/lmplz.${o}.stderr"
  t1=$(date +%s.%N)
  secs=$(echo "$t1 - $t0" | bc)

  # KenLM 的二进制格式要求 >=2 阶（"This ngram implementation assumes at
  # least a bigram model"），1-gram 只能留在 ARPA。失败不算致命。
  if "$BUILD_BINARY" "$arpa" "$bin" > "$LOG/build_binary.${o}.log" 2>&1; then
    bsz=$(stat -c%s "$bin")
  else
    bsz=0
    echo "  注：${o}-gram 无法转二进制（KenLM 下限为 2 阶），保留 ARPA"
  fi

  # ARPA 头部有各阶 n-gram 条数：ngram 1=... ngram 2=...
  counts=()
  for k in 1 2 3 4 5; do
    c=$(grep -m1 "^ngram ${k}=" "$arpa" | cut -d= -f2 || echo "")
    counts+=("${c:-0}")
  done

  asz=$(stat -c%s "$arpa")
  printf '%s\t%.2f\t%s\t%s\t%s\n' "$o" "$secs" "$asz" "$bsz" "$(IFS=$'\t'; echo "${counts[*]}")" >> "$RESULT"
  printf '  %s-gram: %.2fs  arpa %.1fMB  bin %.1fMB  ngram: %s\n' \
      "$o" "$secs" "$(echo "scale=1;$asz/1048576"|bc)" "$(echo "scale=1;$bsz/1048576"|bc)" "$(IFS=','; echo "${counts[*]}")"
done

rm -rf "$TMP"
echo
echo "训练耗时汇总: $RESULT"
column -t -s$'\t' "$RESULT"
