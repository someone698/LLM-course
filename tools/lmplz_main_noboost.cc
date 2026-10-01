// lmplz_main_noboost.cc —— KenLM lmplz 的免 Boost 外壳
//
// 为什么存在这个文件：
//   KenLM 官方 lmplz 的 CMake 构建 `find_package(Boost REQUIRED COMPONENTS
//   program_options system thread unit_test_framework)`，本机无 Boost 且无 sudo。
//   查过全仓后确认：Boost 只被三处用到——
//     (1) 各 *_main.cc 的命令行解析 boost::program_options
//     (2) lm/common/size_option.{hh,cc}（program_options 的 notifier 包装）
//     (3) 单元测试 boost::test
//   真正的训练引擎 lm/builder/{corpus_count,adjust_counts,initial_probabilities,
//   interpolate,output,pipeline}.cc 不含任何 Boost 依赖。
//
//   所以本文件是 lm/builder/lmplz_main.cc 的逐段移植，仅把 (1)(2) 换成手写的
//   参数解析；-S/--memory 的 "1G"/"80%" 语法仍走 KenLM 自己的
//   util::ParseSize（在 util/usage.cc，本来就无 Boost）。
//
//   **训练算法本身完全来自 KenLM 原版**（modified Kneser-Ney，Heafield et al. 2013），
//   未作任何修改。本文件替换的只是 CLI 外鞘。
//
// 编译见 tools/build_lmplz.sh。

#include "lm/builder/output.hh"
#include "lm/builder/pipeline.hh"
#include "lm/lm_exception.hh"
#include "util/file.hh"
#include "util/file_piece.hh"
#include "util/usage.hh"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

// ---- 原 lm/common/size_option.cc 的 boost 版等价物：改用 util::ParseSize ----
inline void SetSize(std::size_t &to, const std::string &from) {
  to = util::ParseSize(from);
}

// ---- 原 boost::lexical_cast<uint64_t> 的等价物 ----
uint64_t ParseU64(const std::string &s, const char *what) {
  try {
    std::size_t pos = 0;
    uint64_t v = std::stoull(s, &pos);
    UTIL_THROW_IF(pos != s.size(), util::Exception, "Bad " << what << " " << s);
    return v;
  } catch (const std::invalid_argument &) {
    UTIL_THROW(util::Exception, "Bad " << what << " " << s);
  } catch (const std::out_of_range &) {
    UTIL_THROW(util::Exception, "Out of range " << what << " " << s);
  }
}

// 原样保留：解析并校验剪枝阈值
std::vector<uint64_t> ParsePruning(const std::vector<std::string> &param, std::size_t order) {
  std::vector<uint64_t> prune_thresholds;
  prune_thresholds.reserve(order);
  for (std::vector<std::string>::const_iterator it(param.begin()); it != param.end(); ++it) {
    prune_thresholds.push_back(ParseU64(*it, "pruning threshold"));
  }
  if (prune_thresholds.empty()) {
    prune_thresholds.resize(order, 0);
    return prune_thresholds;
  }
  UTIL_THROW_IF(prune_thresholds.size() > order, util::Exception,
                "You specified pruning thresholds for orders 1 through " << prune_thresholds.size()
                << " but the model only has order " << order);
  uint64_t lower_threshold = 0;
  for (std::vector<uint64_t>::iterator it = prune_thresholds.begin(); it != prune_thresholds.end(); ++it) {
    UTIL_THROW_IF(lower_threshold > *it, util::Exception,
                  "Pruning thresholds should be in non-decreasing order.  "
                  "Otherwise substrings would be removed, which is bad for query-time data structures.");
    lower_threshold = *it;
  }
  prune_thresholds.resize(order, prune_thresholds.back());
  return prune_thresholds;
}

// 原样保留：解析 fallback 折扣
lm::builder::Discount ParseDiscountFallback(const std::vector<std::string> &param) {
  lm::builder::Discount ret;
  UTIL_THROW_IF(param.size() > 3, util::Exception,
                "Specify at most three fallback discounts: 1, 2, and 3+");
  UTIL_THROW_IF(param.empty(), util::Exception,
                "Fallback discounting enabled, but no discount specified");
  ret.amount[0] = 0.0;
  for (unsigned i = 0; i < 3; ++i) {
    float discount = std::stof(param[i < param.size() ? i : (param.size() - 1)]);
    UTIL_THROW_IF(discount < 0.0 || discount > static_cast<float>(i + 1), util::Exception,
                  "The discount for count " << (i + 1) << " was parsed as " << discount
                  << " which is not in the range [0, " << (i + 1) << "].");
    ret.amount[i + 1] = discount;
  }
  return ret;
}

struct Options {
  std::size_t order = 0;
  bool has_order = false;
  std::string text, arpa, intermediate, limit_vocab_file;
  bool has_text = false, has_arpa = false, has_intermediate = false;
  std::string memory, minimum_block, sort_block, temp_prefix;
  std::size_t block_count = 2;
  lm::WordIndex vocab_estimate = 1000000;
  uint64_t vocab_pad = 0;
  bool interpolate_unigrams = true, has_interpolate = false;
  bool skip_symbols = false, verbose_header = false, renumber = false;
  bool collapse_values = false, help = false;
  bool has_discount_fallback = false;
  std::vector<std::string> pruning, discount_fallback;
};

void PrintHelp() {
  std::cerr <<
    "Builds unpruned language models with modified Kneser-Ney smoothing.\n"
    "(KenLM lmplz, 免 Boost 外壳版：参数解析替换，训练引擎原样)\n\n"
    "Please cite:\n"
    "@inproceedings{Heafield-estimate,\n"
    "  author = {Kenneth Heafield and Ivan Pouzyrevsky and Jonathan H. Clark and Philipp Koehn},\n"
    "  title = {Scalable Modified {Kneser-Ney} Language Model Estimation},\n"
    "  year = {2013},\n"
    "  month = {8},\n"
    "  booktitle = {Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics},\n"
    "  address = {Sofia, Bulgaria},\n"
    "  url = {http://kheafield.com/professional/edinburgh/estimate\\_paper.pdf},\n"
    "}\n\n"
    "Options:\n"
    "  -o, --order N              Order of the model (required)\n"
    "  --text FILE                Read text from a file instead of stdin\n"
    "  --arpa FILE                Write ARPA to a file instead of stdout\n"
    "  -S, --memory SIZE          Sorting memory (default 80%, or 1G if unknown)\n"
    "  -T, --temp_prefix PREFIX   Temporary file prefix\n"
    "  --minimum_block SIZE       Minimum block size to allow (default 8K)\n"
    "  --sort_block SIZE          Size of IO operations for sort (default 64M)\n"
    "  --block_count N            Block count per order (default 2)\n"
    "  --vocab_estimate N         Assume this vocabulary size (default 1000000)\n"
    "  --vocab_pad N              Pad vocabulary with <unk> to this size\n"
    "  --interpolate_unigrams [0|1]  Interpolate unigrams (default 1)\n"
    "  --skip_symbols             Treat <s>, </s>, <unk> as whitespace\n"
    "  --verbose_header           Verbose ARPA header (token count, smoothing, ...)\n"
    "  --prune T1 T2 ...          Prune n-grams with count <= threshold, per order\n"
    "  --limit_vocab_file FILE    Read allowed vocabulary from FILE\n"
    "  --discount_fallback [d1 d2 d3]  Fallback discounts (default 0.5 1 1.5)\n"
    "  --renumber                 Renumber vocabulary to be monotone with hash\n"
    "  --collapse_values          Collapse probability and backoff into q\n"
    "  -h, --help                 This message\n\n"
    "Memory sizes are specified like GNU sort: a number followed by a unit character.\n"
    "Valid units are \% for percentage of memory and (in increasing powers of 1024):\n"
    "b, K, M, G, T, P, E, Z, Y.  Default is K (*1024).\n";
  uint64_t mem = util::GuessPhysicalMemory();
  if (mem) std::cerr << "This machine has " << mem << " bytes of memory.\n";
  else std::cerr << "Unable to determine the amount of memory on this machine.\n";
}

// 手写参数解析：语义与 boost::program_options 版一致
Options ParseArgs(int argc, char *argv[]) {
  Options o;
  o.temp_prefix = util::DefaultTempDirectory();
  o.memory = util::GuessPhysicalMemory() ? "80%" : "1G";
  o.minimum_block = "8K";
  o.sort_block = "64M";
  o.discount_fallback.push_back("0.5");
  o.discount_fallback.push_back("1");
  o.discount_fallback.push_back("1.5");

  std::vector<std::string> disc_tmp;  // --discount_fallback 显式给的

  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]), val;
    bool has_eq = false;
    if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
      std::size_t eq = a.find('=');
      if (eq != std::string::npos) { val = a.substr(eq + 1); a = a.substr(0, eq); has_eq = true; }
    }
    // 取下一个 argv 作值
    auto next = [&](const char *name) -> std::string {
      if (has_eq) return val;
      UTIL_THROW_IF(i + 1 >= argc, util::Exception, name << " requires a value");
      return std::string(argv[++i]);
    };

    if (a == "-h" || a == "--help") { o.help = true; }
    else if (a == "-o" || a == "--order") { o.order = ParseU64(next("--order"), "order"); o.has_order = true; }
    else if (a == "--text") { o.text = next("--text"); o.has_text = true; }
    else if (a == "--arpa") { o.arpa = next("--arpa"); o.has_arpa = true; }
    else if (a == "--intermediate") { o.intermediate = next("--intermediate"); o.has_intermediate = true; }
    else if (a == "-T" || a == "--temp_prefix") { o.temp_prefix = next("--temp_prefix"); }
    else if (a == "-S" || a == "--memory") { o.memory = next("--memory"); }
    else if (a == "--minimum_block") { o.minimum_block = next("--minimum_block"); }
    else if (a == "--sort_block") { o.sort_block = next("--sort_block"); }
    else if (a == "--block_count") { o.block_count = ParseU64(next("--block_count"), "block_count"); }
    else if (a == "--vocab_estimate") { o.vocab_estimate = static_cast<lm::WordIndex>(ParseU64(next("--vocab_estimate"), "vocab_estimate")); }
    else if (a == "--vocab_pad") { o.vocab_pad = ParseU64(next("--vocab_pad"), "vocab_pad"); }
    else if (a == "--limit_vocab_file") { o.limit_vocab_file = next("--limit_vocab_file"); }
    else if (a == "--interpolate_unigrams") {
      o.has_interpolate = true;
      if (has_eq) { o.interpolate_unigrams = (val != "0"); }
      else if (i + 1 < argc && argv[i + 1][0] != '-') { o.interpolate_unigrams = (std::string(argv[++i]) != "0"); }
      else { o.interpolate_unigrams = true; }
    }
    else if (a == "--skip_symbols") { o.skip_symbols = true; }
    else if (a == "--verbose_header") { o.verbose_header = true; }
    else if (a == "--renumber") { o.renumber = true; }
    else if (a == "--collapse_values") { o.collapse_values = true; }
    else if (a == "--discount_fallback") {
      o.has_discount_fallback = true;
      if (has_eq) { disc_tmp.push_back(val); }
      else { while (i + 1 < argc && argv[i + 1][0] != '-') disc_tmp.push_back(argv[++i]); }
    }
    else if (a == "--prune") {
      if (has_eq) { o.pruning.push_back(val); }
      else { while (i + 1 < argc && argv[i + 1][0] != '-') o.pruning.push_back(argv[++i]); }
    }
    else { UTIL_THROW(util::Exception, "Unknown option " << a << " (try --help)"); }
  }
  // 对应 boost 的 implicit_value("0.5 1 1.5")：只写 --discount_fallback
  // 而不给值时用默认折扣，给了值才覆盖。
  if (o.has_discount_fallback && !disc_tmp.empty()) o.discount_fallback = disc_tmp;
  return o;
}

} // namespace

int main(int argc, char *argv[]) {
  try {
    lm::builder::PipelineConfig pipeline;

    if (argc == 1) { PrintHelp(); return 1; }
    Options opt = ParseArgs(argc, argv);
    if (opt.help) { PrintHelp(); return 1; }
    UTIL_THROW_IF(!opt.has_order, util::Exception, "the option '--order' is required but missing");

    // ---- 以下与官方 lmplz_main.cc 逐行对应 ----
    pipeline.order = opt.order;
    pipeline.sort.temp_prefix = opt.temp_prefix;
    pipeline.sort.total_memory = 0; SetSize(pipeline.sort.total_memory, opt.memory);
    pipeline.minimum_block = 0;     SetSize(pipeline.minimum_block, opt.minimum_block);
    pipeline.sort.buffer_size = 0;  SetSize(pipeline.sort.buffer_size, opt.sort_block);
    pipeline.block_count = opt.block_count;
    pipeline.vocab_estimate = opt.vocab_estimate;
    pipeline.vocab_size_for_unk = opt.vocab_pad;
    pipeline.initial_probs.interpolate_unigrams = opt.interpolate_unigrams;

    if (pipeline.vocab_size_for_unk && !pipeline.initial_probs.interpolate_unigrams) {
      std::cerr << "--vocab_pad requires --interpolate_unigrams be on" << std::endl;
      return 1;
    }

    pipeline.disallowed_symbol_action = opt.skip_symbols ? lm::COMPLAIN : lm::THROW_UP;

    if (opt.has_discount_fallback) {
      pipeline.discount.fallback = ParseDiscountFallback(opt.discount_fallback);
      pipeline.discount.bad_action = lm::COMPLAIN;
    } else {
      pipeline.discount.fallback = lm::builder::Discount();
      pipeline.discount.bad_action = lm::THROW_UP;
    }

    pipeline.prune_thresholds = ParsePruning(opt.pruning, pipeline.order);
    pipeline.prune_vocab_file = opt.limit_vocab_file;
    pipeline.prune_vocab = !opt.limit_vocab_file.empty();
    pipeline.renumber_vocabulary = opt.renumber;
    pipeline.output_q = opt.collapse_values;

    util::NormalizeTempPrefix(pipeline.sort.temp_prefix);

    lm::builder::InitialProbabilitiesConfig &initial = pipeline.initial_probs;
    initial.adder_in.total_memory = 32768;
    initial.adder_in.block_count = 2;
    initial.adder_out.total_memory = 32768;
    initial.adder_out.block_count = 2;
    pipeline.read_backoffs = initial.adder_out;

    util::scoped_fd in(0), out(1);
    if (opt.has_text) in.reset(util::OpenReadOrThrow(opt.text.c_str()));
    if (opt.has_arpa) out.reset(util::CreateOrThrow(opt.arpa.c_str()));

    try {
      bool writing_intermediate = opt.has_intermediate;
      if (writing_intermediate) pipeline.renumber_vocabulary = true;
      lm::builder::Output output(writing_intermediate ? opt.intermediate : pipeline.sort.temp_prefix,
                                 writing_intermediate, pipeline.output_q);
      if (!writing_intermediate || opt.has_arpa) {
        output.Add(new lm::builder::PrintHook(out.release(), opt.verbose_header));
      }
      lm::builder::Pipeline(pipeline, in.release(), output);
    } catch (const util::MallocException &e) {
      std::cerr << e.what() << std::endl;
      std::cerr << "Try rerunning with a more conservative -S setting than " << opt.memory << std::endl;
      return 1;
    }
    util::PrintUsage(std::cerr);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
