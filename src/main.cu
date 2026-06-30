#include <cstdlib>
#include <iostream>
#include <string>

#include "tes.cuh"

namespace {

constexpr int K_DEFAULT_GRID_QUALITY = 128;
constexpr double K_DEFAULT_ALPHA = 0.999;
constexpr double K_DEFAULT_EPSILON = 1e-6;

void print_usage(char const* prog) {
  std::cerr << "Usage: " << prog << " --input <inner.msh> --output <out.json>\n"
            << "                       [--outer <outer.msh>]\n"
            << "                       [--grid-quality <int>     (default " << K_DEFAULT_GRID_QUALITY
            << ")]\n"
            << "                       [--alpha <double>         (default " << K_DEFAULT_ALPHA
            << ")]\n"
            << "                       [--epsilon <double>       (default " << K_DEFAULT_EPSILON
            << ")]\n"
            << "                       [--deformable]\n";
}

[[noreturn]] void fail_with_usage(char const* prog, std::string const& message) {
  std::cerr << "[ERROR] " << message << "\n";
  print_usage(prog);
  std::exit(EXIT_FAILURE);
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path;
  std::string outer_path;
  std::string output_path;
  int grid_quality = K_DEFAULT_GRID_QUALITY;
  double alpha = K_DEFAULT_ALPHA;
  double epsilon = K_DEFAULT_EPSILON;
  bool is_deformable = false;

  auto const consume_value = [&](int& i, char const* flag) -> std::string {
    if (i + 1 >= argc) {
      fail_with_usage(argv[0], std::string("Missing value for ") + flag);
    }
    return argv[++i];
  };

  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    if (arg == "--input" || arg == "-i") {
      input_path = consume_value(i, "--input");
    } else if (arg == "--outer") {
      outer_path = consume_value(i, "--outer");
    } else if (arg == "--output" || arg == "-o") {
      output_path = consume_value(i, "--output");
    } else if (arg == "--grid-quality" || arg == "--grid_quality") {
      grid_quality = std::stoi(consume_value(i, "--grid-quality"));
    } else if (arg == "--alpha") {
      alpha = std::stod(consume_value(i, "--alpha"));
    } else if (arg == "--epsilon") {
      epsilon = std::stod(consume_value(i, "--epsilon"));
    } else if (arg == "--deformable") {
      is_deformable = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      fail_with_usage(argv[0], "Unknown argument: " + arg);
    }
  }

  if (input_path.empty()) {
    fail_with_usage(argv[0], "--input is required");
  }
  if (output_path.empty()) {
    fail_with_usage(argv[0], "--output is required");
  }

  tes::TES encoder;

  if (!encoder.Load_Inner_MSH(input_path)) {
    return EXIT_FAILURE;
  }
  if (!outer_path.empty() && !encoder.Load_Outer_MSH(outer_path)) {
    return EXIT_FAILURE;
  }
  if (!encoder.Make_Geometry()) {
    return EXIT_FAILURE;
  }
  if (!encoder.Encode_With_Optimization(grid_quality, alpha, epsilon, is_deformable)) {
    return EXIT_FAILURE;
  }

  encoder.Print_Statistics();

  if (!encoder.Export(output_path)) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
