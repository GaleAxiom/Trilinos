/*
 * Copyright(C) 1999-2025 National Technology & Engineering Solutions
 * of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
 * NTESS, the U.S. Government retains certain rights in this software.
 *
 * See packages/seacas/LICENSE for details
 */
#include "CP_SystemInterface.h"
#include "CP_Version.h" // for qainfo
#include "FileInfo.h"
#include "GetLongOpt.h" // for GetLongOption, etc
#include "Ioss_CodeTypes.h"
#include "Ioss_Utils.h"
#include "SL_tokenize.h" // for tokenize
#include <algorithm>     // for sort, transform
#include <cctype>        // for tolower
#include <copyright.h>
#include <cstdlib> // for strtol, abs, exit, strtoul, etc
#include <cstring> // for strchr, strlen
#include <fmt/ostream.h>
#include <glob.h>
#include <sstream>
#include <stdexcept>
#include <string> // for string, char_traits, etc
#include <term_width.h>
#include <unistd.h>
#include <utility> // for pair, make_pair
#include <vector>  // for vector

#if defined(WIN32) || defined(__WIN32__) || defined(_WIN32) || defined(_MSC_VER) ||                \
    defined(__MINGW32__) || defined(_WIN64) || defined(__MINGW64__)
#define __SUP_WINDOWS__ 1
#include <direct.h>
#endif

#if !defined(__SUP_WINDOWS__)
#include <dirent.h>
#endif

namespace {
  bool str_equal(const std::string &s1, const std::string &s2)
  {
    return (s1.size() == s2.size()) &&
           std::equal(s1.begin(), s1.end(), s2.begin(),
                      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
  }

  void        parse_variable_names(const char *tokens, Cpup::StringVector *variable_list);
  std::string find_matching_file(const std::string &path, const std::string &basename);
} // namespace

Cpup::SystemInterface::SystemInterface(int rank) : myRank_(rank) { enroll_options(); }

void Cpup::SystemInterface::enroll_options()
{
  options_.usage("[options] basename");

  options_.enroll("auto", GetLongOption::NoValue,
                  "Automatically set Root, Proc, Ext from filename 'Root/basename.ext.#p.00'.",
                  nullptr, nullptr, true);
  options_.enroll("variables", GetLongOption::MandatoryValue,
                  "Comma-separated list of variables (fields) to be joined or ALL or NONE.",
                  nullptr);
  options_.enroll("steps", GetLongOption::MandatoryValue,
                  "Specify subset of timesteps to transfer to output file.\n"
                  "\t\tFormat is beg:end:step. 1:10:2 --> 1,3,5,7,9\n"
                  "\t\tIf the 'beg' or 'end' is < 0, then it is the \"-Nth\" step...\n"
                  "\t\t-1 is \"first last\" or last, -3 is \"third last\"\n"
                  "\t\tTo copy just the last 3 steps, do: `-steps -3:-1`\n"
                  "\t\tEnter LAST for last step",
                  "1:", nullptr, true);

  options_.enroll("extension", GetLongOption::MandatoryValue,
                  "CGNS database extension for the input files", "cgns");

  options_.enroll("output", GetLongOption::MandatoryValue, "filename for output CGNS database",
                  nullptr);

  options_.enroll("output_extension", GetLongOption::MandatoryValue,
                  "CGNS database extension for the output file", nullptr);

  options_.enroll("processor_count", GetLongOption::MandatoryValue, "Number of processors", "1");

  options_.enroll("current_directory", GetLongOption::MandatoryValue, "Current Directory", ".");

  options_.enroll("Root_directory", GetLongOption::MandatoryValue, "Root directory", nullptr);

  options_.enroll("Subdirectory", GetLongOption::MandatoryValue,
                  "subdirectory containing input cgns files", nullptr, nullptr, true);

#if 0
  options_.enroll("subcycle", GetLongOption::OptionalValue,
                  "Subcycle. Create $val subparts if $val is specified.\n"
                  "\t\tOtherwise, create multiple parts each of size 'Part_count'.\n"
                  "\t\tThe subparts can then be joined by a subsequent run of cpup.\n"
                  "\t\tUseful if the maximum number of open files is less\n"
                  "\t\tthan the processor count.",
                  nullptr, "0");

  options_.enroll("join_subcycles", GetLongOption::NoValue,
                  "If -subcycle is specified, then after the subcycle files are processed,\n"
                  "\t\trun cpup one more time and join the subcycle files into a single file.",
                  nullptr);

  options_.enroll("Part_count", GetLongOption::MandatoryValue,
                  "How many pieces (files) of the model should be joined.", "0");

  options_.enroll("start_part", GetLongOption::MandatoryValue, "Start with piece {L} (file)", "0");

  options_.enroll("cycle", GetLongOption::MandatoryValue,
                  "Cycle number. If subcycle # is specified, then only execute\n"
                  "\t\tcycle $val ($val < #).  The cycle number is 0-based.",
                  "-1");

  options_.enroll("keep_temporary", GetLongOption::NoValue,
                  "If -join_subcycles is specified, then after joining the subcycle files,\n"
                  "\t\tthey are automatically deleted unless -keep_temporary is specified.",
                  nullptr);

  options_.enroll(
      "verify_valid_file", GetLongOption::NoValue,
      "Reopen the output file right after closing it to verify that the file is valid.\n"
      "\t\tThis tries to detect file corruption immediately instead of later. Mainly useful in "
      "large subcycle runs.",
      nullptr);

  options_.enroll("append", GetLongOption::NoValue,
                  "Append to database instead of opening a new database.\n"
                  "\t\tTimestep transfer will start after last timestep on database",
                  nullptr);

  options_.enroll("add_processor_id", GetLongOption::NoValue,
                  "Add 'processor_id' element variable to the output file which shows the\n"
                  "\t\tprocessor that an element was on in the decomposed mesh.\n"
                  "\t\tCan be used by SLICE or auto-decomp to reproduce decomposition.",
                  nullptr);

  options_.enroll("add_map_processor_id", GetLongOption::NoValue,
                  "Add 'processor_id' element map to the output file which shows the\n"
                  "\t\tprocessor that an element was on in the decomposed mesh.\n"
                  "\t\tCan be used by SLICE or auto-decomp to reproduce decomposition.",
                  nullptr, nullptr, true);

  options_.enroll("gvar", GetLongOption::MandatoryValue,
                  "Comma-separated list of global variables to be joined or ALL or NONE.", nullptr);

  options_.enroll("evar", GetLongOption::MandatoryValue,
                  "Comma-separated list of element variables to be joined or ALL or NONE.\n"
                  "\t\tVariables can be limited to certain blocks by appending a\n"
                  "\t\tcolon followed by the block id.  E.g. -evar sigxx:10:20",
                  nullptr);

  options_.enroll("nvar", GetLongOption::MandatoryValue,
                  "Comma-separated list of nodal variables to be joined or ALL or NONE.", nullptr);

  options_.enroll("nsetvar", GetLongOption::MandatoryValue,
                  "Comma-separated list of nodeset variables to be joined or ALL or NONE.",
                  nullptr);

  options_.enroll("ssetvar", GetLongOption::MandatoryValue,
                  "Comma-separated list of sideset variables to be joined or ALL or NONE.",
                  nullptr);

  options_.enroll("omit_nodesets", GetLongOption::NoValue,
                  "Don't transfer nodesets to output file.", nullptr);

  options_.enroll("omit_sidesets", GetLongOption::NoValue,
                  "Don't transfer sidesets to output file.", nullptr, nullptr, true);

  options_.enroll("sum_shared_nodes", GetLongOption::NoValue,
                  "The nodal results data on all shared nodes (nodes on processor boundaries)\n"
                  "\t\twill be the sum of the individual nodal results data on each shared node.\n"
                  "\t\tThe default behavior assumes that the values are equal.",
                  nullptr);

  options_.enroll("max_open_files", GetLongOption::MandatoryValue,
                  "For testing auto subcycle only.  Sets file limit that triggers auto subcycling.",
                  "0");

#endif
  options_.enroll(
      "minimize_open_files", GetLongOption::NoValue,
      "Keep only one input file open at a time.\n"
      "\t\tUsed when number of files exceeds the maximum number of open files on a system.",
      nullptr);

  options_.enroll("width", GetLongOption::MandatoryValue, "Width of output screen, default = 80",
                  nullptr);

  options_.enroll("debug", GetLongOption::MandatoryValue,
                  "debug level (values are or'd)\n"
                  "\t\t  1 = Timestamp output.\n"
                  "\t\t  2 = Output region summary for each input part mesh.\n"
                  "\t\t  4 = Verbose Structured block information.",
                  "0");
  options_.enroll("copyright", GetLongOption::NoValue, "Show copyright and license data.", nullptr);
  options_.enroll("help", GetLongOption::NoValue, "Print this summary and exit", nullptr);

  options_.enroll("version", GetLongOption::NoValue, "Print version and exit", nullptr, nullptr);
}

bool Cpup::SystemInterface::parse_options(int argc, char **argv)
{
  // Get options from environment variable also...
  char *options = getenv("CPUP_OPTIONS");
  if (options != nullptr) {
    if (myRank_ == 0) {
      fmt::print(
          "\nThe following options were specified via the CPUP_OPTIONS environment variable:\n"
          "\t{}\n\n",
          options);
    }
    options_.parse(options, GetLongOption::basename(*argv));
  }

  int option_index = options_.parse(argc, argv);
  if (option_index < 1) {
    throw std::runtime_error("ERROR: (CPUP) No arguments specified.");
  }

  if (options_.retrieve("help") != nullptr) {
    if (myRank_ == 0) {
      options_.usage();
      fmt::print("\n\tCan also set options via CPUP_OPTIONS environment variable.\n\n"
                 "\n\tDocumentation: "
                 "https://sandialabs.github.io/seacas-docs/sphinx/html/index.html#cpup\n"
                 "\tWrites: current_directory/basename.output_extension\n"
                 "\tReads:  root/sub/basename.extension.#p.0 to\n"
                 "\t\troot/sub/basename.extension.#p.#p-1\n"
                 "\n\t->->-> Send email to gdsjaar@sandia.gov for cpup support.<-<-<-\n");
    }
    return false;
  }

  if (options_.retrieve("version") != nullptr) {
    // Version is printed up front, just exit...
    return false;
  }

  inExtension_    = options_.get_option_value("extension", inExtension_);
  outExtension_   = options_.get_option_value("output_extension", outExtension_);
  outputFilename_ = options_.get_option_value("output", outputFilename_);
  //  partCount_      = options_.get_option_value("Part_count", partCount_);
  //  startPart_      = options_.get_option_value("start_part", startPart_);
  //  maxOpenFiles_   = options_.get_option_value("max_open_files", maxOpenFiles_);
  cwd_           = options_.get_option_value("current_directory", cwd_);
  rootDirectory_ = options_.get_option_value("Root_directory", rootDirectory_);
  subDirectory_  = options_.get_option_value("Subdirectory", subDirectory_);
  debugLevel_    = options_.get_option_value("debug", debugLevel_);

  screenWidth_ = options_.get_option_value("width", term_width());

  {
    const char *temp = options_.retrieve("steps");
    if (temp != nullptr) {
      parse_step_option(temp);
    }
  }

  {
    const char *temp = options_.retrieve("variables");
    parse_variable_names(temp, &varNames_);
  }

#if 0
  {
    const char *temp = options_.retrieve("nvar");
    parse_variable_names(temp, &nodeVarNames_);
  }

  {
    const char *temp = options_.retrieve("evar");
    parse_variable_names(temp, &elemVarNames_);
  }

  {
    const char *temp = options_.retrieve("nsetvar");
    parse_variable_names(temp, &nsetVarNames_);
  }

  {
    const char *temp = options_.retrieve("ssetvar");
    parse_variable_names(temp, &ssetVarNames_);
  }

  addProcessorIdField_ = options_.retrieve("add_processor_id") != nullptr;
  addProcessorIdMap_   = options_.retrieve("add_map_processor_id") != nullptr;

  append_ = options_.retrieve("append") != nullptr;

  append_ = options_.retrieve("append") != nullptr;
  //subcycle_        = options_.get_option_value("subcycle", subcycle_);
  cycle_           = options_.get_option_value("cycle", cycle_);
  subcycleJoin_    = options_.retrieve("join_subcycles") != nullptr;
  keepTemporary_   = options_.retrieve("keep_temporary") != nullptr;
  verifyValidFile_ = options_.retrieve("verify_valid_file") != nullptr;

  omitNodesets_ = options_.retrieve("omit_nodesets") != nullptr;
  omitSidesets_ = options_.retrieve("omit_sidesets") != nullptr;
#endif

  minimizeOpenFiles_ = options_.retrieve("minimize_open_files") != nullptr;

  if (options_.retrieve("copyright") != nullptr) {
    if (myRank_ == 0) {
      fmt::print("{}", copyright("2010-2022"));
    }
    return false;
  }

  // Parse remaining options as directory paths.
  if (option_index < argc) {
    basename_ = argv[option_index];

    if (options_.retrieve("auto") != nullptr) {
      // Determine Root, Proc, Extension, and Basename automatically
      // by parsing the basename_ entered by the user.  Assumed to be
      // in the form: "/directory/sub/basename.ext.#proc.34"
      FileInfo file(basename_);
      auto     path = file.pathname();
      if (path.empty()) {
        path = ".";
      }
#if defined(__SUP_WINDOWS__)
      rootDirectory_ = _fullpath(nullptr, path.c_str(), _MAX_PATH);
#else
      char *tmp = ::realpath(path.c_str(), nullptr);
      if (tmp != nullptr) {
        rootDirectory_ = std::string(tmp);
        free(tmp);
      }
#endif

      basename_ = file.tailname();
      if (basename_.empty()) {
        std::ostringstream errmsg;
        fmt::print(
            errmsg,
            "\nERROR: (CPUP) If the '-auto' option is specified, the basename must specify an "
            "existing filename or portion of a base of a filename (no rank/proc count).\n"
            "       The entered basename ('{}') does not contain a filename.\n",
            basename_);
        throw std::runtime_error(errmsg.str());
      }
      bool success = decompose_filename(basename_);
      if (!success) {
        // See if we can find files that match the basename and take the first match as the "new"
        // basename...
        std::string candidate = find_matching_file(rootDirectory_, basename_);
        if (!candidate.empty()) {
          basename_ = candidate;
          success   = decompose_filename(basename_);
          if (!success) {
            std::ostringstream errmsg;
            fmt::print(
                errmsg,
                "\nERROR: (CPUP) If the '-auto' option is specified, the basename must specify an "
                "existing filename or a basename (no rank/proc count).\n"
                "       The entered basename ('{}') does not contain an extension or processor "
                "count.\n",
                basename_);
            throw std::runtime_error(errmsg.str());
          }
        }
      }
      auto_ = true;
      if (myRank_ == 0) {
        fmt::print("\nThe following options were determined automatically:\n"
                   "\t basename = '{}'\n"
                   "\t-processor_count {}\n"
                   "\t-extension {}\n"
                   "\t-Root_directory {}\n\n",
                   basename_, processorCount_, inExtension_, rootDirectory_);
      }
    }
  }
  else {
    throw std::runtime_error("\nERROR: (CPUP) basename not specified\n");
  }

#if 0
  // Check that subcycle count does not match processor count --
  // in that case the existing files will be overwritten.
  if (processorCount_ <= subcycle_) {
    if (myRank_ == 0) {
      fmt::print(stderr,
                 "\nERROR: (CPUP) Invalid subcycle count specified: '{}'."
                 "\n             Must be less than processor count '{}'.\n\n",
                 subcycle_, processorCount_);
    }
    return false;
  }

  // If subcycle is specified, but not part_count, then calculate partCount_
  if (partCount_ <= 0 && subcycle_ > 0) {
    partCount_ = processorCount_ / subcycle_;
  }
#endif
  return true;
}

std::string Cpup::SystemInterface::output_filename() const
{
  if (outputFilename_.empty()) {
    outputFilename_ = basename();
    if (!output_suffix().empty()) {
      outputFilename_ += "." + output_suffix();
    }

    if (!cwd().empty() && !Ioss::Utils::is_path_absolute(outputFilename_)) {
      outputFilename_ = cwd() + "/" + outputFilename_;
    }
    set_output_filename(outputFilename_);
  }
  return outputFilename_;
}

void Cpup::SystemInterface::dump(std::ostream & /*unused*/) const {}

std::string Cpup::SystemInterface::output_suffix() const
{
  if (outExtension_.empty()) {
    return inExtension_;
  }
  return outExtension_;
}

void Cpup::SystemInterface::show_version(int rank)
{
  if (rank == 0) {
    fmt::print("{}\n"
               "\tCGNS Parallel Unification Program\n"
               "\t(Version: {}) Modified: {}\n",
               qainfo[0], qainfo[1], qainfo[2]);
  }
}

void Cpup::SystemInterface::parse_step_option(const char *tokens)
{
  //: The defined formats for the count attribute are:<br>
  //:  <ul>
  //:    <li><missing> -- default -- 1 <= count <= oo  (all steps)</li>
  //:    <li>"X"                  -- X <= count <= X  (just step X) LAST for last step.</li>
  //:    <li>"X:Y"                -- X to Y by 1</li>
  //:    <li>"X:"                 -- X to oo by 1</li>
  //:    <li>":Y"                 -- 1 to Y by 1</li>
  //:    <li>"::Z"                -- 1 to oo by Z</li>
  //:  </ul>
  //: The step must always be > 0
  //: If the 'from' or 'to' is < 0, then it is the "-Nth" step...
  //: -1 is "first last" or last
  //: -4 is "fourth last step"
  //: To copy just the last 3 steps, do: `-steps -3:-1`

  // Break into tokens separated by ":"

  // Default is given in constructor above...

  if (tokens != nullptr) {
    if (strchr(tokens, ':') != nullptr) {
      // The string contains a separator

      std::array<int, 3> vals{stepMin_, stepMax_, stepInterval_};

      int j = 0;
      for (auto &val : vals) {
        // Parse 'i'th field
        char tmp_str[128];

        int k = 0;
        while (tokens[j] != '\0' && tokens[j] != ':') {
          tmp_str[k++] = tokens[j++];
        }

        tmp_str[k] = '\0';
        if (strlen(tmp_str) > 0) {
          val = std::stoi(tmp_str);
        }

        if (tokens[j++] == '\0') {
          break; // Reached end of string
        }
      }
      stepMin_      = vals[0];
      stepMax_      = vals[1];
      stepInterval_ = abs(vals[2]); // step is always positive...
    }
    else if (str_equal("LAST", tokens)) {
      stepMin_ = stepMax_ = -1;
    }
    else {
      // Does not contain a separator, min == max
      stepMin_ = stepMax_ = std::stoi(tokens);
    }
  }
}

bool Cpup::SystemInterface::decompose_filename(const std::string &cs)
{
  std::string s(cs);
  // Decompose the input string 's' into processor count, extension, basename, and root.
  // Input string should be of the form:
  // "root/basename.ext.proc.nn"
  // 'root/' is optional and is all characters preceding the last (if any) '/';
  // 'basename' can contain multiple '.'

  // NOTE: This used to use the tokenize function, but that didn't work since we need
  // to handle leading and embedded '..' which tokenize threw away...

  // Get rid of the 'nn' which is not used at this time...
  size_t ind = s.find_last_of('.', std::string::npos); // last '.'
  if (ind == std::string::npos) {
    return false;
  }
  s.erase(ind);

  // Now find the processor count...
  ind = s.find_last_of('.', std::string::npos);
  if (ind == std::string::npos) {
    return false;
  }

  std::string tmp = s.substr(ind + 1); // Skip the '.'
  processorCount_ = std::stoi(tmp);
  if (processorCount_ <= 0) {
    fmt::print(
        stderr,
        "\nERROR: (CPUP) Invalid processor count specified: '{}'. Must be greater than zero.\n",
        processorCount_);
    return false;
  }
  s.erase(ind);

  // Should now be an extension...
  ind = s.find_last_of('.', std::string::npos);
  if (ind == std::string::npos) {
    inExtension_ = "";
  }
  else {
    inExtension_ = s.substr(ind + 1);
    s.erase(ind);
  }

  // The directory path was stripped prior to entering this function,
  // so remainder of 's' is just the new basename_
  basename_ = s;
  return true;
}

namespace {
  std::string LowerCase(const std::string &name)
  {
    std::string s = name;
    std::transform(s.begin(), s.end(), // source
                   s.begin(),          // destination
                   ::tolower);         // operation
    return s;
  }

  void parse_variable_names(const char *tokens, Cpup::StringVector *variable_list)
  {
    // Break into tokens separated by ","
    if (tokens != nullptr) {
      std::string        token_string(tokens);
      Cpup::StringVector var_list = SLIB::tokenize(token_string, ",");
      for (const auto &var : var_list) {
        std::string low_var = LowerCase(var);
        (*variable_list).push_back(low_var);
      }
      // Sort the list...
      std::sort(variable_list->begin(), variable_list->end());
    }
  }

  std::string find_matching_file(const std::string &path, const std::string &basename)
  {
    glob::glob g(basename + ".*.*");
#if !defined(__SUP_WINDOWS__)
    struct dirent *entry = nullptr;
    DIR           *dp    = nullptr;

    dp = opendir(path.c_str());
    if (dp != nullptr) {
      while ((entry = readdir(dp))) {
        std::string filename = entry->d_name;
        if (glob::glob_match(filename, g)) {
          closedir(dp);
          return filename;
        }
      }
    }
    closedir(dp);
#endif
    return "";
  }
} // namespace
