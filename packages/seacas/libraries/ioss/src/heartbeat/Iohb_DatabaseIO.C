// Copyright(C) 1999-2024 National Technology & Engineering Solutions
// of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
// NTESS, the U.S. Government retains certain rights in this software.
//
// See packages/seacas/LICENSE for details

#include "Ioss_CodeTypes.h"
#include "heartbeat/Iohb_DatabaseIO.h"
#include "heartbeat/Iohb_Layout.h"
#include <cassert>
#include <cstddef>
#include <ctime>
#include <fmt/ostream.h>
#include <fstream>
#include <string>
#include <vector>

#include "Ioss_DBUsage.h"
#include "Ioss_DatabaseIO.h"
#include "Ioss_EntityType.h"
#include "Ioss_Field.h"
#include "Ioss_IOFactory.h"
#include "Ioss_ParallelUtils.h"
#include "Ioss_Property.h"
#include "Ioss_PropertyManager.h"
#include "Ioss_State.h"
#include "Ioss_Utils.h"

namespace {
  std::string time_stamp(const std::string &format)
  {
    if (format.empty()) {
      return {""};
    }
    const int   length = 256;
    static char time_string[length];

    time_t     calendar_time = time(nullptr);
    struct tm *local_time    = localtime(&calendar_time);

    size_t error = strftime(time_string, length, format.c_str(), local_time);
    if (error != 0) {
      time_string[length - 1] = '\0';
      return {time_string};
    }
    return {"[ERROR]"};
  }

  std::ostream *open_stream(const std::string &filename, bool *needs_delete, bool append_file)
  {
    // A little weirdness and ambiguity is possible here.  We want to
    // minimize the number of commands, but maximize the
    // functionality. For example, we want to be able to specify output
    // to existing streams (cout/stdout, cerr/stderr, outputP0), files,
    // sockets, XML-RPC?, SOAP?.  However, we want to be able to specify
    // this with a single command.
    // So..., we first check for some 'reserved' stream names.  These
    // are the 'cout, stdout, cerr, stderr, output, outputP0' type.
    // Note that this means the user can't specify a file name of
    // 'cerr', but that shouldn't be too much of a hardship.  [If it is,
    // we can devise a syntax as: 'FILE:cout' or do a './cout'

    std::ostream *log_stream = nullptr;
    *needs_delete            = false;
    if (filename == "cout" || filename == "stdout") {
      log_stream = &std::cout;
    }
    else if (filename == "cerr" || filename == "stderr") {
      log_stream = &std::cerr; // This is also default if nothing specified
    }
    else if (filename == "output" || filename == "outputP0") {
      log_stream = &std::cout; // This should be sierra::Env::outputP0(), but not
                               // available during transition to stk...
    }
    else if (filename == "clog" || filename == "log") {
      log_stream = &std::clog; // Same as cerr, but not flushed automatically.
    }
    else {
      // Open the file (on processor 0 only) Might need to do
      // something better here if we want to share streams among
      // different heartbeats or logging mechanisms.  Need perhaps a
      // 'logger' class which handles sharing and destruction...
      std::ofstream *tmp = nullptr;
      if (append_file) {
        tmp = new std::ofstream(filename, std::ios::out | std::ios::app);
      }
      else {
        tmp = new std::ofstream(filename);
      }
      if (!tmp->is_open()) {
        delete tmp;
      }
      else {
        log_stream    = tmp;
        *needs_delete = true;
      }
    }
    return log_stream;
  }
} // namespace

namespace Iohb {

  // ========================================================================
  const IOFactory *IOFactory::factory()
  {
    static IOFactory registerThis;
    return &registerThis;
  }

  IOFactory::IOFactory() : Ioss::IOFactory("heartbeat") {}

  Ioss::DatabaseIO *IOFactory::make_IO(const std::string &filename, Ioss::DatabaseUsage db_usage,
                                       Ioss_MPI_Comm                communicator,
                                       const Ioss::PropertyManager &props) const
  {
    return new DatabaseIO(nullptr, filename, db_usage, communicator, props);
  }

  // ========================================================================
  DatabaseIO::DatabaseIO(Ioss::Region *region, const std::string &filename,
                         Ioss::DatabaseUsage db_usage, Ioss_MPI_Comm communicator,
                         const Ioss::PropertyManager &props)
      : Ioss::DatabaseIO(region, filename, db_usage, communicator, props)
  {
    timeLastFlush_ = time(nullptr);
    dbState        = Ioss::STATE_UNKNOWN;
  }

  DatabaseIO::~DatabaseIO()
  {
    if (streamNeedsDelete && (logStream != nullptr)) {
      delete logStream;
    }
  }

  void DatabaseIO::initialize() const
  {
    if (!initialized_) {
      auto *new_this = const_cast<DatabaseIO *>(this);

      if (properties.exists("FILE_FORMAT")) {
        std::string format = properties.get("FILE_FORMAT").get_string();
        if (Ioss::Utils::str_equal(format, "spyhis")) {
          new_this->fileFormat = Iohb::Format::SPYHIS;
        }
        else if (Ioss::Utils::str_equal(format, "csv")) {
          new_this->fileFormat = Iohb::Format::CSV;
        }
        else if (Ioss::Utils::str_equal(format, "ts_csv")) {
          new_this->fileFormat = Iohb::Format::TS_CSV;
        }
        else if (Ioss::Utils::str_equal(format, "text")) {
          new_this->fileFormat = Iohb::Format::TEXT;
        }
        else if (Ioss::Utils::str_equal(format, "ts_text")) {
          new_this->fileFormat = Iohb::Format::TS_TEXT;
        }
      }

      bool append = open_create_behavior() == Ioss::DB_APPEND;

      // Try to open file...
      new_this->logStream = nullptr;
      if (util().parallel_rank() == 0) {
        new_this->logStream = open_stream(get_filename(), &(new_this->streamNeedsDelete), append);

        if (new_this->logStream == nullptr) {
          IOSS_ERROR(fmt::format("ERROR: Could not create heartbeat file '{}'\n", get_filename()));
        }
      }

      // "Predefined" formats... (put first so can modify settings if wanted)
      if (fileFormat == Iohb::Format::CSV) {
        new_this->addTimeField = true;
        new_this->showLegend   = true;
        new_this->showLabels   = false;
        new_this->separator_   = ", ";
      }
      else if (fileFormat == Iohb::Format::TS_CSV) {
        new_this->addTimeField = true;
        new_this->showLegend   = true;
        new_this->showLabels   = false;
        new_this->separator_   = ", ";
        new_this->tsFormat     = defaultTsFormat;
      }
      else if (fileFormat == Iohb::Format::TEXT) {
        new_this->addTimeField = true;
        new_this->showLegend   = true;
        new_this->showLabels   = false;
        new_this->separator_   = "\t";
      }
      else if (fileFormat == Iohb::Format::TS_TEXT) {
        new_this->addTimeField = true;
        new_this->showLegend   = true;
        new_this->showLabels   = false;
        new_this->separator_   = "\t";
        new_this->tsFormat     = defaultTsFormat;
      }

      // Pull variables from the regions property data...
      if (properties.exists("FIELD_SEPARATOR")) {
        new_this->separator_ = properties.get("FIELD_SEPARATOR").get_string();
      }

      if (properties.exists("FLUSH_INTERVAL")) {
        new_this->flushInterval_ = properties.get("FLUSH_INTERVAL").get_int();
      }

      if (properties.exists("HEARTBEAT_FLUSH_INTERVAL")) {
        new_this->flushInterval_ = properties.get("HEARTBEAT_FLUSH_INTERVAL").get_int();
      }

      if (properties.exists("TIME_STAMP_FORMAT")) {
        new_this->tsFormat = properties.get("TIME_STAMP_FORMAT").get_string();
      }

      bool show_time_stamp = !new_this->tsFormat.empty();
      Ioss::Utils::check_set_bool_property(properties, "SHOW_TIME_STAMP", show_time_stamp);
      if (show_time_stamp) {
        if (tsFormat.empty()) {
          new_this->tsFormat = defaultTsFormat;
        }
      }
      else {
        new_this->tsFormat = "";
      }

      if (properties.exists("PRECISION")) {
        new_this->precision_ = properties.get("PRECISION").get_int();
      }

      if (properties.exists("FIELD_WIDTH")) {
        new_this->fieldWidth_ = properties.get("FIELD_WIDTH").get_int();
      }
      else {
        // +1.xxxxxxe+00 The x count is the precision the "+1.e+00" is the 7
        new_this->fieldWidth_ = precision_ + 7;
      }

      Ioss::Utils::check_set_bool_property(properties, "SHOW_LABELS", new_this->showLabels);

      if (!new_this->appendOutput) {
        Ioss::Utils::check_set_bool_property(properties, "SHOW_LEGEND", new_this->showLegend);
      }

      Ioss::Utils::check_set_bool_property(properties, "SHOW_TIME_FIELD", new_this->addTimeField);

      if (properties.exists("FULL_PRECISION")) {
        bool precision = true;
        Ioss::Utils::check_set_bool_property(properties, "FULL_PRECISION", precision);
        if (precision) {
          new_this->precision_ = -1;
        }
      }

      // SpyHis format is specific format, so don't override these settings:
      if (fileFormat == Iohb::Format::SPYHIS) {
        new_this->addTimeField = true;
        new_this->showLegend   = true;
        new_this->showLabels   = false;
        new_this->tsFormat     = "";
      }

      if (showLegend) {
        new_this->legend_ = std::make_unique<Layout>(false, precision_, separator_, fieldWidth_);
        if (!tsFormat.empty()) {
          new_this->legend_->add_literal("+");
          new_this->legend_->add_literal(time_stamp(tsFormat));
          new_this->legend_->add_literal(" ");
        }

        if (addTimeField) {
          if (fileFormat == Iohb::Format::SPYHIS) {
            new_this->legend_->add_legend("TIME");
          }
          else {
            new_this->legend_->add_legend("Time");
          }
        }
      }

      new_this->initialized_ = true;
    }
  }

  bool DatabaseIO::begin_nl(Ioss::State /* state */) { return true; }

  bool DatabaseIO::end_nl(Ioss::State /* state */) { return true; }

  bool DatabaseIO::begin_state_nl(int /* state */, double time)
  {
    // If this is the first time, open the output stream and see if user wants a legend
    initialize();

    layout_ = std::make_unique<Layout>(showLabels, precision_, separator_, fieldWidth_);
    if (!tsFormat.empty()) {
      layout_->add_literal("+");
      layout_->add_literal(time_stamp(tsFormat));
      layout_->add_literal(" ");
    }

    if (addTimeField) {
      layout_->add("TIME", time / timeScaleFactor);
    }

    return true;
  }

  void DatabaseIO::flush_database_nl() const
  {
    if (myProcessor == 0) {
      logStream->flush();
    }
  }

  bool DatabaseIO::end_state_nl(int /* state */, double /* time */)
  {
    if (legend_ != nullptr) {
      if (fileFormat == Iohb::Format::SPYHIS) {
        time_t calendar_time = time(nullptr);
        // ctime include \n; the legend is output twice for SPYHIS.
        fmt::print(*logStream, "% Sierra SPYHIS Output {}{}\n", ctime(&calendar_time),
                   legend_->layout()); // ctime includes \n
      }

      fmt::print(*logStream, "{}\n", legend_->layout());
      legend_.reset();
    }

    fmt::print(*logStream, "{}\n", layout_->layout());
    layout_.reset();

    // Flush the buffer to disk...
    // flush if there is more than 'flushInterval_' seconds since the last flush to avoid
    // the flush eating up cpu time for small fast jobs...
    // If want a flush every step for some reason, set `flushInterval_` to 0.0.

    // This code is derived from code in finalize_write() in Ioex_DatabaseIO.C
    // See other comments there...

    time_t cur_time = time(nullptr);
    if (cur_time - timeLastFlush_ >= flushInterval_) {
      timeLastFlush_ = cur_time;
      flush_database();
    }

    return true;
  }

  int64_t DatabaseIO::put_field_internal(const Ioss::Region * /* region */,
                                         const Ioss::Field &field, void *data,
                                         size_t data_size) const
  {
    initialize();
    Ioss::Field::RoleType role       = field.get_role();
    int64_t               num_to_get = field.verify(data_size);

    if ((role == Ioss::Field::TRANSIENT || role == Ioss::Field::REDUCTION) && num_to_get == 1) {

      int ncomp = field.get_component_count(Ioss::Field::InOut::OUTPUT);

      if (legend_ != nullptr && layout_ != nullptr) {
        if (ncomp == 1) {
          legend_->add_legend(field.get_name());
        }
        else {
          for (int i = 0; i < ncomp; i++) {
            std::string var_name = get_component_name(field, Ioss::Field::InOut::OUTPUT, i + 1);
            legend_->add_legend(var_name);
          }
        }
      }

      if (field.get_type() == Ioss::Field::STRING) {
        // Assume that if layout_ is nullptr, then we want special one-line output.
        if (layout_ == nullptr) {
          Layout layout(false, 0, separator_, fieldWidth_);
          layout.add_literal("-");
          layout.add_literal(time_stamp(tsFormat));
          layout.add_literal(" ");
          layout.add_literal(*reinterpret_cast<std::string *>(data));
          if (logStream != nullptr) {
            fmt::print(*logStream, "{}\n", layout.layout());
          }
        }
        else {
          layout_->add(field.get_name(), *reinterpret_cast<std::string *>(data));
        }
      }
      else {
        if (layout_ == nullptr) {
          IOSS_ERROR("INTERNAL ERROR: Unexpected nullptr layout.\n");
        }
        if (field.get_type() == Ioss::Field::INTEGER) {
          assert(field.transformed_count() == 1);

          int             *i_data = reinterpret_cast<int *>(data);
          std::vector<int> idata(ncomp);
          for (int i = 0; i < ncomp; i++) {
            idata[i] = i_data[i];
          }
          layout_->add(field.get_name(), idata);
        }
        else {
          std::vector<double> rdata(ncomp);
          auto               *r_data = reinterpret_cast<double *>(data);
          for (int i = 0; i < ncomp; i++) {
            rdata[i] = r_data[i];
          }
          layout_->add(field.get_name(), rdata);
        }
      }
    }
    else {
      IOSS_ERROR("ERROR: Can not handle non-TRANSIENT or non-REDUCTION fields on regions.\n");
    }
    return num_to_get;
  }

  unsigned DatabaseIO::entity_field_support() const { return Ioss::REGION; }
} // namespace Iohb
