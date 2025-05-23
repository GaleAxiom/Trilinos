// Copyright(C) 1999-2025 National Technology & Engineering Solutions
// of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
// NTESS, the U.S. Government retains certain rights in this software.
//
// See packages/seacas/LICENSE for details
#pragma once

#include "exo_entity.h"
#include "util.h"

#include <iostream>
#include <string>
#include <vector>

#include "smart_assert.h"

//
//  Notes: 1) Most functions will return a string as a result.  An empty string
//            indicates success, while a non-empty string indicates an error or
//            a warning, either of which will be contained in the string.
//

class Exo_Entity;
template <typename INT> class Exo_Block;

template <typename INT> class Node_Set;
template <typename INT> class Side_Set;

template <typename INT> class Edge_Block;
template <typename INT> class Face_Block;
template <typename INT> class Assembly;

template <typename INT> class Exo_Read
{
public:
  Exo_Read() = delete;
  explicit Exo_Read(std::string fname);
  ~Exo_Read();
  const Exo_Read &operator=(const Exo_Read &) = delete;
  Exo_Read(const Exo_Read &)                  = delete;

  // File operations:

  std::string        Open_Change_Set(const std::string &name);
  std::string        Open_Change_Set(int index);
  void               Get_Meta_Data(); // Reads metadata
  int                Get_Change_Set_Index() const { return current_change_set_index; }
  const std::string &Get_Change_Set_Name() const
  {
    return change_set_names[current_change_set_index];
  }

  std::string Close_File();
  std::string File_Name() const { return file_name; }
  int         Open() const { return (file_id >= 0); }
  int         IO_Word_Size() const { return io_word_size; }

  void modify_time_values(double scale, double offset)
  {
    time_scale  = scale;
    time_offset = offset;
  }

  // Global data:
  const std::string &Title() const { return title; }
  int                Dimension() const { return dimension; }
  size_t             Num_Nodes() const { return num_nodes; }
  size_t             Num_Elements() const { return num_elmts; }
  size_t             Num_Faces() const { return num_faces; }
  size_t             Num_Edges() const { return num_edges; }
  size_t             Num_Node_Sets() const { return num_node_sets; }
  size_t             Num_Side_Sets() const { return num_side_sets; }
  size_t             Num_Edge_Blocks() const { return num_edge_blocks; }
  size_t             Num_Face_Blocks() const { return num_face_blocks; }
  size_t             Num_Assembly() const { return num_assemblies; }

  // Times:
  int    Num_Times() const { return static_cast<int>(times.size()); }
  double Time(int time_num) const;

  // Change Sets:
  int                     Num_Change_Sets() const { return num_change_sets; }
  const NameList         &Change_Set_Names() const { return change_set_names; }
  const std::vector<int> &Change_Set_Ids() const { return change_set_ids; }

  // Variables:

  size_t          Num_Global_Vars() const { return global_vars.size(); }
  size_t          Num_Nodal_Vars() const { return nodal_vars.size(); }
  size_t          Num_Element_Vars() const { return elmt_vars.size(); }
  size_t          Num_Element_Atts() const { return elmt_atts.size(); }
  size_t          Num_NS_Vars() const { return ns_vars.size(); }
  size_t          Num_SS_Vars() const { return ss_vars.size(); }
  size_t          Num_EB_Vars() const { return eb_vars.size(); }
  size_t          Num_FB_Vars() const { return fb_vars.size(); }
  const NameList &Global_Var_Names() const { return global_vars; }
  const NameList &Nodal_Var_Names() const { return nodal_vars; }
  const NameList &Element_Var_Names() const { return elmt_vars; }
  const NameList &Element_Att_Names() const { return elmt_atts; }
  const NameList &NS_Var_Names() const { return ns_vars; }
  const NameList &SS_Var_Names() const { return ss_vars; }
  const NameList &EB_Var_Names() const { return eb_vars; }
  const NameList &FB_Var_Names() const { return fb_vars; }

  const std::string &Global_Var_Name(size_t index) const;
  const std::string &Nodal_Var_Name(size_t index) const;
  const std::string &Element_Var_Name(size_t index) const;
  const std::string &Element_Att_Name(size_t index) const;
  const std::string &NS_Var_Name(size_t index) const;
  const std::string &SS_Var_Name(size_t index) const;
  const std::string &EB_Var_Name(size_t index) const;
  const std::string &FB_Var_Name(size_t index) const;

  // Element blocks:
  size_t Num_Element_Blocks() const { return num_elmt_blocks; }

  std::string Load_Element_Block_Description(size_t block_index) const;
  std::string Load_Element_Block_Descriptions() const;      // Loads all blocks.
  std::string Free_Element_Block(size_t block_index) const; // Frees all dynamic memory.
  std::string Free_Element_Blocks() const;                  // Frees all blocks.

  // Number maps:
  std::string Load_Node_Map();
  std::string Free_Node_Map();
  const INT  *Get_Node_Map() { return node_map; }
  std::string Load_Element_Map();
  std::string Free_Element_Map();
  const INT  *Get_Element_Map() { return elmt_map; }
  inline INT  Node_Map(size_t node_num) const;    // numbers are global, 1-offset
  inline INT  Element_Map(size_t elmt_num) const; // numbers are global, 1-offset

  // Nodal data:

  std::string   Load_Nodal_Coordinates();
  const double *X_Coords() const { return nodes; }
  const double *Y_Coords() const
  {
    if (dimension < 2) {
      return nullptr;
    }
    return nodes == nullptr ? nullptr : nodes + num_nodes;
  }
  const double *Z_Coords() const
  {
    if (dimension < 3) {
      return nullptr;
    }
    return nodes == nullptr ? nullptr : nodes + 2 * num_nodes;
  }
  void Free_Nodal_Coordinates();

  // (First time step = 1.)
  std::string   Load_Nodal_Results(int time_step_num, size_t var_index);
  const double *Get_Nodal_Results(size_t var_index) const;
  const double *Get_Nodal_Results(int t1, int t2, double proportion,
                                  size_t var_index) const; // Interpolated results
  void          Free_Nodal_Results();
  void          Free_Nodal_Results(size_t var_index);

  // Global data:  (NOTE:  Global and Nodal data are always stored at the same
  //                       time step.  Therefore, if current time step number
  //                       is changed, the results will all be deleted.)

  std::string   Load_Global_Results(int time_step_num);
  std::string   Load_Global_Results(int t1, int t2, double proportion); // Interpolated results
  const double *Get_Global_Results() const { return global_vals; }

  // Node/Side sets:

  Exo_Entity *Get_Entity_by_Index(EXOTYPE type, size_t block_index) const;
  Exo_Entity *Get_Entity_by_Id(EXOTYPE type, size_t id) const;
  Exo_Entity *Get_Entity_by_Name(EXOTYPE type, const std::string &name) const;

  Assembly<INT> *Get_Assembly_by_Id(size_t id) const;
  Assembly<INT> *Get_Assembly_by_Index(size_t assembly_index) const;
  Assembly<INT> *Get_Assembly_by_Name(const std::string &name) const;

  size_t          Block_Id(size_t block_index) const; // Returns associated block id.
  Exo_Block<INT> *Get_Element_Block_by_Id(size_t id) const;
  Exo_Block<INT> *Get_Element_Block_by_Index(size_t block_index) const;
  Exo_Block<INT> *Get_Element_Block_by_Name(const std::string &name) const;

  Side_Set<INT> *Get_Side_Set_by_Id(size_t set_id) const;
  Side_Set<INT> *Get_Side_Set_by_Index(size_t side_set_index) const;
  Side_Set<INT> *Get_Side_Set_by_Name(const std::string &name) const;

  Node_Set<INT> *Get_Node_Set_by_Id(size_t set_id) const;
  Node_Set<INT> *Get_Node_Set_by_Index(size_t set_index) const;
  Node_Set<INT> *Get_Node_Set_by_Name(const std::string &name) const;

  Edge_Block<INT> *Get_Edge_Block_by_Id(size_t block_id) const;
  Edge_Block<INT> *Get_Edge_Block_by_Index(size_t set_index) const;
  Edge_Block<INT> *Get_Edge_Block_by_Name(const std::string &name) const;

  Face_Block<INT> *Get_Face_Block_by_Id(size_t block_id) const;
  Face_Block<INT> *Get_Face_Block_by_Index(size_t set_index) const;
  Face_Block<INT> *Get_Face_Block_by_Name(const std::string &name) const;

  // Misc functions:

  int                    Check_State() const;                // Checks state of obj (not the file).
  int                    File_ID() const { return file_id; } // This is temporary.
  std::pair<int, size_t> Global_to_Block_Local(size_t global_elmt_num) const;

private:
  std::string file_name{};
  int         file_id{-1}; // Exodus file id; also used to determine if file is open.

  // GENESIS info:

  std::string title{};
  NameList    coord_names{};
  size_t      num_nodes{0};
  int         dimension{0};
  size_t      num_elmts{0};
  size_t      num_faces{0};
  size_t      num_edges{0};
  size_t      num_elmt_blocks{0};
  size_t      num_node_sets{0};
  size_t      num_side_sets{0};
  size_t      num_edge_blocks{0};
  size_t      num_face_blocks{0};
  size_t      num_assemblies{0};
  int         num_change_sets{0};
  int         current_change_set_index{-2};
  int         io_word_size{0}; // Note: The "compute word size" is always 8.

  Exo_Block<INT>  *eblocks{nullptr};     // Array.
  Node_Set<INT>   *nsets{nullptr};       // Array.
  Side_Set<INT>   *ssets{nullptr};       // Array.
  Edge_Block<INT> *edge_blocks{nullptr}; // Array.
  Face_Block<INT> *face_blocks{nullptr}; // Array.
  Assembly<INT>   *assemblies{nullptr};

  double *nodes{nullptr}; // Matrix;  dimension by num_nodes (row major form).
                          //          I.e., all x's then all y's, etc.

  INT *node_map{nullptr}; // Array; num_nodes long when filled.
  INT *elmt_map{nullptr}; // Array; num_elmts long when filled.

  // RESULTS info:

  NameList global_vars{};
  NameList nodal_vars{};
  NameList elmt_vars{};
  NameList elmt_atts{};
  NameList ns_vars{};
  NameList ss_vars{};
  NameList eb_vars{};
  NameList fb_vars{};

  std::vector<int> change_set_ids{};
  NameList         change_set_names{};

  double time_scale{1.0};
  double time_offset{0.0};

  std::vector<double> times{};

  int      cur_time{0};          // Current timestep number of the results (0 means none).
  double **results{nullptr};     // Array of pointers (to arrays of results data);
                                 // length is number of nodal variables.
  double *global_vals{nullptr};  // Array of global variables for the current timestep.
  double *global_vals2{nullptr}; // Array of global variables used if interpolating.

  // Internal methods:
  void Reset_Meta_Data();
};

template <typename INT> inline INT Exo_Read<INT>::Node_Map(size_t node_num) const
{
  SMART_ASSERT(Check_State());
  SMART_ASSERT(node_num <= num_nodes);

  if (node_map) {
    return node_map[node_num - 1];
  }
  return 0;
}

template <typename INT> inline INT Exo_Read<INT>::Element_Map(size_t elmt_num) const
{
  SMART_ASSERT(Check_State());
  SMART_ASSERT(elmt_num <= num_elmts);

  if (elmt_map) {
    return elmt_map[elmt_num - 1];
  }
  return 0;
}
