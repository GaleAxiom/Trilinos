// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER


/** \file test_orientation_PYR_TriFace_newBasis.hpp
    \brief  Test for checking orientation tools for the Hierarchical basis on Pyramids

    The test considers two pyramids in physical space sharing a common triangular face.
    In order to test significant configurations, we consider 4 mappings of the reference pyramid
    to the first (physical) pyramid, so that the common face is mapped from all 4 triangular faces
    of the reference pyramid.
    Then, for each of the mappings, the global ids of the vertices of the common face are permuted.
    This gives a total of 24 combinations.

    The test considers HGRAD, HCURL and HDIV Hierarchical basis functions, and for each:
    1. Computes the oriented basis.
    2. Computes the basis coefficients, separately for each pyramid, for functions belonging to the H-space spanned by the basis.
    3. Checks that the dofs shared between the two pyramids are equivalent (this ensures that the orientation works correctly)
    4. Checks that the functions are indeed exactly reproduced

    Note that there is some subtlety with these tests, stemming largely from the fact that the pyramidal basis
    consists of rational functions which are only guaranteed to be polynomial on the space-appropriate (scalar/tangent/normal)
    restrictions of the basis to the faces (the minimal requirement for compatibility with elements of other topologies that might
    share the face).  When setting up the H(div) and H(curl) systems in particular, we restrict the test points to the shared face,
    and work only with the normal or tangent part of the polynomial function that we are seeking to reproduce with the basis.
    We similarly restrict the basis functions to those associated with the shared face, since the other functions will have
    vanishing normals or tangents there.
  
    \author Created by Nate Roberts, based on HEX tests by Mauro Perego.
 */

#include "Intrepid2_config.h"

#ifdef HAVE_INTREPID2_DEBUG
#define INTREPID2_TEST_FOR_DEBUG_ABORT_OVERRIDE_TO_CONTINUE
#endif

#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"
#include "Intrepid2_PointTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_TestUtils.hpp"

#include "Intrepid2_HGRAD_PYR_C1_FEM.hpp"
#include "Intrepid2_HierarchicalBasisFamily.hpp"

#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_RCP.hpp"
#include <array>
#include <set>
#include <random>
#include <algorithm>

namespace Intrepid2 {

namespace Test {

#define INTREPID2_TEST_ERROR_EXPECTED( S )              \
    try {                                                               \
      ++nthrow;                                                         \
      S ;                                                               \
    }                                                                   \
    catch (std::exception &err) {                                        \
      ++ncatch;                                                         \
      *outStream << "Expected Error ----------------------------------------------------------------\n"; \
      *outStream << err.what() << '\n';                                 \
      *outStream << "-------------------------------------------------------------------------------" << "\n\n"; \
    }

template<typename ValueType, typename DeviceType>
int OrientationPyrTriFaceNewBasis(const bool verbose) {

  typedef Kokkos::DynRankView<ValueType,DeviceType> DynRankView;
  typedef Kokkos::DynRankView<ordinal_type,DeviceType> DynRankViewInt;
#define ConstructWithLabel(obj, ...) obj(#obj, __VA_ARGS__)

  static Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing

  if (verbose)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs,       false);

  Teuchos::oblackholestream oldFormatState;
  oldFormatState.copyfmt(std::cout);

  int errorFlag = 0;
  const ValueType tol = tolerence();

  struct Fun {
    ValueType
    KOKKOS_INLINE_FUNCTION
    operator()(const ValueType& x, const ValueType& y, const ValueType& z) {
      return (x+1)*(y-2);//*(z+3)*(x + 2*y +5*z+ 1.0/3.0);
    }
  };

  struct FunDivTriFace { // commmon tri face is y-orthogonal
    ValueType
    KOKKOS_INLINE_FUNCTION
    operator()(const ValueType& x, const ValueType& y, const ValueType& z, const int comp=0) {
      ValueType a = 2*x*z+x;
      ValueType f0 = 5+z+x*x;
      ValueType f1 = x+z*z;
      //fun = f + a x
      switch (comp) {
      case 0:
        return 0;
      case 1:
        return f0 + a*x + f1 + a*z;
      case 2:
        return 0;
      default:
        return 0;
      }
    }
  };

  struct FunCurl {
    ValueType
    KOKKOS_INLINE_FUNCTION
    operator()(const ValueType& x, const ValueType& y, const ValueType& z, const int comp=0) {
      ValueType a0 = y-7+z*z;
      ValueType a1 = 2*z-1+z*x;
      ValueType a2 = z-2+x*x;
      ValueType f0 = 2+x+z+x*y;
      ValueType f1 = 3-3*z;
      ValueType f2 = -5+x;
      //fun = f + a \times x
      switch (comp) {
      case 0:
        return f0 + (a1*z-a2*y);//2*x+y-z + (x+2*(y+z);
      case 1:
        return f1 + (a2*x-a0*z);//y+2*(z+x);
      case 2:
        return f2 + (a0*y-a1*x);//z+2*(x+y);
      default:
        return 0;
      }
    }
  };


  class BasisFunctionsSystem{
  public:
    BasisFunctionsSystem(const ordinal_type basisCardinality_, const ordinal_type numRefCoords_, const ordinal_type  dim_,
                         std::set<ordinal_type> &dofsToExclude0_, std::set<ordinal_type> &dofsToExclude1_) :
      fullBasisCardinality(basisCardinality_),
      numRefCoords(numRefCoords_),
      dim(dim_),
      dofsToExclude0(dofsToExclude0_),
      dofsToExclude1(dofsToExclude1_),
      work("lapack_work", fullBasisCardinality-std::min(dofsToExclude0.size(),dofsToExclude1.size())+dim*numRefCoords, 1) ,
      cellMassMat("basisMat", dim*numRefCoords,fullBasisCardinality-std::min(dofsToExclude0.size(),dofsToExclude1.size())),
      cellRhsMat("rhsMat",dim*numRefCoords, 1) {
        
      ordinal_type includedDofCount = fullBasisCardinality-std::min(dofsToExclude0.size(),dofsToExclude1.size());
      if (numRefCoords < includedDofCount)
      {
        std::cout << "point count is " << numRefCoords << std::endl;
        std::cout << "fullBasisCardinality is " << fullBasisCardinality << std::endl;
        std::cout << "dofsToExclude0.size() is " << dofsToExclude0.size() << std::endl;
        std::cout << "dofsToExclude1.size() is " << dofsToExclude1.size() << std::endl;
      }
      INTREPID2_TEST_FOR_EXCEPTION(numRefCoords < includedDofCount, std::invalid_argument, "numRefCoords must be at least as large as boundaryBasisCardinality");
    };

    std::vector<int> computeBasisCoeffs(DynRankView basisCoeffs, ordinal_type& errorFlag, const DynRankView basisValues, const DynRankView funAtPhysRefCoords) {
      const ordinal_type numCells = 2;
      INTREPID2_TEST_FOR_EXCEPTION(basisCoeffs.extent_int(0) != numCells, std::invalid_argument, "cell dimension must be 2");
      std::vector<int> info(numCells);
      for(ordinal_type ic=0; ic<numCells; ++ic) {
        auto dofsToExclude = (ic==0) ? dofsToExclude0 : dofsToExclude1;
        for(ordinal_type i=0; i<numRefCoords; ++i) {
          int j = 0;
          for(ordinal_type jFull=0; jFull<fullBasisCardinality; ++jFull) {
            if (dofsToExclude.find(jFull) == dofsToExclude.end())
            {
              if (dim==1) {
                cellMassMat(i,j) = basisValues(ic,jFull,i);
              }
              else
              {
                for (ordinal_type id=0; id<dim; ++id) {
                  cellMassMat(i*dim+id,j) = basisValues(ic,jFull,i,id);
                }
              }
              j++;
            }
          }
          if (dim==1) 
          {
            cellRhsMat(i,0) = funAtPhysRefCoords(ic,i);
          }
          else
          {
            for (ordinal_type id=0; id<dim; ++id) {
              cellRhsMat(i*dim+id,0) = funAtPhysRefCoords(ic,i,id);
            }
          }
        }
        lapack.GELS('N', dim*numRefCoords, fullBasisCardinality-dofsToExclude.size(),1,
            cellMassMat.data(),
            cellMassMat.stride_1(),
            cellRhsMat.data(),
            cellRhsMat.stride_1(),
            work.data(),
            fullBasisCardinality-dofsToExclude.size()+dim*numRefCoords,
            &info[ic]);

        int i = 0;
        for(ordinal_type iFull=0; iFull<fullBasisCardinality; ++iFull) {
          if (dofsToExclude.find(iFull) == dofsToExclude.end())
          {
            basisCoeffs(ic,iFull) = cellRhsMat(i,0);
            i++;
          }
          else
          {
            basisCoeffs(ic,iFull) = 0.0;
          }
        }
      }
      return info;
    }
  private:
    Teuchos::LAPACK<ordinal_type,ValueType> lapack;
    ordinal_type fullBasisCardinality, numRefCoords, dim;
    std::set<ordinal_type> dofsToExclude0, dofsToExclude1;
    Kokkos::View<ValueType**,Kokkos::LayoutLeft,Kokkos::HostSpace> work;
    Kokkos::View<ValueType**,Kokkos::LayoutLeft,Kokkos::HostSpace> cellMassMat;
    Kokkos::View<ValueType**,Kokkos::LayoutLeft,Kokkos::HostSpace> cellRhsMat;
  };


  typedef std::array<ordinal_type,2> edgeType;
  typedef std::array<ordinal_type,3> faceType;
  typedef CellTools<DeviceType> ct;
  typedef OrientationTools<DeviceType> ots;
  typedef RealSpaceTools<DeviceType> rst;
  typedef FunctionSpaceTools<DeviceType> fst;

  using  basisType = Basis<DeviceType,ValueType,ValueType>;

  constexpr ordinal_type dim = 3;
  constexpr ordinal_type numCells = 2;
  constexpr ordinal_type numElemVertices = 5;
  constexpr ordinal_type numTotalVertices = 7;
  constexpr ordinal_type numSharedVertices = 3;
  constexpr ordinal_type numSharedEdges = 3;

  ValueType  vertices_orig[numTotalVertices][dim] = {{1,1,0},{0,1,0},{0,0,0},{1,0,0},{0,0,1},{0,-1,0},{1,-1,0}};
  ordinal_type pyrs_orig[numCells][numElemVertices] = {{2,3,0,1,4},{5,6,3,2,4}};  //common face is {2,3,4}
  faceType common_face = {{2,3,4}};
  ordinal_type pyrs_rotated[numCells][numElemVertices];

  static std::set<edgeType> common_edges { {{2,3}}, {{2,4}}, {{3,4}} };
  
  static ordinal_type shared_vertices[numCells][numSharedVertices];
  static ordinal_type edgeIndices[numCells][numSharedEdges];
  static ordinal_type faceIndex[numCells];

  class TestResults
  {
  private:
    const DynRankView basisCoeffs, transformedBasisValuesAtRefCoords, transformedBasisValuesAtRefCoordsOriented, interfaceBasisValues, funAtPhysRefCoords;
    const Kokkos::DynRankView<Orientation,DeviceType> elemOrts;
    const basisType* basis;


  public:

    TestResults(const DynRankView basisCoeffs_,
                const DynRankView transformedBasisValuesAtRefCoords_,
                const DynRankView transformedBasisValuesAtRefCoordsOriented_,
                const DynRankView interfaceBasisValues_,
                const DynRankView funAtPhysRefCoords_,
                const Kokkos::DynRankView<Orientation,DeviceType> elemOrts_,
                const basisType* basis_) :
          basisCoeffs(basisCoeffs_),
          transformedBasisValuesAtRefCoords(transformedBasisValuesAtRefCoords_),
          transformedBasisValuesAtRefCoordsOriented(transformedBasisValuesAtRefCoordsOriented_),
          interfaceBasisValues(interfaceBasisValues_),
          funAtPhysRefCoords(funAtPhysRefCoords_),
          elemOrts(elemOrts_),
          basis(basis_){}

    //check that fun values are consistent at the common vertices
    void test(ordinal_type& errorFlag, ValueType tol){

      auto numVertexDOFs = basis->getDofCount(0,0);
      if(numVertexDOFs >0) {
        bool areDifferent(false);


        for(ordinal_type j=0;j<numSharedVertices && !areDifferent;j++)
          areDifferent = std::abs(basisCoeffs(0,basis->getDofOrdinal(0,shared_vertices[0][j],0))
              - basisCoeffs(1,basis->getDofOrdinal(0,shared_vertices[1][j],0))) > 10*tol;

        if(areDifferent) {
          errorFlag++;
          *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
          *outStream << "Function  DOFs on shared vertices computed using Cell 0 basis functions are not consistent with those computed using Cell 1 basis functions\n";
          *outStream << "Function DOFs for Cell 0 are:";
          for(ordinal_type j=0;j<numSharedVertices;j++)
            *outStream << " " << basisCoeffs(0,basis->getDofOrdinal(0,shared_vertices[0][j],0));
          *outStream << "\nFunction DOFs for Cell 1 are:";
          for(ordinal_type j=0;j<numSharedVertices;j++)
            *outStream << " " << basisCoeffs(1,basis->getDofOrdinal(0,shared_vertices[1][j],0));
          *outStream << std::endl;
        }

      }


      //check that fun values are consistent on shared edges dofs
      auto numEdgeDOFs = basis->getDofCount(1,0);
      if(numEdgeDOFs>0)
      {

        bool areDifferent(false);
        for(std::size_t iEdge=0;iEdge<numSharedEdges;iEdge++) {
        for(ordinal_type j=0;j<numEdgeDOFs && !areDifferent;j++) {
          areDifferent = std::abs(basisCoeffs(0,basis->getDofOrdinal(1,edgeIndices[0][iEdge],j))
              - basisCoeffs(1,basis->getDofOrdinal(1,edgeIndices[1][iEdge],j))) > 10*tol;
        }
        if(areDifferent) {
          errorFlag++;
          *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
          *outStream << "Function DOFs on shared edge " << iEdge << " computed using Cell 0 basis functions are not consistent with those computed using Cell 1 basis functions\n";
          *outStream << "Function DOFs for Cell 0 are:";
          for(ordinal_type j=0;j<numEdgeDOFs;j++)
            *outStream << " " << basisCoeffs(0,basis->getDofOrdinal(1,edgeIndices[0][iEdge],j));
          *outStream << "\nFunction DOFs for Cell 1 are:";
          for(ordinal_type j=0;j<numEdgeDOFs;j++)
            *outStream << " " << basisCoeffs(1,basis->getDofOrdinal(1,edgeIndices[1][iEdge],j));
          *outStream << std::endl;
        }
        }
      }


      //check that fun values are consistent on common face dofs
      auto numFaceDOFs = basis->getDofCount(2,0);
      if(numFaceDOFs > 0 && dim>2)
      {
        bool areDifferent(false);
        for(ordinal_type j=0;j<numFaceDOFs && !areDifferent;j++) {
          areDifferent = std::abs(basisCoeffs(0,basis->getDofOrdinal(2,faceIndex[0],j))
              - basisCoeffs(1,basis->getDofOrdinal(2,faceIndex[1],j))) > 100*tol;
        }

        if(areDifferent) {
          errorFlag++;
          *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
          *outStream << "Function DOFs on common face computed using Pyr 0 basis functions are not consistent with those computed using Pyr 1\n";
          *outStream << "Function DOFs for Pyr 0 are:";
          for(ordinal_type j=0;j<numFaceDOFs;j++)
            *outStream << " " << basisCoeffs(0,basis->getDofOrdinal(2,faceIndex[0],j));
          *outStream << "\nFunction DOFs for Pyr 1 are:";
          for(ordinal_type j=0;j<numFaceDOFs;j++)
            *outStream << " " << basisCoeffs(1,basis->getDofOrdinal(2,faceIndex[1],j));
          
          ordinal_type basis_dim = (interfaceBasisValues.rank()==3) ? 1 : dim;
          const auto numPoints = interfaceBasisValues.extent_int(2);
          *outStream << "\nOriented basis interface value for (point, face dof, cell):\n";
          for (int p=0; p<numPoints; p++)
          {
            for(ordinal_type j=0;j<numFaceDOFs;j++)
            {
              for (int ic=0; ic<=1; ic++)
              {
                *outStream << "(" << p << "," << j << "," << ic << "): ";
                if(basis_dim==1)
                  *outStream << " (" << interfaceBasisValues(ic,basis->getDofOrdinal(2,faceIndex[ic],j),p);
                else
                  *outStream << " (" << interfaceBasisValues(ic,basis->getDofOrdinal(2,faceIndex[ic],j),p,0);
                for(ordinal_type d=1; d<basis_dim; d++)
                  *outStream <<  ", " << interfaceBasisValues(ic,basis->getDofOrdinal(2,faceIndex[ic],j),p,d);
                *outStream  << ")\n";
              }
            }
          }
          *outStream << "\nFunction DOFs for Pyr 1 are:";
          for(ordinal_type j=0;j<numFaceDOFs;j++)
            *outStream << " " << basisCoeffs(1,basis->getDofOrdinal(2,faceIndex[1],j));
          
          *outStream << std::endl;
        }
      }

      ordinal_type numRefCoords = funAtPhysRefCoords.extent(1);
      ordinal_type fullBasisCardinality = basisCoeffs.extent(1);

      // we only use the shared points (here, on the common triangle) for coefficient setting.
      const ordinal_type triDofCount = basis->getDofCount(2, 0); // face 0 is a triangle (face 4 is the quad)
      
      INTREPID2_TEST_FOR_EXCEPTION(numRefCoords < triDofCount, std::invalid_argument, "numRefCoords must be at least as large as triDofCount");
      ordinal_type basis_dim = (interfaceBasisValues.rank()==3) ? 1 : dim;

      DynRankView ConstructWithLabel(funAtRefCoordsOriented, numCells, numRefCoords, basis_dim);
      for(ordinal_type ic=0; ic<numCells; ++ic) {
        ValueType error=0;
        for(ordinal_type j=0; j<numRefCoords; ++j) {
          if (basis_dim==1) {
            for(ordinal_type k=0; k<fullBasisCardinality; ++k)
              funAtRefCoordsOriented(ic,j,0) += basisCoeffs(ic,k)*interfaceBasisValues(ic,k,j);
            error = std::max(std::abs( funAtPhysRefCoords(ic,j) - funAtRefCoordsOriented(ic,j,0)), error);
          } else {
            for(ordinal_type d=0; d<basis_dim; ++d) {
              for(ordinal_type k=0; k<fullBasisCardinality; ++k)
                funAtRefCoordsOriented(ic,j,d) += basisCoeffs(ic,k)*interfaceBasisValues(ic,k,j,d);
              error = std::max(std::abs( funAtPhysRefCoords(ic,j,d) - funAtRefCoordsOriented(ic,j,d)), error);
            }
          }
        }

        if(error>100*tol) {
          errorFlag++;
          *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
          *outStream << "Function values at reference points differ from those computed using basis functions on Cell " << ic << "\n";
          *outStream << "Function values at reference points are:\n";
          for(ordinal_type j=0; j<numRefCoords; ++j) {
            if(basis_dim==1)
              *outStream << " (" << funAtPhysRefCoords(ic,j);
            else
              *outStream << " (" << funAtPhysRefCoords(ic,j,0);
            for(ordinal_type d=1; d<basis_dim; d++)
              *outStream <<  ", " << funAtPhysRefCoords(ic,j,d);
            *outStream  << ")";
          }
          *outStream << "\nFunction values at reference points computed using basis functions are\n";
          for(ordinal_type j=0; j<numRefCoords; ++j) {
            *outStream << " (" << funAtRefCoordsOriented(ic,j,0);
            for(ordinal_type d=1; d<basis_dim; d++)
              *outStream <<  ", " << funAtRefCoordsOriented(ic,j,d);
            *outStream  << ")";
          }
          *outStream << std::endl;
        }
      }
      
      // check that global (oriented) basis coefficients contracted with the transformed oriented basis values agrees with the transpose-oriented basis coefficients contracted with the unoriented transformed basis values
      DynRankView ConstructWithLabel(localBasisCoeffs, numCells, fullBasisCardinality);
      OrientationTools<DeviceType>::modifyBasisByOrientationTranspose(localBasisCoeffs,
                                                                      basisCoeffs,
                                                                      elemOrts,
                                                                      basis);
      
      DynRankView ConstructWithLabel(basisCoeffsModified, numCells, fullBasisCardinality);
      OrientationTools<DeviceType>::modifyBasisByOrientation(basisCoeffsModified,
                                                             basisCoeffs,
                                                             elemOrts,
                                                             basis);
      
      
      for(ordinal_type ic=0; ic<numCells; ++ic) {
        ValueType error=0;
        for(ordinal_type j=0; j<numRefCoords; ++j) {
          for (int d=0; d<transformedBasisValuesAtRefCoords.extent_int(3); d++)
          {
            ValueType globalValue_d = 0; // oriented/global
            ValueType localValue_d  = 0; // unoriented/local
            for(ordinal_type k=0; k<fullBasisCardinality; ++k)
            {
              auto orientedBasisValue = transformedBasisValuesAtRefCoordsOriented(ic,k,j,d);
              auto basisValue         = transformedBasisValuesAtRefCoords        (ic,k,j,d);
              
              globalValue_d += basisCoeffs(ic,k)      * orientedBasisValue;
              localValue_d  += localBasisCoeffs(ic,k) * basisValue;
            }
            error = std::max(std::abs( globalValue_d - localValue_d), error);
          }
        }

        if(error>100*tol) {
          errorFlag++;
          *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
          *outStream << "global values do not agree with local on cell " << ic << "\n";
        }
      }
    }
  };


  try {

    const ordinal_type order = 3;
    ordinal_type reorder[numTotalVertices] = {0,1,2,3,4,5,6};

    do {
      ordinal_type orderback[numTotalVertices];
      for(ordinal_type i=0;i<numTotalVertices;++i) {
        orderback[reorder[i]]=i;
      }
      ValueType vertices[numTotalVertices][dim];
      ordinal_type pyrs[numCells][numElemVertices];
      std::copy(&pyrs_orig[0][0], &pyrs_orig[0][0]+numCells*numElemVertices, &pyrs_rotated[0][0]);
      for (ordinal_type rotationOrdinal=0; rotationOrdinal<4; ++rotationOrdinal) {
        // rotations of the pyramid correspond to rotations of the bottom quadrilateral
        // for now, we do not consider reflection symmetries
        for (int vertexOrdinal=0; vertexOrdinal<4; vertexOrdinal++)
        {
          pyrs_rotated[0][vertexOrdinal] = pyrs_orig[0][(vertexOrdinal + rotationOrdinal) % 4];
        }

        for(ordinal_type i=0; i<numCells;++i)
          for(ordinal_type j=0; j<numElemVertices;++j)
            pyrs[i][j] = reorder[pyrs_rotated[i][j]];

        for(ordinal_type i=0; i<numTotalVertices;++i)
          for(ordinal_type d=0; d<dim;++d)
            vertices[i][d] = vertices_orig[orderback[i]][d];

        *outStream <<  "Considering Pyr 0: [ ";
        for(ordinal_type j=0; j<numElemVertices;++j)
          *outStream << pyrs[0][j] << " ";
        *outStream << "] and Pyr 1: [ ";
        for(ordinal_type j=0; j<numElemVertices;++j)
          *outStream << pyrs[1][j] << " ";
        *outStream << "]\n";

        shards::CellTopology pyramid(shards::getCellTopologyData<shards::Pyramid<5> >());

        //computing vertices coords
        DynRankView ConstructWithLabel(physVertices, numCells, pyramid.getNodeCount(), dim);
        for(ordinal_type i=0; i<numCells; ++i)
          for(std::size_t j=0; j<pyramid.getNodeCount(); ++j)
            for(ordinal_type k=0; k<dim; ++k)
              physVertices(i,j,k) = vertices[pyrs[i][j]][k];



        //computing common face and edges
        {
          for (int i=0; i<numCells; ++i)
          {
            faceIndex[i] = -1;
          }
          
          faceType face={};
          edgeType edge={};
          for(ordinal_type i=0; i<numCells; ++i) {
            for (std::size_t iv=0; iv<pyramid.getNodeCount(); ++iv) {
              auto vertex = pyrs_rotated[i][pyramid.getNodeMap(0,iv,0)];
              for (std::size_t isv=0; isv<common_face.size(); ++isv)
                if(common_face[isv] == vertex)
                  shared_vertices[i][isv] = iv;
            }
            for (std::size_t is=0; is<pyramid.getSideCount(); ++is) {
              const std::size_t nodeCount = pyramid.getNodeCount(2,is);
              for (std::size_t k=0; k<nodeCount; ++k)
              {
                auto pyrNode = pyramid.getNodeMap(2,is,k);
                face[k]= pyrs_rotated[i][pyrNode];
              }

              //rotate and flip
              auto minElPtr= std::min_element(face.begin(), face.end());
              std::rotate(face.begin(),minElPtr,face.end());
              if(face[2]<face[1]) {auto tmp=face[1]; face[1]=face[2]; face[2]=tmp;}

              if(face == common_face) faceIndex[i]=is;
            }
            for (std::size_t ie=0; ie<pyramid.getEdgeCount(); ++ie) {
              for (std::size_t k=0; k<pyramid.getNodeCount(1,ie); ++k)
                edge[k]= pyrs_rotated[i][pyramid.getNodeMap(1,ie,k)];
              std::sort(edge.begin(),edge.end());
              auto it=common_edges.find(edge);
              if(it !=common_edges.end()){
                auto edge_lid = std::distance(common_edges.begin(),it);
                edgeIndices[i][edge_lid]=ie;
              }
            }
            if (faceIndex[i] == -1)
            {
              INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "common face not found for cell");
            }
          }
        }

        using CG_HBasis = HierarchicalBasisFamily<DeviceType,ValueType,ValueType>;
        std::vector<basisType*> basis_set;

        //HGRAD BASIS
        {
          //compute reference points: use a lattice on the quad base of the pyramid
          auto refPoints = getInputPointsView<double,DeviceType>(pyramid, order*3);
          ordinal_type numRefCoords = refPoints.extent_int(0);
          
          // compute orientations for cells (one time computation)
          DynRankViewInt elemNodes(&pyrs[0][0], numCells, numElemVertices);
          Kokkos::DynRankView<Orientation,DeviceType> elemOrts("elemOrts", numCells);
          ots::getOrientation(elemOrts, elemNodes, pyramid);

          //Compute physical Dof Coordinates and Reference coordinates
          DynRankView ConstructWithLabel(physRefCoords, numCells, numRefCoords, dim);
          {
            Basis_HGRAD_PYR_C1_FEM<DeviceType,ValueType,ValueType> pyramidLinearBasis; //used for computing physical coordinates
            DynRankView ConstructWithLabel(pyramidLinearBasisValuesAtRefCoords, pyramid.getNodeCount(), numRefCoords);
            pyramidLinearBasis.getValues(pyramidLinearBasisValuesAtRefCoords, refPoints);
            for(ordinal_type i=0; i<numCells; ++i)
              for(ordinal_type d=0; d<dim; ++d) {
                for(ordinal_type j=0; j<numRefCoords; ++j)
                  for(std::size_t k=0; k<pyramid.getNodeCount(); ++k)
                    physRefCoords(i,j,d) += vertices[pyrs[i][k]][d]*pyramidLinearBasisValuesAtRefCoords(k,j);
              }
          }
          
          
          Fun fun;
          DynRankView ConstructWithLabel(funAtPhysRefCoords, numCells, numRefCoords);
          for(ordinal_type i=0; i<numCells; ++i) {
            for(ordinal_type j=0; j<numRefCoords; ++j)
              funAtPhysRefCoords(i,j) = fun(physRefCoords(i,j,0), physRefCoords(i,j,1), physRefCoords(i,j,2));
          }

          basis_set.push_back(new typename  CG_HBasis::HGRAD_PYR(order));

          for (auto basisPtr:basis_set) {
            auto& basis = *basisPtr;
            auto name = basis.getName();
            *outStream << " " << name << std::endl;
            ordinal_type basisCardinality = basis.getCardinality();

            //check that fun values at reference points coincide with those computed using basis functions
            DynRankView ConstructWithLabel(basisValuesAtRefCoordsOriented, numCells, basisCardinality, numRefCoords);
            DynRankView ConstructWithLabel(transformedBasisValuesAtRefCoordsOriented, numCells, basisCardinality, numRefCoords);
            DynRankView ConstructWithLabel(transformedBasisValuesAtRefCoords, numCells, basisCardinality, numRefCoords);

            DynRankView ConstructWithLabel(basisValuesAtRefCoords, basisCardinality, numRefCoords);
            basis.getValues(basisValuesAtRefCoords, refPoints);

            // modify basis values to account for orientations
            ots::modifyBasisByOrientation(basisValuesAtRefCoordsOriented,
                basisValuesAtRefCoords,
                elemOrts,
                &basis);
            
            // transform basis values
            fst::HGRADtransformVALUE(transformedBasisValuesAtRefCoords, basisValuesAtRefCoords);
            deep_copy(transformedBasisValuesAtRefCoordsOriented,
                basisValuesAtRefCoordsOriented);

            DynRankView ConstructWithLabel(basisCoeffs, numCells, basisCardinality);

            // on the interior, pyramid basis is not polynomial.  So we exclude those dofs when we do the coefficient setting.
            const ordinal_type volumeDofCount = basis.getDofCount(dim, 0);
            std::set<ordinal_type> volumeDofs;
            for (ordinal_type i=0; i<volumeDofCount; i++)
            {
              ordinal_type dofOrdinal = basis.getDofOrdinal(dim, 0, i);
              volumeDofs.insert(dofOrdinal);
            }
            
            BasisFunctionsSystem  basisFunctionsSystem(basisCardinality, numRefCoords, 1, volumeDofs, volumeDofs);
            auto info = basisFunctionsSystem.computeBasisCoeffs(basisCoeffs, errorFlag, transformedBasisValuesAtRefCoordsOriented, funAtPhysRefCoords);

            for (int ic =0; ic < numCells; ++ic) {
              if(info[ic] != 0) {
                errorFlag++;
                *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
                *outStream << "LAPACK error flag for cell " << ic << " is: " << info[ic] << std::endl;
              }
            }

            TestResults testResults(basisCoeffs, transformedBasisValuesAtRefCoords, transformedBasisValuesAtRefCoordsOriented, transformedBasisValuesAtRefCoordsOriented,
                                    funAtPhysRefCoords, elemOrts, basisPtr);
            testResults.test(errorFlag,tol);
            delete basisPtr;
          }
        }


        //HCURL Case
        //TODO: uncomment after we implement H(curl) basis for pyramids
        /*{
         // compute reference points on the shared face of the pyramid
         shards::CellTopology tri(shards::getCellTopologyData<shards::Triangle<3> >());
         auto refPointsTri = getInputPointsView<double,DeviceType>(tri, order);
         ordinal_type numRefCoords = refPointsTri.extent_int(0);
         
         // refPointsTri will include the apex of the pyramid, which is an excluded (singular) point.
         // to correct this, we move the points toward the center by a factor of s=0.1:
         
         const double s = 0.1;
         const double centroid_x = 1./3.;
         const double centroid_y = 1./3.;
         using ExecutionSpace = typename DynRankView::execution_space;
         Kokkos::RangePolicy < ExecutionSpace > policy(0,numRefCoords);
         Kokkos::parallel_for( policy,
         KOKKOS_LAMBDA (const ordinal_type &pointOrdinal )
         {
           const auto p_x = refPointsTri(pointOrdinal,0);
           const auto p_y = refPointsTri(pointOrdinal,1);
           refPointsTri(pointOrdinal,0) = (1.0 - s) * p_x + s * centroid_x;
           refPointsTri(pointOrdinal,1) = (1.0 - s) * p_y + s * centroid_y;
         });
         Kokkos::fence();
         
         DynRankView ConstructWithLabel(refPointsMultiCell, 2, numRefCoords, dim);
         auto refPoints0 = Kokkos::subview(refPointsMultiCell, 0, Kokkos::ALL(), Kokkos::ALL());
         auto refPoints1 = Kokkos::subview(refPointsMultiCell, 1, Kokkos::ALL(), Kokkos::ALL());
         
         // map the ref points to the pyramid
         const ordinal_type FACE_DIM  = 2;
         CellTools<DeviceType>::mapToReferenceSubcell(refPoints0, refPointsTri,
                                                      FACE_DIM, faceIndex[0],
                                                      pyramid);
         CellTools<DeviceType>::mapToReferenceSubcell(refPoints1, refPointsTri,
                                                      FACE_DIM, faceIndex[1],
                                                      pyramid);
         
         // compute orientations for cells (one time computation)
         DynRankViewInt elemNodes(&pyrs[0][0], numCells, numElemVertices);
         Kokkos::DynRankView<Orientation,DeviceType> elemOrts("elemOrts", numCells);
         ots::getOrientation(elemOrts, elemNodes, pyramid);

         //Compute physical Dof Coordinates and Reference coordinates
         DynRankView ConstructWithLabel(physRefCoords, numCells, numRefCoords, dim);
         {
           Basis_HGRAD_PYR_C1_FEM<DeviceType,ValueType,ValueType> pyramidLinearBasis; //used for computing physical coordinates
           DynRankView ConstructWithLabel(pyramidLinearBasisValuesAtRefCoords0, pyramid.getNodeCount(), numRefCoords);
           DynRankView ConstructWithLabel(pyramidLinearBasisValuesAtRefCoords1, pyramid.getNodeCount(), numRefCoords);
           pyramidLinearBasis.getValues(pyramidLinearBasisValuesAtRefCoords0, refPoints0);
           pyramidLinearBasis.getValues(pyramidLinearBasisValuesAtRefCoords1, refPoints1);
           std::vector<DynRankView> pyramidLinearBasisValuesAtRefCoords {pyramidLinearBasisValuesAtRefCoords0, pyramidLinearBasisValuesAtRefCoords1};
           for(ordinal_type i=0; i<numCells; ++i)
             for(ordinal_type d=0; d<dim; ++d) {
               for(ordinal_type j=0; j<numRefCoords; ++j)
                 for(std::size_t k=0; k<pyramid.getNodeCount(); ++k)
                   physRefCoords(i,j,d) += vertices[pyrs[i][k]][d]*pyramidLinearBasisValuesAtRefCoords[i](k,j);
             }
         }
         
         FunCurlTriFace fun;
         DynRankView ConstructWithLabel(funAtPhysRefCoords, numCells, numRefCoords, dim);
         for(ordinal_type i=0; i<numCells; ++i) {
           for(ordinal_type j=0; j<numRefCoords; ++j) {
             for(ordinal_type k=0; k<dim; ++k)
               funAtPhysRefCoords(i,j,k) = fun(physRefCoords(i,j,0), physRefCoords(i,j,1), physRefCoords(i,j,2), k);
           }
         }
         basis_set.clear();
         basis_set.push_back(new typename CG_HBasis::HCURL_PYR(order));

         basis_set.clear();
         basis_set.push_back(new typename  CG_HBasis::HCURL_PYR(order));

         for (auto basisPtr:basis_set) {
           auto& basis = *basisPtr;
           auto name = basis.getName();
           *outStream << " " << name << std::endl;

           ordinal_type basisCardinality = basis.getCardinality();
           DynRankView ConstructWithLabel(basisCoeffs, numCells, basisCardinality);

           //check that fun values at reference points coincide with those computed using basis functions
           DynRankView ConstructWithLabel(basisValuesAtRefCoordsOriented, numCells, basisCardinality, numRefCoords, dim);
           DynRankView ConstructWithLabel(transformedBasisValuesAtRefCoordsOriented, numCells, basisCardinality, numRefCoords, dim);
           DynRankView ConstructWithLabel(transformedBasisValuesAtRefCoords, numCells, basisCardinality, numRefCoords, dim);
           DynRankView basisValuesAtRefCoordsCells("inValues", numCells, basisCardinality, numRefCoords, dim);


           DynRankView ConstructWithLabel(basisValuesAtRefCoords, basisCardinality, numRefCoords, dim);
           basis.getValues(basisValuesAtRefCoords, refPoints);
           rst::clone(basisValuesAtRefCoordsCells,basisValuesAtRefCoords);

           // modify basis values to account for orientations
           ots::modifyBasisByOrientation(basisValuesAtRefCoordsOriented,
               basisValuesAtRefCoordsCells,
               elemOrts,
               &basis);

           // transform basis values
           DynRankView ConstructWithLabel(jacobianAtRefCoords, numCells, numRefCoords, dim, dim);
           DynRankView ConstructWithLabel(jacobianAtRefCoords_inv, numCells, numRefCoords, dim, dim);
           ct::setJacobian(jacobianAtRefCoords, refPoints, physVertices, pyramid);
           ct::setJacobianInv (jacobianAtRefCoords_inv, jacobianAtRefCoords);
           fst::HCURLtransformVALUE(transformedBasisValuesAtRefCoordsOriented,
               jacobianAtRefCoords_inv,
               basisValuesAtRefCoordsOriented);
           fst::HCURLtransformVALUE(transformedBasisValuesAtRefCoords,
               jacobianAtRefCoords_inv,
               basisValuesAtRefCoords);

           // on the interior, pyramid basis is not polynomial.  We span the appropriate polynomial space in the tangent direction on any face; that's what we test here.
           // exclude all dofs that don't belong to the quad face (which is face 4 in Intrepid2/shards numbering)
           const ordinal_type QUAD_FACE = 4;
           const ordinal_type quadDofCount = basis.getDofCount(QUAD_DIM, QUAD_FACE);
           std::set<ordinal_type> quadDofs;
           for (ordinal_type i=0; i<quadDofCount; i++)
           {
             ordinal_type dofOrdinal = basis.getDofOrdinal(QUAD_DIM, QUAD_FACE, i);
             quadDofs.insert(dofOrdinal);
           }
           std::set<ordinal_type> nonQuadDofs;
           for (ordinal_type i=0; i<basisCardinality; i++)
           {
             if (quadDofs.find(i) == quadDofs.end())
             {
               nonQuadDofs.insert(i);
             }
           }
           
           BasisFunctionsSystem  basisFunctionsSystem(basisCardinality, numRefCoords, 3, nonQuadDofs);
 
           // cross with the normal
           auto transformedBasisValuesAtRefCoordsOriented_tangent = ...; // TODO: finish this
           auto funAtPhysRefCoords_tangent = ...; // TODO: finish this
           
           DynRankView ConstructWithLabel(tangentBasisValues, numCells, basisCardinality, numRefCoords, dim);
           Kokkos::deep_copy(tangentBasisValues, transformedBasisValuesAtRefCoordsOriented_tangent);
           
           DynRankView ConstructWithLabel(funTangent, numCells, numRefCoords);
           Kokkos::deep_copy(funNormal, funAtPhysRefCoords_tangent);
           
           auto info = basisFunctionsSystem.computeBasisCoeffs(basisCoeffs, errorFlag, tangentBasisValues, funTangent);
 
           for (int ic =0; ic < numCells; ++ic) {
             if(info[ic] != 0) {
               errorFlag++;
               *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
               *outStream << "LAPACK error flag for cell " << ic << " is: " << info[ic] << std::endl;
             }
           }

           TestResults testResults(basisCoeffs, transformedBasisValuesAtRefCoords, transformedBasisValuesAtRefCoordsOriented,
                                   tangentBasisValues, funTangent, elemOrts, basisPtr);
           testResults.test(errorFlag,tol);
           delete basisPtr;
         }
        }*/


        //HDIV Case
        {
          // compute reference points on the shared face of the pyramid
          shards::CellTopology tri(shards::getCellTopologyData<shards::Triangle<3> >());
          auto refPointsTri = getInputPointsView<double,DeviceType>(tri, order);
          ordinal_type numRefCoords = refPointsTri.extent_int(0);
          
          // refPointsTri will include the apex of the pyramid, which is an excluded (singular) point.
          // to correct this, we move the points toward the center by a factor of s=0.1:
          
          const double s = 0.1;
          const double centroid_x = 1./3.;
          const double centroid_y = 1./3.;
          using ExecutionSpace = typename DynRankView::execution_space;
          Kokkos::RangePolicy < ExecutionSpace > policy(0,numRefCoords);
          Kokkos::parallel_for( policy,
          KOKKOS_LAMBDA (const ordinal_type &pointOrdinal )
          {
            const auto p_x = refPointsTri(pointOrdinal,0);
            const auto p_y = refPointsTri(pointOrdinal,1);
            refPointsTri(pointOrdinal,0) = (1.0 - s) * p_x + s * centroid_x;
            refPointsTri(pointOrdinal,1) = (1.0 - s) * p_y + s * centroid_y;
          });
          Kokkos::fence();
          
          DynRankView ConstructWithLabel(refPointsMultiCell, 2, numRefCoords, dim);
          auto refPoints0 = Kokkos::subview(refPointsMultiCell, 0, Kokkos::ALL(), Kokkos::ALL());
          auto refPoints1 = Kokkos::subview(refPointsMultiCell, 1, Kokkos::ALL(), Kokkos::ALL());
          
          // map the ref points to the pyramid
          const ordinal_type FACE_DIM  = 2;
          CellTools<DeviceType>::mapToReferenceSubcell(refPoints0, refPointsTri,
                                                       FACE_DIM, faceIndex[0],
                                                       pyramid);
          CellTools<DeviceType>::mapToReferenceSubcell(refPoints1, refPointsTri,
                                                       FACE_DIM, faceIndex[1],
                                                       pyramid);
          
          // compute orientations for cells (one time computation)
          DynRankViewInt elemNodes(&pyrs[0][0], numCells, numElemVertices);
          Kokkos::DynRankView<Orientation,DeviceType> elemOrts("elemOrts", numCells);
          ots::getOrientation(elemOrts, elemNodes, pyramid);

          //Compute physical Dof Coordinates and Reference coordinates
          DynRankView ConstructWithLabel(physRefCoords, numCells, numRefCoords, dim);
          {
            Basis_HGRAD_PYR_C1_FEM<DeviceType,ValueType,ValueType> pyramidLinearBasis; //used for computing physical coordinates
            DynRankView ConstructWithLabel(pyramidLinearBasisValuesAtRefCoords0, pyramid.getNodeCount(), numRefCoords);
            DynRankView ConstructWithLabel(pyramidLinearBasisValuesAtRefCoords1, pyramid.getNodeCount(), numRefCoords);
            pyramidLinearBasis.getValues(pyramidLinearBasisValuesAtRefCoords0, refPoints0);
            pyramidLinearBasis.getValues(pyramidLinearBasisValuesAtRefCoords1, refPoints1);
            std::vector<DynRankView> pyramidLinearBasisValuesAtRefCoords {pyramidLinearBasisValuesAtRefCoords0, pyramidLinearBasisValuesAtRefCoords1};
            for(ordinal_type i=0; i<numCells; ++i)
              for(ordinal_type d=0; d<dim; ++d) {
                for(ordinal_type j=0; j<numRefCoords; ++j)
                  for(std::size_t k=0; k<pyramid.getNodeCount(); ++k)
                    physRefCoords(i,j,d) += vertices[pyrs[i][k]][d]*pyramidLinearBasisValuesAtRefCoords[i](k,j);
              }
          }
          
          FunDivTriFace fun;
          DynRankView ConstructWithLabel(funAtPhysRefCoords, numCells, numRefCoords, dim);
          for(ordinal_type i=0; i<numCells; ++i) {
            for(ordinal_type j=0; j<numRefCoords; ++j) {
              for(ordinal_type k=0; k<dim; ++k)
                funAtPhysRefCoords(i,j,k) = fun(physRefCoords(i,j,0), physRefCoords(i,j,1), physRefCoords(i,j,2), k);
            }
          }
          basis_set.clear();
          basis_set.push_back(new typename  CG_HBasis::HDIV_PYR(order));

          for (auto basisPtr:basis_set) {
            auto& basis = *basisPtr;
            auto name = basis.getName();
            *outStream << " " << name << std::endl;
            ordinal_type basisCardinality = basis.getCardinality();

            //check that fun values at reference points coincide with those computed using basis functions
            DynRankView ConstructWithLabel(basisValuesAtRefCoordsOriented, numCells, basisCardinality, numRefCoords, dim);
            DynRankView ConstructWithLabel(transformedBasisValuesAtRefCoordsOriented, numCells, basisCardinality, numRefCoords, dim);
            DynRankView ConstructWithLabel(transformedBasisValuesAtRefCoords, numCells, basisCardinality, numRefCoords, dim);
            
            DynRankView basisValuesAtRefCoordsCells("basisValuesAtRefCoordsCell", 2, basisCardinality, numRefCoords, dim);
            
            auto basisValuesAtRefCoordsCell0 = Kokkos::subview(basisValuesAtRefCoordsCells, 0, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
            auto basisValuesAtRefCoordsCell1 = Kokkos::subview(basisValuesAtRefCoordsCells, 1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
            
            basis.getValues(basisValuesAtRefCoordsCell0, refPoints0);
            basis.getValues(basisValuesAtRefCoordsCell1, refPoints1);
            
            // modify basis values to account for orientations
            ots::modifyBasisByOrientation(basisValuesAtRefCoordsOriented,
                basisValuesAtRefCoordsCells,
                elemOrts,
                &basis);

            // transform basis values
            DynRankView ConstructWithLabel(jacobianAtRefCoords, numCells, numRefCoords, dim, dim);
            DynRankView ConstructWithLabel(jacobianAtRefCoords_det, numCells, numRefCoords);
            ct::setJacobian(jacobianAtRefCoords, refPointsMultiCell, physVertices, pyramid);
            ct::setJacobianDet (jacobianAtRefCoords_det, jacobianAtRefCoords);
            
            fst::HDIVtransformVALUE(transformedBasisValuesAtRefCoordsOriented,
                jacobianAtRefCoords,
                jacobianAtRefCoords_det,
                basisValuesAtRefCoordsOriented);
            fst::HDIVtransformVALUE(transformedBasisValuesAtRefCoords,
                jacobianAtRefCoords,
                jacobianAtRefCoords_det,
                basisValuesAtRefCoordsCells);

            DynRankView ConstructWithLabel(basisCoeffs, numCells, basisCardinality);

            // on the interior, pyramid basis is not polynomial.  We span the appropriate polynomial space in the normal direction on any face; that's what we test here.
            // exclude all dofs that don't belong to the shared face (which is face 4 in Intrepid2/shards numbering)
            const ordinal_type triDofCount0 = basis.getDofCount(FACE_DIM, faceIndex[0]); // cell 0's dofs on the face
            const ordinal_type triDofCount1 = basis.getDofCount(FACE_DIM, faceIndex[1]); // cell 1's dofs on the face
            std::set<ordinal_type> triDofs0, triDofs1;
            for (ordinal_type i=0; i<triDofCount0; i++)
            {
              ordinal_type dofOrdinal0 = basis.getDofOrdinal(FACE_DIM, faceIndex[0], i);
              triDofs0.insert(dofOrdinal0);
            }
            for (ordinal_type i=0; i<triDofCount1; i++)
            {
              ordinal_type dofOrdinal1 = basis.getDofOrdinal(FACE_DIM, faceIndex[1], i);
              triDofs1.insert(dofOrdinal1);
            }
            std::set<ordinal_type> nonTriDofs0, nonTriDofs1;
            for (ordinal_type i=0; i<basisCardinality; i++)
            {
              if (triDofs0.find(i) == triDofs0.end())
              {
                nonTriDofs0.insert(i);
              }
              if (triDofs1.find(i) == triDofs1.end())
              {
                nonTriDofs1.insert(i);
              }
            }
            
            BasisFunctionsSystem  basisFunctionsSystem(basisCardinality, numRefCoords, 1, nonTriDofs0, nonTriDofs1); // dim 1: matching the normal component on a face

            // dot with the normal (here, by taking the y component)
            auto transformedBasisValuesAtRefCoordsOriented_y = Kokkos::subview(transformedBasisValuesAtRefCoordsOriented,
                                                                               Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 1);
            auto funAtPhysRefCoords_y = Kokkos::subview(funAtPhysRefCoords,
                                                        Kokkos::ALL(), Kokkos::ALL(), 1);
            
            DynRankView ConstructWithLabel(normalBasisValues, numCells, basisCardinality, numRefCoords);
            Kokkos::deep_copy(normalBasisValues, transformedBasisValuesAtRefCoordsOriented_y);
            
            DynRankView ConstructWithLabel(funNormal, numCells, numRefCoords);
            Kokkos::deep_copy(funNormal, funAtPhysRefCoords_y);
            
            auto info = basisFunctionsSystem.computeBasisCoeffs(basisCoeffs, errorFlag, normalBasisValues, funNormal);

            for (int ic =0; ic < numCells; ++ic) {
              if(info[ic] != 0) {
                errorFlag++;
                *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
                *outStream << "LAPACK error flag for cell " << ic << " is: " << info[ic] << std::endl;
              }
            }

            TestResults testResults(basisCoeffs, transformedBasisValuesAtRefCoords, transformedBasisValuesAtRefCoordsOriented, 
                                    normalBasisValues, funNormal, elemOrts, basisPtr);
            testResults.test(errorFlag,tol);
            delete basisPtr;
          }
        }
      } //rotation of first cell vertices
    } while(std::next_permutation(&reorder[0]+2, &reorder[0]+5)); //reorder vertices of common face

  } catch (std::exception &err) {
    std::cout << " Exception\n";
    *outStream << err.what() << "\n\n";
    errorFlag = -1000;
  }


  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED = " << errorFlag << "\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  // reset format state of std::cout
  std::cout.copyfmt(oldFormatState);

  return errorFlag;
}
}
}

