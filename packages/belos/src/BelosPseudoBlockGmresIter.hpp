    // @HEADER
    // *****************************************************************************
    //                 Belos: Block Linear Solvers Package
    //
    // Copyright 2004-2016 NTESS and the Belos contributors.
    // SPDX-License-Identifier: BSD-3-Clause
    // *****************************************************************************
    // @HEADER

    #ifndef BELOS_PSEUDO_BLOCK_GMRES_ITER_HPP
    #define BELOS_PSEUDO_BLOCK_GMRES_ITER_HPP

    /*! \file BelosPseudoBlockGmresIter.hpp
        \brief Belos implementation of the pseudo-block GMRES iteration.
    */

    #include "BelosConfigDefs.hpp"
    #include "BelosTypes.hpp"
    #include "BelosIteration.hpp"

    #include "BelosLinearProblem.hpp"
    #include "BelosMatOrthoManager.hpp"
    #include "BelosOutputManager.hpp"
    #include "BelosStatusTest.hpp"
    #include "BelosOperatorTraits.hpp"
    #include "BelosMultiVecTraits.hpp"

    #include "Teuchos_BLAS.hpp"
    #include "Teuchos_SerialDenseMatrix.hpp"
    #include "Teuchos_SerialDenseVector.hpp"
    #include "Teuchos_ScalarTraits.hpp"
    #include "Teuchos_ParameterList.hpp"
    #include "Teuchos_TimeMonitor.hpp"

    /*!
    \class Belos::PseudoBlockGmresIter

    \brief Implementation of the pseudo-block GMRES iteration.
    
    This class performs the pseudo-block GMRES iteration, constructing a block Krylov 
    subspace for all linear systems simultaneously. For each iteration, the algorithm:
    1. Applies the operator to the latest basis vector
    2. Orthogonalizes the result against previous basis vectors
    3. Updates the Hessenberg matrix
    4. Applies Givens rotations to maintain the QR factorization
    5. Computes the residual norm for each system

    \ingroup belos_solver_framework
    \author Heidi Thornquist
    */

    namespace Belos {
    
    //! @name PseudoBlockGmresIter Structures 
    //@{ 
    
    /** \brief Structure holding the solver state variables.
     *
     * This struct is used by initialize() and getState() methods to store or retrieve 
     * the complete state of the solver.
     */
    template <class ScalarType, class MV>
    struct PseudoBlockGmresIterState {
        typedef Teuchos::ScalarTraits<ScalarType> SCT;
        typedef typename SCT::magnitudeType MagnitudeType;

        //! Current dimension of the Krylov subspace
        int curDim;
        
        //! Krylov basis vectors [V₁, V₂, ..., Vₘ]
        std::vector<Teuchos::RCP<const MV> > V;
        
        //! Hessenberg matrix from the Arnoldi process
        std::vector<Teuchos::RCP<const Teuchos::SerialDenseMatrix<int,ScalarType> > > H;
        
        //! Upper-triangular matrix from QR factorization of H
        std::vector<Teuchos::RCP<const Teuchos::SerialDenseMatrix<int,ScalarType> > > R;
        
        //! Right-hand side of the least squares problem RY = Z
        std::vector<Teuchos::RCP<const Teuchos::SerialDenseVector<int,ScalarType> > > Z;
        
        //! Sine components of Givens rotations
        std::vector<Teuchos::RCP<const Teuchos::SerialDenseVector<int,ScalarType> > > sn;
        
        //! Cosine components of Givens rotations
        std::vector<Teuchos::RCP<const Teuchos::SerialDenseVector<int,MagnitudeType> > > cs;

        //! Constructor initializes all members to empty
        PseudoBlockGmresIterState() : curDim(0), V(0), H(0), R(0), Z(0), sn(0), cs(0) {}
    };
    
    //! @name PseudoBlockGmresIter Exceptions
    //@{ 
    
    /** \brief Exception thrown when orthogonalization fails.
     *
     * This exception is thrown when the orthogonalization manager cannot
     * generate orthonormal columns for the new basis vectors.
     */
    class PseudoBlockGmresIterOrthoFailure : public BelosError {
    public:
        PseudoBlockGmresIterOrthoFailure(const std::string& what_arg) : BelosError(what_arg) {}
    };
    
    //@}
    
    /** \class PseudoBlockGmresIter
     *  \brief Implementation of the GMRES iteration for multiple right-hand sides.
     */
    template<class ScalarType, class MV, class OP>
    class PseudoBlockGmresIter : virtual public Iteration<ScalarType,MV,OP> {
        
    public:
        // Convenience typedefs
        typedef MultiVecTraits<ScalarType,MV> MVT;
        typedef OperatorTraits<ScalarType,MV,OP> OPT;
        typedef Teuchos::ScalarTraits<ScalarType> SCT;
        typedef typename SCT::magnitudeType MagnitudeType;
        
        //! @name Constructors/Destructor
        //@{ 
        
        /** \brief Constructor with linear problem and solver parameters.
         *
         * This constructor takes the linear problem, output manager, status tester, 
         * orthogonalization manager, and solver parameters.
         *
         * Required parameters:
         * - "Num Blocks" - Maximum dimension of the Krylov subspace
         */
        PseudoBlockGmresIter(
            const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > &problem, 
            const Teuchos::RCP<OutputManager<ScalarType> > &printer,
            const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
            const Teuchos::RCP<MatOrthoManager<ScalarType,MV,OP> > &ortho,
            Teuchos::ParameterList &params);
        
        //! Destructor
        virtual ~PseudoBlockGmresIter() {};
        //@}
        
        //! @name Solver methods
        //@{ 
        
        /** \brief Performs GMRES iterations.
         *
         * This method performs GMRES iterations until the status test passes
         * or the maximum number of iterations is reached. The method:
         * 1. Initializes the solver if needed
         * 2. Applies the operator to the current vector
         * 3. Orthogonalizes the result against previous basis vectors
         * 4. Updates the Hessenberg matrix
         * 5. Applies Givens rotations to maintain the QR factorization
         * 6. Checks convergence via the status test
         */
        void iterate();
        
        /** \brief Initialize the solver with provided state.
         *
         * This method initializes all internal data structures based on the
         * provided state. It allows manually setting the Krylov basis,
         * Hessenberg matrix, and other components of the GMRES algorithm.
         *
         * \param newstate The state to use for initialization
         */
        void initialize(const PseudoBlockGmresIterState<ScalarType,MV> & newstate);

        /** \brief Initialize the solver with default empty state.
         *
         * This method calls initialize() with an empty state object,
         * which will create all necessary data structures.
         */
        void initialize() {
        PseudoBlockGmresIterState<ScalarType,MV> empty;
        initialize(empty);
        }

        /** \brief Get the current solver state.
         *
         * Returns an object containing pointers to the current state of the solver,
         * including the Krylov basis, Hessenberg matrix, and other internal data.
         *
         * \return A PseudoBlockGmresIterState object with the current state
         */
        PseudoBlockGmresIterState<ScalarType,MV> getState() const {
        PseudoBlockGmresIterState<ScalarType,MV> state;
        state.curDim = curDim_;
        state.V.resize(numRHS_);
        state.H.resize(numRHS_);
        state.Z.resize(numRHS_);
        state.sn.resize(numRHS_);
        state.cs.resize(numRHS_);
        for (int i=0; i<numRHS_; ++i) {
            state.V[i] = V_[i];
            state.H[i] = H_[i];
            state.Z[i] = Z_[i];
            state.sn[i] = sn_[i];
            state.cs[i] = cs_[i];
        }
        return state;
        }
        //@}

        //! @name Status methods
        //@{ 
        
        //! Get the current iteration count
        int getNumIters() const { return iter_; }
        
        //! Reset the iteration count
        void resetNumIters(int iter = 0) { iter_ = iter; }
        
        /** \brief Get the norms of the current residuals.
         *
         * Fills the provided vector with the norms of the current native residuals.
         * Native residuals are computed internally during GMRES iterations and are
         * cheaper than computing explicit residuals.
         *
         * \param norms Vector to be filled with residual norms
         * \return Always returns Teuchos::null (to satisfy interface)
         */
        Teuchos::RCP<const MV> getNativeResiduals(std::vector<MagnitudeType> *norms) const;
        
        /** \brief Get the current solution update.
         *
         * Computes and returns the current solution update based on the
         * Krylov subspace. This involves solving the least-squares problem
         * for each right-hand side.
         *
         * \return Multivector containing the solution updates, or null if no update is available
         */
        Teuchos::RCP<MV> getCurrentUpdate() const;
        
        /** \brief Update the QR factorization of the Hessenberg matrix.
         *
         * Updates the QR factorization of the upper Hessenberg matrix using
         * Givens rotations.
         *
         * \param dim Dimension for which to update (defaults to current dimension)
         */
        void updateLSQR(int dim = -1);
        
        //! Get the current dimension of the Krylov subspace
        int getCurSubspaceDim() const { 
        if (!initialized_) return 0;
        return curDim_;
        }
        
        //! Get the maximum allowed dimension of the Krylov subspace
        int getMaxSubspaceDim() const { return numBlocks_; }
        //@}
        
        //! @name Accessor methods
        //@{ 
        
        //! Get the linear problem being solved
        const LinearProblem<ScalarType,MV,OP>& getProblem() const { return *lp_; }
        
        //! Get the block size (always 1 for this implementation)
        int getBlockSize() const { return 1; }
        
        //! Set the block size (must be 1 for this implementation)
        void setBlockSize(int blockSize) { 
        TEUCHOS_TEST_FOR_EXCEPTION(
            blockSize != 1, 
            std::invalid_argument,
            "Belos::PseudoBlockGmresIter::setBlockSize(): Block size must be one."
        );
        }
        
        //! Get the maximum number of blocks used by the solver
        int getNumBlocks() const { return numBlocks_; }
        
        //! Set the maximum number of blocks used by the solver
        void setNumBlocks(int numBlocks);
        
        //! Check if the solver has been initialized
        bool isInitialized() { return initialized_; }
        //@}
        
    private:
        // Linear problem and solver components
        const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > lp_;         // Linear problem to solve
        const Teuchos::RCP<OutputManager<ScalarType> > om_;               // Output manager
        const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > stest_;         // Status tester
        const Teuchos::RCP<OrthoManager<ScalarType,MV> > ortho_;          // Orthogonalization manager
        
        // Algorithmic parameters
        int numRHS_;      // Number of right-hand sides (linear systems)
        int numBlocks_;   // Maximum dimension of Krylov subspace
        
        // Givens rotation components
        std::vector<Teuchos::RCP<Teuchos::SerialDenseVector<int,ScalarType> > > sn_;     // Sine coefficients
        std::vector<Teuchos::RCP<Teuchos::SerialDenseVector<int,MagnitudeType> > > cs_;  // Cosine coefficients
        
        // Temporary work vectors
        Teuchos::RCP<MV> U_vec_, AU_vec_;    // Vectors used during iterations

        // Current RHS and solution vectors
        Teuchos::RCP<MV> cur_block_rhs_, cur_block_sol_;

        // Solver state
        bool initialized_;    // Whether the solver has been initialized
        int curDim_;          // Current dimension of the Krylov subspace
        int iter_;            // Current iteration count

        // State storage
        std::vector<Teuchos::RCP<MV> > V_;   // Krylov basis vectors
        
        // Matrices from the Arnoldi process
        std::vector<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > H_;  // Hessenberg matrix
        
        // QR factorization components
        std::vector<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > R_;  // Upper triangular matrix
        std::vector<Teuchos::RCP<Teuchos::SerialDenseVector<int,ScalarType> > > Z_;  // Transformed right-hand side
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    template<class ScalarType, class MV, class OP>
    PseudoBlockGmresIter<ScalarType,MV,OP>::PseudoBlockGmresIter(
        const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > &problem,
        const Teuchos::RCP<OutputManager<ScalarType> > &printer,
        const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
        const Teuchos::RCP<MatOrthoManager<ScalarType,MV,OP> > &ortho,
        Teuchos::ParameterList &params) :
        lp_(problem),
        om_(printer),
        stest_(tester),
        ortho_(ortho),
        numRHS_(0),
        numBlocks_(0),
        initialized_(false),
        curDim_(0),
        iter_(0)
    {
        // Get the maximum number of blocks allowed for each Krylov subspace
        TEUCHOS_TEST_FOR_EXCEPTION(
        !params.isParameter("Num Blocks"), 
        std::invalid_argument,
        "Belos::PseudoBlockGmresIter::constructor: Required parameter 'Num Blocks' is not specified."
        );
        
        int nb = Teuchos::getParameter<int>(params, "Num Blocks");
        setNumBlocks(nb);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Set the maximum number of blocks and reset state
    template <class ScalarType, class MV, class OP>
    void PseudoBlockGmresIter<ScalarType,MV,OP>::setNumBlocks(int numBlocks)
    {
        // This only allocates space; doesn't perform any computation
        // Any change in size will invalidate the solver state
        
        TEUCHOS_TEST_FOR_EXCEPTION(
        numBlocks <= 0, 
        std::invalid_argument, 
        "Belos::PseudoBlockGmresIter::setNumBlocks: Number of blocks must be positive."
        );

        numBlocks_ = numBlocks;
        curDim_ = 0;
        initialized_ = false;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute and return the current solution update
    template <class ScalarType, class MV, class OP>
    Teuchos::RCP<MV> PseudoBlockGmresIter<ScalarType,MV,OP>::getCurrentUpdate() const
    {
        // If this is the first iteration, there is no update
        if (curDim_ == 0) {
        return Teuchos::null;
        }
        
        // Create vector to hold the update
        Teuchos::RCP<MV> currentUpdate = MVT::Clone(*(V_[0]), numRHS_);
        
        // Set up indices
        std::vector<int> index(1);
        std::vector<int> basisIndices(curDim_);
        for (int i = 0; i < curDim_; ++i) {
        basisIndices[i] = i;
        }
        
        // Constants
        const ScalarType one = SCT::one();
        const ScalarType zero = SCT::zero();
        
        // BLAS interface
        Teuchos::BLAS<int, ScalarType> blas;
        
        // For each right-hand side
        for (int i = 0; i < numRHS_; ++i) {
            // Get view of the current column of the update
            index[0] = i;
            Teuchos::RCP<MV> curUpdateCol = MVT::CloneViewNonConst(*currentUpdate, index);
            
            // Copy RHS of least squares problem (don't modify original)
            Teuchos::SerialDenseVector<int, ScalarType> y(Teuchos::Copy, Z_[i]->values(), curDim_);
            
            // Solve the least squares problem: Hy = Z
            blas.TRSM(
                Teuchos::LEFT_SIDE, 
                Teuchos::UPPER_TRI, 
                Teuchos::NO_TRANS,
                Teuchos::NON_UNIT_DIAG, 
                curDim_, 
                1, 
                one,
                H_[i]->values(), 
                H_[i]->stride(), 
                y.values(), 
                y.stride()
            );
            
            // Compute the solution update: V * y
            Teuchos::RCP<const MV> basisVecs = MVT::CloneView(*(V_[i]), basisIndices);
            MVT::MvTimesMatAddMv(one, *basisVecs, y, zero, *curUpdateCol);
        }
        
        return currentUpdate;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Get the native residual norms
    template <class ScalarType, class MV, class OP>
    Teuchos::RCP<const MV>
    PseudoBlockGmresIter<ScalarType,MV,OP>::getNativeResiduals(std::vector<MagnitudeType> *norms) const
    {
        // If norms is provided, fill it with the current residual norms
        if (norms) {
        // Resize if necessary
        if (static_cast<int>(norms->size()) < numRHS_) {
            norms->resize(numRHS_);
        }
        
        // For each right-hand side, get the residual from the last element of Z
        for (int j = 0; j < numRHS_; ++j) {
            const ScalarType curNativeResid = (*Z_[j])(curDim_);
            (*norms)[j] = SCT::magnitude(curNativeResid);
        }
        }
        
        // Always return null (per interface requirement)
        return Teuchos::null;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Initialize the solver with the provided state
    template <class ScalarType, class MV, class OP>
    void
    PseudoBlockGmresIter<ScalarType,MV,OP>::initialize(const PseudoBlockGmresIterState<ScalarType,MV> & newstate)
    {
        using Teuchos::RCP;

        // Get the number of right-hand sides from the linear problem
        this->numRHS_ = MVT::GetNumberVecs(*(lp_->getCurrLHSVec()));

        // Error message for inconsistent states
        std::string errstr("Belos::PseudoBlockGmresIter::initialize(): "
                        "Specified multivectors must have consistent dimensions.");

        // Ensure newstate has non-empty V and Z arrays
        TEUCHOS_TEST_FOR_EXCEPTION(
        (int)newstate.V.size() == 0 || (int)newstate.Z.size() == 0,
        std::invalid_argument,
        "Belos::PseudoBlockGmresIter::initialize(): "
        "V and/or Z arrays cannot be empty."
        );

        // Find a multivector to clone from (prefer RHS, fall back to LHS)
        RCP<const MV> lhsMV = lp_->getLHS();
        RCP<const MV> rhsMV = lp_->getRHS();
        RCP<const MV> cloneSource = rhsMV.is_null() ? lhsMV : rhsMV;

        TEUCHOS_TEST_FOR_EXCEPTION(
        cloneSource.is_null(),
        std::invalid_argument,
        "Belos::PseudoBlockGmresIter::initialize(): "
        "Linear problem must provide non-null multivectors."
        );

        // Ensure the current dimension doesn't exceed maximum blocks
        TEUCHOS_TEST_FOR_EXCEPTION(
        newstate.curDim > numBlocks_ + 1,
        std::invalid_argument,
        errstr
        );
        
        curDim_ = newstate.curDim;

        // Initialize Krylov basis vectors (V)
        V_.resize(numRHS_);
        for (int i = 0; i < numRHS_; ++i) {
        // Create or resize vectors as needed
        if (V_[i].is_null() || MVT::GetNumberVecs(*V_[i]) < numBlocks_ + 1) {
            V_[i] = MVT::Clone(*cloneSource, numBlocks_ + 1);
        }
        
        // Check dimensions
        TEUCHOS_TEST_FOR_EXCEPTION(
            MVT::GetGlobalLength(*newstate.V[i]) != MVT::GetGlobalLength(*V_[i]),
            std::invalid_argument,
            errstr
        );
        TEUCHOS_TEST_FOR_EXCEPTION(
            MVT::GetNumberVecs(*newstate.V[i]) < newstate.curDim,
            std::invalid_argument,
            errstr
        );
        
        // Copy vectors if not identical
        int lclDim = MVT::GetNumberVecs(*newstate.V[i]);
        if (newstate.V[i] != V_[i]) {
            // Warn if discarding vectors due to block size
            if (curDim_ == 0 && lclDim > 1) {
            om_->stream(Warnings)
                << "Belos::PseudoBlockGmresIter::initialize(): Solver initialized with "
                << lclDim << " vectors, but block size is 1. Discarding extra vectors."
                << std::endl;
            }
            
            // Copy vectors up to current dimension
            std::vector<int> indices(curDim_ + 1);
            for (int j = 0; j < curDim_ + 1; ++j) {
            indices[j] = j;
            }

            // Clone and copy
            RCP<const MV> newV = MVT::CloneView(*newstate.V[i], indices);
            RCP<MV> lclV = MVT::CloneViewNonConst(*V_[i], indices);
            const ScalarType one = SCT::one();
            const ScalarType zero = SCT::zero();
            MVT::MvAddMv(one, *newV, zero, *newV, *lclV);
        }
        }

        // Initialize right-hand side vectors (Z)
        Z_.resize(numRHS_);
        for (int i = 0; i < numRHS_; ++i) {
        // Create or resize vectors as needed
        if (Z_[i].is_null()) {
            Z_[i] = Teuchos::rcp(new Teuchos::SerialDenseVector<int, ScalarType>());
        }
        if (Z_[i]->length() < numBlocks_ + 1) {
            Z_[i]->shapeUninitialized(numBlocks_ + 1, 1);
        }
        
        // Check dimensions
        TEUCHOS_TEST_FOR_EXCEPTION(
            newstate.Z[i]->numRows() < curDim_,
            std::invalid_argument,
            errstr
        );
        
        // Copy vectors if not identical
        if (newstate.Z[i] != Z_[i]) {
            // Initialize with zeros if empty
            if (curDim_ == 0) {
            Z_[i]->putScalar();
            }
            
            // Copy values
            Teuchos::SerialDenseVector<int, ScalarType> newZ(
            Teuchos::View, newstate.Z[i]->values(), curDim_ + 1
            );
            RCP<Teuchos::SerialDenseVector<int, ScalarType>> lclZ = 
            Teuchos::rcp(new Teuchos::SerialDenseVector<int, ScalarType>(
                Teuchos::View, Z_[i]->values(), curDim_ + 1
            ));
            lclZ->assign(newZ);
        }
        }

        // Initialize Hessenberg matrices (H)
        H_.resize(numRHS_);
        for (int i = 0; i < numRHS_; ++i) {
        // Create or resize matrices as needed
        if (H_[i].is_null()) {
            H_[i] = Teuchos::rcp(new Teuchos::SerialDenseMatrix<int, ScalarType>());
        }
        if (H_[i]->numRows() < numBlocks_ + 1 || H_[i]->numCols() < numBlocks_) {
            H_[i]->shapeUninitialized(numBlocks_ + 1, numBlocks_);
        }
        
        // Copy matrices if provided and not identical
        if ((int)newstate.H.size() == numRHS_) {
            // Check dimensions
            TEUCHOS_TEST_FOR_EXCEPTION(
            (newstate.H[i]->numRows() < curDim_ || newstate.H[i]->numCols() < curDim_),
            std::invalid_argument,
            "Belos::PseudoBlockGmresIter::initialize(): "
            "Hessenberg matrices must be consistent with current dimension."
            );
            
            if (newstate.H[i] != H_[i]) {
            // Copy values
            Teuchos::SerialDenseMatrix<int, ScalarType> newH(
                Teuchos::View, *newstate.H[i], curDim_ + 1, curDim_
            );
            RCP<Teuchos::SerialDenseMatrix<int, ScalarType>> lclH =
                Teuchos::rcp(new Teuchos::SerialDenseMatrix<int, ScalarType>(
                Teuchos::View, *H_[i], curDim_ + 1, curDim_
                ));
            lclH->assign(newH);
            }
        }
        }

        // Initialize Givens rotation components
        cs_.resize(numRHS_);
        sn_.resize(numRHS_);
        
        // Copy rotation angles if provided
        if ((int)newstate.cs.size() == numRHS_ && (int)newstate.sn.size() == numRHS_) {
        for (int i = 0; i < numRHS_; ++i) {
            if (cs_[i] != newstate.cs[i]) {
            cs_[i] = Teuchos::rcp(new Teuchos::SerialDenseVector<int, MagnitudeType>(*newstate.cs[i]));
            }
            if (sn_[i] != newstate.sn[i]) {
            sn_[i] = Teuchos::rcp(new Teuchos::SerialDenseVector<int, ScalarType>(*newstate.sn[i]));
            }
        }
        }
        
        // Create or resize rotation vectors
        for (int i = 0; i < numRHS_; ++i) {
        if (cs_[i].is_null()) {
            cs_[i] = Teuchos::rcp(new Teuchos::SerialDenseVector<int, MagnitudeType>(numBlocks_ + 1));
        } else {
            cs_[i]->resize(numBlocks_ + 1);
        }
        
        if (sn_[i].is_null()) {
            sn_[i] = Teuchos::rcp(new Teuchos::SerialDenseVector<int, ScalarType>(numBlocks_ + 1));
        } else {
            sn_[i]->resize(numBlocks_ + 1);
        }
        }

        // Mark solver as initialized
        initialized_ = true;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Main iteration method
    template <class ScalarType, class MV, class OP>
    void PseudoBlockGmresIter<ScalarType,MV,OP>::iterate()
    {
        // Initialize if needed
        if (!initialized_) {
            initialize();
        }

        // Constants
        const ScalarType one = SCT::one();
        const ScalarType zero = SCT::zero();
        
        // Maximum search dimension
        int searchDim = numBlocks_;
        
        // Set up indices and work vectors
        std::vector<int> index(1);
        std::vector<int> index2(1);
        index[0] = curDim_;
        
        // Clone vectors for the current basis vector and its image
        Teuchos::RCP<MV> U_vec = MVT::Clone(*V_[0], numRHS_);
        Teuchos::RCP<MV> AU_vec = MVT::Clone(*V_[0], numRHS_);

        // Copy current basis vector to U_vec
        for (int i = 0; i < numRHS_; ++i) {
            index2[0] = i;
            Teuchos::RCP<const MV> tmp_vec = MVT::CloneView(*V_[i], index);
            Teuchos::RCP<MV> U_vec_view = MVT::CloneViewNonConst(*U_vec, index2);
            MVT::MvAddMv(one, *tmp_vec, zero, *tmp_vec, *U_vec_view);
        }
        
        // Main iteration loop
        // Continue until status test passes or basis is full
        while (stest_->checkStatus(this) != Passed && curDim_ < searchDim) {
            // Increment iteration counter
            iter_++;
            
            // Apply operator: AU_vec = A * U_vec
            lp_->apply(*U_vec, *AU_vec);
            
            // Resize index for all previous vectors
            int num_prev = curDim_ + 1;
            index.resize(num_prev);
            for (int i = 0; i < num_prev; ++i) {
                index[i] = i;
            }
            
            // Orthogonalize new Krylov vector for each right-hand side
            for (int i = 0; i < numRHS_; ++i) {
                // Get view of previous Krylov vectors
                Teuchos::RCP<const MV> V_prev = MVT::CloneView(*V_[i], index);
                Teuchos::Array<Teuchos::RCP<const MV>> V_array(1, V_prev);
                
                // Get view of new candidate vector
                index2[0] = i;
                Teuchos::RCP<MV> V_new = MVT::CloneViewNonConst(*AU_vec, index2);
                
                // Get view of current Hessenberg column
                Teuchos::RCP<Teuchos::SerialDenseMatrix<int, ScalarType>> h_new =
                Teuchos::rcp(new Teuchos::SerialDenseMatrix<int, ScalarType>(
                    Teuchos::View, *H_[i], num_prev, 1, 0, curDim_
                ));
                Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int, ScalarType>>> h_array(1, h_new);
                
                // Get view for the subdiagonal element
                Teuchos::RCP<Teuchos::SerialDenseMatrix<int, ScalarType>> r_new =
                Teuchos::rcp(new Teuchos::SerialDenseMatrix<int, ScalarType>(
                    Teuchos::View, *H_[i], 1, 1, num_prev, curDim_
                ));
                
                // Orthonormalize against previous vectors
                ortho_->projectAndNormalize(*V_new, h_array, r_new, V_array);
                
                // Copy orthonormalized vector back to the Krylov basis
                index2[0] = curDim_ + 1;
                Teuchos::RCP<MV> tmp_vec = MVT::CloneViewNonConst(*V_[i], index2);
                MVT::MvAddMv(one, *V_new, zero, *V_new, *tmp_vec);
            }
            
            // Swap vectors for next iteration
            // AU_vec is now normalized and ready to be the next U_vec
            Teuchos::RCP<MV> tmp_AU_vec = U_vec;
            U_vec = AU_vec;
            AU_vec = tmp_AU_vec;
            
            // Update QR factorization of the Hessenberg matrix
            updateLSQR();
            
            // Increment dimension
            curDim_ += 1;
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Update the QR factorization of the Hessenberg matrix
    template<class ScalarType, class MV, class OP>
    void PseudoBlockGmresIter<ScalarType,MV,OP>::updateLSQR(int dim)
    {
        // Determine the current dimension to update
        int curDim = curDim_;
        if (dim >= curDim_ && dim < getMaxSubspaceDim()) {
            curDim = dim;
        }

        // Zero constant for zeroing elements
        const ScalarType zero = SCT::zero();
        
        // BLAS interface
        Teuchos::BLAS<int, ScalarType> blas;
        
        // Process each right-hand side
        for (int i = 0; i < numRHS_; ++i) {
        // Apply previous Givens rotations to new column of Hessenberg matrix
        for (int j = 0; j < curDim; j++) {
            // Apply rotation: [ c  s ] * [ h_j,k ]
            //                 [-s  c ]   [ h_j+1,k ]
            blas.ROT(
            1,                         // Single element
            &(*H_[i])(j, curDim),     // h_j,k
            1,                         // Stride
            &(*H_[i])(j + 1, curDim), // h_j+1,k
            1,                         // Stride
            &(*cs_[i])[j],            // Cosine
            &(*sn_[i])[j]             // Sine
            );
        }
        
        // Calculate new Givens rotation to zero the subdiagonal element
        blas.ROTG(
            &(*H_[i])(curDim, curDim),     // Diagonal element (will be updated)
            &(*H_[i])(curDim + 1, curDim), // Subdiagonal to be zeroed
            &(*cs_[i])[curDim],            // New cosine
            &(*sn_[i])[curDim]             // New sine
        );
        
        // Zero the subdiagonal element (should already be zero from ROTG)
        (*H_[i])(curDim + 1, curDim) = zero;
        
        // Apply the new rotation to the right-hand side
        blas.ROT(
            1,                       // Single element 
            &(*Z_[i])(curDim),      // Element k of rhs
            1,                       // Stride
            &(*Z_[i])(curDim + 1),  // Element k+1 of rhs
            1,                       // Stride
            &(*cs_[i])[curDim],     // Cosine
            &(*sn_[i])[curDim]      // Sine
        );
        }
    }

    } // end Belos namespace

    #endif /* BELOS_PSEUDO_BLOCK_GMRES_ITER_HPP */
