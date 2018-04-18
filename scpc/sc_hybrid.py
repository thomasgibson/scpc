from firedrake.matrix_free.preconditioners import PCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor, AssembledVector
from pyop2.profiling import timed_region, timed_function


__all__ = ['HybridSCPC']


class HybridSCPC(PCBase):
    """A Slate-based python preconditioner implementation of
    static condensation for three-field hybridized problems. This
    applies to the mixed-hybrid methods, such as the RT-H and BDM-H
    methods, as well as hybridized-DG discretizations like the LDG-H
    method.
    """

    @timed_function("HybridSCInit")
    def initialize(self, pc):
        """Set up the problem context. This takes the incoming
        three-field hybridized system and constructs the static
        condensation operators using Slate expressions.

        A KSP is created for the reduced system for the Lagrange
        multipliers. The scalar and flux fields are reconstructed
        locally.
        """
        from firedrake.assemble import (allocate_matrix,
                                        create_assembly_callable)
        from firedrake.bcs import DirichletBC
        from firedrake.function import Function
        from firedrake.functionspace import FunctionSpace
        from firedrake.interpolation import interpolate

        prefix = pc.getOptionsPrefix() + "hybrid_sc_"
        _, P = pc.getOperators()
        self.cxt = P.getPythonContext()
        if not isinstance(self.cxt, ImplicitMatrixContext):
            raise ValueError("Context must be an ImplicitMatrixContext")

        # Retrieve the mixed function space, which is expected to
        # be of the form: W = (DG_k)^n \times DG_k \times DG_trace
        W = self.cxt.a.arguments()[0].function_space()
        if len(W) != 3:
            raise RuntimeError("Expecting three function spaces.")

        # Assert a specific ordering of the spaces
        # TODO: Clean this up
        assert W[2].ufl_element().family() == "HDiv Trace"

        # This operator has the form:
        # | A  B  C |
        # | D  E  F |
        # | G  H  J |
        # NOTE: It is often the case that D = B.T,
        # G = C.T, H = F.T, and J = 0, but we're not making
        # that assumption here.
        O = Tensor(self.cxt.a)

        # Extract sub-block:
        # | A B |
        # | D E |
        # which has block row indices (0, 1) and block
        # column indices (0, 1) as well.
        M = O.block(((0, 1), (0, 1)))

        # Extract sub-block:
        # | C |
        # | F |
        # which has block row indices (0, 1) and block
        # column indices (2,)
        K = O.block(((0, 1), 2))

        # Extract sub-block:
        # | G H |
        # which has block row indices (2,) and block column
        # indices (0, 1)
        L = O.block((2, (0, 1)))

        # And the final block J has block row-column
        # indices (2, 2)
        J = O.block((2, 2))

        # Schur complement for traces
        S = J - L * M.inv * K

        # Extract trace space
        T = W[2]

        # Need to duplicate a trace space which is NOT
        # associated with a subspace of a mixed space.
        Tr = FunctionSpace(T.mesh(), T.ufl_element())
        bcs = []
        cxt_bcs = self.cxt.row_bcs
        for bc in cxt_bcs:
            assert bc.function_space() == T, (
                "BCs should be imposing vanishing conditions on traces"
            )
            if isinstance(bc.function_arg, Function):
                bc_arg = interpolate(bc.function_arg, Tr)
            else:
                # Constants don't need to be interpolated
                bc_arg = bc.function_arg
            bcs.append(DirichletBC(Tr, bc_arg, bc.sub_domain))

        self.S = allocate_matrix(S,
                                 bcs=bcs,
                                 form_compiler_parameters=self.cxt.fc_params)
        self._assemble_S = create_assembly_callable(
            S,
            tensor=self.S,
            bcs=bcs,
            form_compiler_parameters=self.cxt.fc_params)

        self._assemble_S()
        self.S.force_evaluation()
        Smat = self.S.petscmat

        # Set up ksp for the trace problem
        trace_ksp = PETSc.KSP().create(comm=pc.comm)
        trace_ksp.setOptionsPrefix(prefix)
        trace_ksp.setOperators(Smat)
        trace_ksp.setUp()
        trace_ksp.setFromOptions()
        self.trace_ksp = trace_ksp

        # Local tensors needed for reconstruction
        A = O.block((0, 0))
        B = O.block((0, 1))
        C = O.block((0, 2))
        D = O.block((1, 0))
        E = O.block((1, 1))
        F = O.block((1, 2))
        Se = E - D * A.inv * B
        Sf = F - D * A.inv * C

        # Expression for RHS assembly
        # Set up the functions for trace solution and
        # the residual for the trace system
        self.r_lambda = Function(T)
        self.residual = Function(W)
        self.r_lambda_thunk = Function(T)
        v1, v2, v3 = self.residual.split()

        # Create mixed function for residual computation.
        # This projects the non-trace residual bits into
        # the trace space:
        # -L * M.inv * | v1 v2 |^T
        R = AssembledVector(self.residual)
        v1v2 = R.block(((0, 1),))
        v3 = R.block((2,))
        r_lambda_thunk = v3 - L * M.inv * v1v2
        self._assemble_Srhs_thunk = create_assembly_callable(
            r_lambda_thunk,
            tensor=self.r_lambda_thunk,
            form_compiler_parameters=self.cxt.fc_params)

        self.solution = Function(W)
        q_h, u_h, lambda_h = self.solution.split()

        # Assemble u_h using lambda_h
        self._assemble_u = create_assembly_callable(
            Se.inv * (AssembledVector(v2) -
                      D * A.inv * AssembledVector(v1) -
                      Sf * AssembledVector(lambda_h)),
            tensor=u_h,
            form_compiler_parameters=self.cxt.fc_params)

        # Recover q_h using both u_h and lambda_h
        self._assemble_q = create_assembly_callable(
            A.inv * (AssembledVector(v1) -
                     B * AssembledVector(u_h) -
                     C * AssembledVector(lambda_h)),
            tensor=q_h,
            form_compiler_parameters=self.cxt.fc_params)

    @timed_function("HybridSCUpdate")
    def update(self, pc):
        """Update by assembling into the KSP operator. No
        need to reconstruct symbolic objects.
        """
        self._assemble_S()
        self.S.force_evaluation()

    def apply(self, pc, x, y):
        """Solve the reduced system for the Lagrange multipliers.
        The system is assembled using operators constructed from
        the Slate expressions in the initialize method of this PC.
        Recovery of the scalar and flux fields are assembled cell-wise
        from Slate expressions describing the local problem.
        """
        with self.residual.dat.vec_wo as v:
            x.copy(v)

        with timed_region("HybridSCRHS"):
            # Now assemble residual for the reduced problem
            self._assemble_Srhs_thunk()

        with timed_region("HybridSCSolve"):
            # Solve the system for the Lagrange multipliers
            with self.r_lambda_thunk.dat.vec_ro as b:
                if self.trace_ksp.getInitialGuessNonzero():
                    acc = self.solution.split()[2].dat.vec
                else:
                    acc = self.solution.split()[2].dat.vec_wo
                with acc as x_trace:
                    self.trace_ksp.solve(b, x_trace)

        with timed_region("HybridSCReconstruct"):
            # Recover u_h and q_h
            self._assemble_u()
            self._assemble_q()

        with self.solution.dat.vec_ro as w:
            w.copy(y)

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner."""
        raise NotImplementedError("Transpose application is not implemented.")
