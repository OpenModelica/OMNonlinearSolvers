/*
 * This file belongs to the OpenModelica Run-Time System
 *
 * Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC), c/o Linköpings
 * universitet, Department of Computer and Information Science, SE-58183 Linköping, Sweden. All rights
 * reserved.
 *
 * THIS PROGRAM IS PROVIDED UNDER THE TERMS OF THE BSD NEW LICENSE OR THE
 * AGPL VERSION 3 LICENSE OR THE OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.8. ANY
 * USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES RECIPIENT'S
 * ACCEPTANCE OF THE BSD NEW LICENSE OR THE OSMC PUBLIC LICENSE OR THE AGPL
 * VERSION 3, ACCORDING TO RECIPIENTS CHOICE.
 *
 * The OpenModelica software and the OSMC (Open Source Modelica Consortium) Public License
 * (OSMC-PL) are obtained from OSMC, either from the above address, from the URLs:
 * http://www.openmodelica.org or https://github.com/OpenModelica/ or
 * http://www.ida.liu.se/projects/OpenModelica, and in the OpenModelica distribution. GNU
 * AGPL version 3 is obtained from: https://www.gnu.org/licenses/licenses.html#GPL. The BSD NEW
 * License is obtained from: http://www.opensource.org/licenses/BSD-3-Clause.
 *
 * This program is distributed WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY
 * SET FORTH IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF
 * OSMC-PL.
 *
 */

//! A system described by C callbacks, as a model of the solvers.

use crate::observer::HomObserver;
use faer::{Mat, MatMut};
use openmodelica_homotopy::Fallback;
use openmodelica_newton::{LinearSolver, Singular, SolveStatus, TotalPivot};
use openmodelica_nls_model::{EvalError, ModelFunctions, SystemInfo, forward_differences};
use std::ffi::{c_int, c_void};
use std::time::Instant;

/// The callback evaluated successfully
pub const HOM_EVAL_OK: c_int = 0;
/// The model asserted, the solver tries another point
pub const HOM_EVAL_ASSERT: c_int = 1;
/// A casual tearing set violated a constraint, the solver gives up
pub const HOM_EVAL_CONSTRAINT_VIOLATED: c_int = 2;

/// Evaluate the residuals `f = F(x)`.
///
/// `x` has `num_unknowns` entries, `f` has `num_residuals`. Returns
/// [`HOM_EVAL_OK`], [`HOM_EVAL_ASSERT`] or [`HOM_EVAL_CONSTRAINT_VIOLATED`],
/// other values count as an assert. The callback must return: it must not
/// leave with `longjmp` or an exception, since that would cross the solver's
/// Rust stack frames.
pub type HomResidual =
    Option<unsafe extern "C" fn(user: *mut c_void, x: *const f64, f: *mut f64) -> c_int>;

/// Evaluate the Jacobian `jacobian = J(x)` with `f = F(x)`.
///
/// `jacobian` is a column-major `num_residuals×num_unknowns` matrix whose
/// columns start `leading_dimension` entries apart, the entry `(i, j)` is
/// `jacobian[i + j*leading_dimension]`. Returns a value like [`HomResidual`]
/// and must return like it.
pub type HomJacobian = Option<
    unsafe extern "C" fn(
        user: *mut c_void,
        x: *const f64,
        f: *const f64,
        jacobian: *mut f64,
        leading_dimension: usize,
    ) -> c_int,
>;

/// Solve the system from `x`, the start point of a failed Newton's method,
/// for example with hybrd of Minpack. Returns nonzero if `x` holds a solution
/// then. Called whenever Newton's method fails, except for a casual tearing
/// set, and before the next homotopy. Must return like [`HomResidual`].
pub type HomFallback = Option<unsafe extern "C" fn(user: *mut c_void, x: *mut f64) -> c_int>;

/// Called when the solver has checked its start point, evaluated the
/// residuals and the Jacobian there, before Newton's method and the
/// homotopies start. Not called if the start point solves the system. Must
/// return like [`HomResidual`].
pub type HomStartPointChecked = Option<unsafe extern "C" fn(user: *mut c_void)>;

/// Solve the linear system `matrix·y = rhs` of a step of Newton's method
/// after the first, for example with `dgesv` of LAPACK.
///
/// `matrix` is the scaled Jacobian, a column-major `n×n` matrix whose columns
/// start `leading_dimension` entries apart, like `lda` of LAPACK. It may be
/// overwritten. `rhs` holds the scaled residual and receives the solution
/// `y`. `determinant` receives the determinant of `matrix`, which decides
/// whether the system of a casual tearing set is singular. Returns 0 if
/// solved, nonzero if not. Must return like [`HomResidual`].
pub type HomLinearSolver = Option<
    unsafe extern "C" fn(
        user: *mut c_void,
        n: usize,
        matrix: *mut f64,
        leading_dimension: usize,
        rhs: *mut f64,
        determinant: *mut f64,
    ) -> c_int,
>;

/// A nonlinear system `F(x) = 0` of a C host.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HomModel {
    /// Passed to the callbacks as it is
    pub user: *mut c_void,
    /// Number of residuals
    pub num_residuals: usize,
    /// Number of unknowns, with `λ` as last one for a model with its own
    /// homotopy
    pub num_unknowns: usize,
    /// Nominal values of the unknowns, or `NULL` for 1
    pub nominal: *const f64,
    /// Lower bounds of the unknowns, or `NULL` for none. Not used by the
    /// solver, like in the C runtime.
    pub min: *const f64,
    /// Upper bounds of the unknowns, or `NULL` for none. The forward
    /// differences step backwards where they would pass one.
    pub max: *const f64,
    /// The system is the casual set of a dynamic tearing
    pub casual_tearing_set: bool,
    /// Evaluates `F(x)`, required
    pub residual: HomResidual,
    /// Evaluates `J(x)`, or `NULL` for forward differences
    pub jacobian: HomJacobian,
    /// Gets the events of the homotopy paths, or `NULL`
    pub observer: *const HomObserver,
    /// Solves the system when Newton's method fails, or `NULL`
    pub fallback: HomFallback,
    /// Solves the linear systems of Newton's method after its first step, or
    /// `NULL` for total pivoting, which solves the first step
    pub linear_solver: HomLinearSolver,
    /// Told when the start point is checked, or `NULL`
    pub start_point_checked: HomStartPointChecked,
}

/// The fallback of a [`HomModel`] as a fallback of the solver.
pub(crate) struct CFallback {
    callback: HomFallback,
    user: *mut c_void,
}

impl CFallback {
    pub(crate) fn new(model: &HomModel) -> Self {
        CFallback {
            callback: model.fallback,
            user: model.user,
        }
    }
}

impl Fallback for CFallback {
    fn solve<M: ModelFunctions>(&mut self, model: &mut M, x: &mut [f64]) -> bool {
        let Some(callback) = self.callback else {
            return false;
        };
        assert_eq!(x.len(), model.info().num_unknowns);
        // SAFETY: `x` has the size of the model
        unsafe { callback(self.user, x.as_mut_ptr()) != 0 }
    }
}

/// The forward differences of a [`CModel`] in the latest solve.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DerivativeStatistics {
    /// Jacobians by forward differences
    pub(crate) evaluations: usize,
    /// Their time in seconds
    pub(crate) seconds: f64,
}

/// The buffers of a [`CModel`] and its [`CLinearSolver`], kept by the solver
/// between calls, so that a solve doesn't allocate.
pub(crate) struct ModelBuffers {
    /// The facts of the latest model, filled from it for each solve
    info: SystemInfo,
    /// The perturbed point of the forward differences
    x_perturbed: Box<[f64]>,
    /// Solves without a linear solver callback
    total_pivot: TotalPivot,
    /// Of the latest solve
    statistics: DerivativeStatistics,
}

impl ModelBuffers {
    /// Allocates the buffers for models with `num_residuals` residuals and
    /// `num_unknowns` unknowns.
    pub(crate) fn new(num_residuals: usize, num_unknowns: usize) -> Self {
        ModelBuffers {
            info: SystemInfo::new(num_residuals, num_unknowns),
            x_perturbed: vec![0.0; num_unknowns].into(),
            total_pivot: TotalPivot::new(num_residuals),
            statistics: DerivativeStatistics::default(),
        }
    }

    /// The forward differences of the latest wrapped model.
    pub(crate) fn statistics(&self) -> DerivativeStatistics {
        self.statistics
    }

    /// Wraps `model` as a model of the solvers and its linear solver, or
    /// returns `None` if it has no residual callback or no unknowns.
    ///
    /// The buffers are only allocated again if the sizes of `model` differ
    /// from the latest one, for example when a solver switches to a model
    /// with its own homotopy.
    ///
    /// # Safety
    ///
    /// `nominal`, `min` and `max` of `model` must be `NULL` or point to
    /// `num_unknowns` values.
    pub(crate) unsafe fn wrap<'b>(
        &'b mut self,
        model: &'b HomModel,
    ) -> Option<(CModel<'b>, CLinearSolver<'b>)> {
        let residual = model.residual?;
        let (num_residuals, num_unknowns) = (model.num_residuals, model.num_unknowns);
        if num_unknowns == 0 {
            return None;
        }
        if (self.info.num_residuals, self.info.num_unknowns) != (num_residuals, num_unknowns) {
            *self = ModelBuffers::new(num_residuals, num_unknowns);
        }

        let fill = |values: &mut [f64], pointer: *const f64, default: f64| {
            if pointer.is_null() {
                values.fill(default);
            } else {
                // SAFETY: guaranteed by the caller
                values
                    .copy_from_slice(unsafe { std::slice::from_raw_parts(pointer, num_unknowns) });
            }
        };
        fill(&mut self.info.nominal, model.nominal, 1.0);
        fill(&mut self.info.min, model.min, f64::NEG_INFINITY);
        fill(&mut self.info.max, model.max, f64::INFINITY);
        self.info.casual_tearing_set = model.casual_tearing_set;
        self.statistics = DerivativeStatistics::default();

        let c_model = CModel {
            residual: CResidual {
                model,
                residual,
                info: &self.info,
            },
            x_perturbed: &mut self.x_perturbed,
            statistics: &mut self.statistics,
        };
        let linear_solver = CLinearSolver {
            callback: model.linear_solver,
            user: model.user,
            total_pivot: &mut self.total_pivot,
        };
        Some((c_model, linear_solver))
    }
}

/// The linear solver of a [`HomModel`] as a linear solver of Newton's method.
pub(crate) struct CLinearSolver<'b> {
    callback: HomLinearSolver,
    user: *mut c_void,
    /// Solves without a callback
    total_pivot: &'b mut TotalPivot,
}

impl LinearSolver for CLinearSolver<'_> {
    /// Solves with the callback like `linearSolverWrapper` with LAPACK:
    /// the callback solves `J·y = f`, `y` is negated afterwards. Without a
    /// callback with [`TotalPivot`].
    fn solve(
        &mut self,
        system: &mut Mat<f64>,
        casual_tearing_set: bool,
        y: &mut [f64],
    ) -> Result<SolveStatus, Singular> {
        let Some(callback) = self.callback else {
            return self.total_pivot.solve(system, casual_tearing_set, y);
        };
        let n = y.len();
        assert_eq!((system.nrows(), system.ncols()), (n, n + 1));
        assert_eq!(system.row_stride(), 1, "System isn't column-major");
        let leading_dimension =
            usize::try_from(system.col_stride()).expect("System has columns in order");

        for (y, f) in y.iter_mut().zip(system.col(n).iter()) {
            *y = *f;
        }

        let mut determinant = 0.0;
        // SAFETY: the callback gets the first `n` columns of `system`, which
        // it may overwrite, with their column stride, and `y` of `n` entries
        let status = unsafe {
            callback(
                self.user,
                n,
                system.as_ptr_mut(),
                leading_dimension,
                y.as_mut_ptr(),
                &mut determinant,
            )
        };
        if status != 0 {
            return Err(Singular);
        }
        // Like in `linearSolverWrapper`, this solution isn't negated
        if determinant.abs() < 1e-9 && casual_tearing_set {
            return Ok(SolveStatus::SolvedDeterminantZero);
        }
        for y in y.iter_mut() {
            *y = -*y;
        }
        Ok(SolveStatus::Solved)
    }
}

/// The residual callback of a [`HomModel`] with the facts of the model, the
/// model the forward differences of a [`CModel`] evaluate.
struct CResidual<'b> {
    model: &'b HomModel,
    residual: unsafe extern "C" fn(user: *mut c_void, x: *const f64, f: *mut f64) -> c_int,
    info: &'b SystemInfo,
}

impl ModelFunctions for CResidual<'_> {
    fn info(&self) -> &SystemInfo {
        self.info
    }

    fn eval(&mut self, x: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
        assert_eq!(x.len(), self.info.num_unknowns);
        assert_eq!(res.len(), self.info.num_residuals);
        // SAFETY: the slices have the sizes the callback expects
        let status = unsafe { (self.residual)(self.model.user, x.as_ptr(), res.as_mut_ptr()) };
        eval_result(status)
    }
}

/// A [`HomModel`] as a model of the solvers, see [`ModelBuffers::wrap`].
pub(crate) struct CModel<'b> {
    residual: CResidual<'b>,
    /// The perturbed point of the forward differences
    x_perturbed: &'b mut [f64],
    /// Counts and times the forward differences
    statistics: &'b mut DerivativeStatistics,
}

/// The result of a callback
fn eval_result(status: c_int) -> Result<(), EvalError> {
    match status {
        HOM_EVAL_OK => Ok(()),
        HOM_EVAL_CONSTRAINT_VIOLATED => Err(EvalError::ConstraintViolated),
        _ => Err(EvalError::Assert),
    }
}

impl ModelFunctions for CModel<'_> {
    fn info(&self) -> &SystemInfo {
        self.residual.info
    }

    fn eval(&mut self, x: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
        self.residual.eval(x, res)
    }

    fn start_point_checked(&mut self) {
        if let Some(callback) = self.residual.model.start_point_checked {
            // SAFETY: the callback gets the host's pointer only
            unsafe { callback(self.residual.model.user) };
        }
    }

    /// The Jacobian callback's `J(x)`, each column `j` multiplied by
    /// `x_scaling[j]` like in `getAnalyticalJacobianHomotopy`, or forward
    /// differences without a callback.
    fn derivative(
        &mut self,
        x: &[f64],
        fx: &[f64],
        x_scaling: &[f64],
        jacobian: MatMut<'_, f64>,
    ) -> Result<(), EvalError> {
        let Some(callback) = self.residual.model.jacobian else {
            let start = Instant::now();
            let result = forward_differences(
                &mut self.residual,
                x,
                fx,
                x_scaling,
                self.x_perturbed,
                jacobian,
            );
            self.statistics.evaluations += 1;
            self.statistics.seconds += start.elapsed().as_secs_f64();
            return result;
        };
        let info = self.residual.info;
        assert_eq!(x.len(), info.num_unknowns);
        assert_eq!(fx.len(), info.num_residuals);
        assert_eq!(x_scaling.len(), x.len(), "Scaling vector has wrong length");
        assert_eq!(
            (jacobian.nrows(), jacobian.ncols()),
            (fx.len(), x.len()),
            "Jacobian has wrong shape"
        );

        assert_eq!(jacobian.row_stride(), 1, "Jacobian isn't column-major");
        let leading_dimension =
            usize::try_from(jacobian.col_stride()).expect("Jacobian has columns in order");

        // SAFETY: the callback writes the shape of `jacobian` with its column
        // stride, into memory `jacobian` borrows exclusively. `x` and `fx`
        // have the sizes of the model.
        let status = unsafe {
            callback(
                self.residual.model.user,
                x.as_ptr(),
                fx.as_ptr(),
                jacobian.as_ptr_mut(),
                leading_dimension,
            )
        };
        eval_result(status)?;

        for (column, scaling) in jacobian.col_iter_mut().zip(x_scaling) {
            for entry in column.iter_mut() {
                *entry *= scaling;
            }
        }
        Ok(())
    }
}
