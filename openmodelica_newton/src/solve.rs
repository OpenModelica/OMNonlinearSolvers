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

//! The Newton-Raphson algorithm, its workspace and its results.

use crate::line_search::{LineSearchError, Start, line_search};
use crate::linear_solver::{LinearSolver, Singular, TotalPivot};
use crate::settings::Settings;
use crate::tracker::{Decision, Step, Tracker};
use faer::{ColRef, Mat};
use log::trace;
use openmodelica_nls_model::scaling::{self, scaled_squared_error, squared_error};
use openmodelica_nls_model::{EvalError, ModelFunctions};
use openmodelica_total_pivot_solver::SolveStatus;

/// Buffers of the Newton-Raphson algorithm.
///
/// Allocated once and reused by every call of [`solve`], so the iterations
/// don't allocate. Sized for a system with `num_residuals` residuals and
/// `num_unknowns` unknowns.
pub struct NewtonWorkspace {
    /// Residual of the latest evaluated point, `F(x)` or `F(x1)`
    f: Box<[f64]>,
    /// Linear system `[J | f]` of the Newton step, `num_residuals` rows and
    /// `num_unknowns + 1` columns. The total pivot solver solves it as
    /// `J·dy + f = 0`, so the last column holds `f` itself.
    jacobian: Mat<f64>,
    /// Newton step of the scaled system, solution of `J(x)·diag(s)·y = −f`
    y: Box<[f64]>,
    /// Newton step, solution of `J(x)·dy = −f`
    dy: Box<[f64]>,
    /// Trial point `x + λ·dy`, candidate for next iteration
    x1: Box<[f64]>,

    /// Variable scaling, one factor per unknown, see
    /// [`openmodelica_nls_model::scaling::variables`]
    x_scaling: Box<[f64]>,
    /// Solves the first step, see [`prepare_step`]
    total_pivot: TotalPivot,
    /// Residual scaling, one factor per residual, see
    /// [`openmodelica_nls_model::scaling::residuals`]
    res_scaling: Box<[f64]>,
    /// The maxima of the row scaling, see [`openmodelica_nls_model::scaling::rows`]
    row_maxima: Box<[f64]>,

    /// Iterations of the latest [`solve`]
    iterations: usize,
}

impl NewtonWorkspace {
    /// Allocates zero-initialized buffers for a system with `num_unknowns`
    /// unknowns and `num_residuals` residuals.
    pub fn new(num_unknowns: usize, num_residuals: usize) -> NewtonWorkspace {
        NewtonWorkspace {
            f: vec![0.0; num_residuals].into_boxed_slice(),
            jacobian: Mat::zeros(num_residuals, num_unknowns + 1),
            y: vec![0.0; num_unknowns].into_boxed_slice(),
            dy: vec![0.0; num_unknowns].into_boxed_slice(),
            x1: vec![0.0; num_unknowns].into_boxed_slice(),

            x_scaling: vec![0.0; num_unknowns].into_boxed_slice(),
            total_pivot: TotalPivot::new(num_unknowns),
            res_scaling: vec![0.0; num_residuals].into_boxed_slice(),
            row_maxima: vec![0.0; num_residuals].into_boxed_slice(),
            iterations: 0,
        }
    }

    /// The residual scaling of the latest Jacobian of [`solve`].
    pub fn res_scaling(&self) -> &[f64] {
        &self.res_scaling
    }

    /// The number of iterations of the latest [`solve`], also when it failed.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The residual `F(x)` of the latest evaluation, see [`evaluate`].
    pub fn f(&self) -> &[f64] {
        &self.f
    }

    /// Set the variable scaling for [`prepare_step`] and [`solve_prepared`],
    /// one factor per unknown. [`solve`] sets it itself.
    ///
    /// # Panics
    ///
    /// Panics if `x_scaling` doesn't have one factor per unknown.
    pub fn set_x_scaling(&mut self, x_scaling: &[f64]) {
        self.x_scaling.copy_from_slice(x_scaling);
    }
}

/// How a successful [`solve`] ended.
#[derive(Debug, PartialEq)]
pub enum NewtonStatus {
    /// Converged: the residual and the last step are below their tolerances
    Solved,
    /// The steps became tiny while the residual is small, but not below its
    /// tolerance. The point is accepted as a solution with less accuracy.
    SolvedLessAccurate,
}

/// Why [`solve`] found no solution.
#[derive(Debug)]
pub enum NewtonError {
    /// The maximum number of iterations of the [`Settings`] is reached
    MaxIterations,
    /// The steps became tiny while the residual is still large, for example
    /// in a local minimum of `‖F‖`
    Stuck,
    /// The iteration moves away from any solution: the error grew strongly,
    /// cycled or stagnated for many iterations
    Diverging,
    /// The Jacobian is singular and the linear system has no solution
    Singular,
    /// The model can't be evaluated: an assert or, for a casual tearing set,
    /// a violated constraint
    ModelError(EvalError),
}

/// A model that fails to evaluate is a [`NewtonError::ModelError`]
impl From<EvalError> for NewtonError {
    fn from(error: EvalError) -> Self {
        NewtonError::ModelError(error)
    }
}

/// A linear system that can't be solved is singular
impl From<Singular> for NewtonError {
    fn from(_: Singular) -> Self {
        NewtonError::Singular
    }
}

/// A model that asserted at every step factor of the line search fails like a
/// single assert
impl From<LineSearchError> for NewtonError {
    fn from(error: LineSearchError) -> Self {
        match error {
            LineSearchError::BackedOffBelowLambdaMin => NewtonError::ModelError(EvalError::Assert),
            LineSearchError::ModelError(error) => NewtonError::ModelError(error),
        }
    }
}

/// Solve `F(x) = 0` for the `model` with a damped Newton-Raphson method.
///
/// `x` holds the start point on entry. On success it holds the solution, on
/// failure the last accepted iterate. `settings` sets the tolerances and the
/// maximum number of iterations, `work` provides the buffers, sized for the
/// model.
///
/// One iteration:
///
/// ```text
/// 1. dy := solution of J(x)·dy = −F(x)   the full Newton step
/// 2. x1 := x + λ·dy                      trial point, λ ∈ (0, 1] from the line search
/// 3. stop?                               see the tracker's decision
/// 4. x  := x1, refresh J(x)              the trial point becomes the new iterate
/// ```
///
/// The unknowns are scaled with their nominal values or magnitudes at the
/// start point, the residuals with the row sums of the Jacobian. It is
/// [`evaluate`], [`prepare_step`] and [`solve_prepared`] in a row. From
/// `newtonAlgorithm` of the C runtime, together with the setup in
/// `solveHomotopy` before it. All linear systems are solved with
/// [`TotalPivot`], whose buffers for the later steps are allocated per call,
/// [`solve_prepared`] takes a linear solver instead.
///
/// # Errors
///
/// Returns a [`NewtonError`] if the iteration doesn't converge, the Jacobian
/// is singular or the model can't be evaluated.
///
/// # Panics
///
/// Panics if `x` or `work` don't match the size of the model.
pub fn solve<M: ModelFunctions>(
    model: &mut M,
    x: &mut [f64],
    settings: &Settings,
    work: &mut NewtonWorkspace,
) -> Result<NewtonStatus, NewtonError> {
    assert_eq!(
        x.len(),
        model.info().num_unknowns,
        "Start point has wrong length"
    );
    scaling::variables(&model.info().nominal, x, &mut work.x_scaling);
    work.iterations = 0;
    evaluate(model, x, work)?;
    prepare_step(model, x, true, work)?;
    let mut total_pivot = TotalPivot::new(x.len());
    solve_prepared(model, x, settings, &mut total_pivot, work)
}

/// Evaluate `F(x)` into `work`, see [`NewtonWorkspace::f`].
///
/// # Errors
///
/// Returns the [`EvalError`] of the model if `F` can't be evaluated at `x`.
pub fn evaluate<M: ModelFunctions>(
    model: &mut M,
    x: &[f64],
    work: &mut NewtonWorkspace,
) -> Result<(), EvalError> {
    model.eval(x, &mut work.f)
}

/// Prepare the first Newton step at `x`, with `F(x)` from [`evaluate`] in
/// `work`.
///
/// Evaluates the Jacobian with its columns scaled by the variable scaling of
/// `work`, computes the residual scaling and solves the linear system for the
/// step with [`TotalPivot`]. With `make_finite`, non-finite factors of the
/// residual scaling become 1, as `solveHomotopy` does at the start point.
/// Like there, a vanishing determinant of this system is ignored.
///
/// # Errors
///
/// Returns [`NewtonError::ModelError`] if the Jacobian can't be evaluated and
/// [`NewtonError::Singular`] if the linear system can't be solved.
pub fn prepare_step<M: ModelFunctions>(
    model: &mut M,
    x: &[f64],
    make_finite: bool,
    work: &mut NewtonWorkspace,
) -> Result<(), NewtonError> {
    refresh_jacobian(model, x, work)?;
    if make_finite {
        scaling::make_finite(&mut work.res_scaling);
    }
    solve_step(
        &mut work.total_pivot,
        model.info().casual_tearing_set,
        &mut work.jacobian,
        &work.x_scaling,
        &mut work.y,
        &mut work.dy,
    )?;
    Ok(())
}

/// Solve `F(x) = 0` from `x`, starting with the step [`prepare_step`] left in
/// `work` for `x`.
///
/// Like [`solve`] after its setup, with the linear systems of the later
/// iterations solved by `linear_solver`. The caller sets the variable
/// scaling, see [`NewtonWorkspace::set_x_scaling`].
///
/// # Errors
///
/// Like [`solve`].
///
/// # Panics
///
/// Panics if `x` or `work` don't match the size of the model.
pub fn solve_prepared<M: ModelFunctions>(
    model: &mut M,
    x: &mut [f64],
    settings: &Settings,
    linear_solver: &mut impl LinearSolver,
    work: &mut NewtonWorkspace,
) -> Result<NewtonStatus, NewtonError> {
    let n = model.info().num_unknowns;
    let casual_tearing_set = model.info().casual_tearing_set;
    assert_eq!(x.len(), n, "Start point has wrong length");
    assert_eq!(work.x1.len(), n, "Workspace has wrong size");

    work.iterations = 0;
    let mut err = squared_error(&work.f);
    let mut err_scaled = scaled_squared_error(&work.f, &work.res_scaling);
    let mut tracker = Tracker::new(settings, err, err_scaled);
    trace!(target: "NLS", "Newton iteration started, error {err:e}, scaled {err_scaled:e}");

    // The determinant of the step of `prepare_step` is ignored
    let mut determinant_zero = false;
    for iteration in 1.. {
        work.iterations = iteration;

        let start = Start {
            err,
            err_scaled,
            g: -2.0 * err,
            g_scaled: -2.0 * err_scaled,
        };
        let mut eval = |x: &[f64], f: &mut [f64]| model.eval(x, f);
        let step_factor = line_search(
            &mut eval,
            x,
            &work.dy,
            &work.res_scaling,
            &start,
            &mut work.x1,
            &mut work.f,
        )?;

        err = squared_error(&work.f);
        err_scaled = scaled_squared_error(&work.f, &work.res_scaling);
        let step = Step {
            err,
            err_scaled,
            dy: ColRef::from_slice(&work.dy).squared_norm_l2(),
            dy_scaled: scaled_squared_error(&work.dy, &work.x_scaling),
            step_factor,
        };
        tracker.update(step);
        trace!(
            target: "NLS",
            "Newton iteration {iteration}: error {err:e}, scaled {err_scaled:e}, ‖dy‖² {:e}, λ {step_factor}",
            step.dy
        );

        match tracker.decide(determinant_zero) {
            Decision::Continue => {}
            Decision::Converged { status, take_step } => {
                if take_step {
                    x.copy_from_slice(&work.x1);
                }
                trace!(target: "NLS", "Newton iteration {iteration}: {status:?}");
                return Ok(status);
            }
            Decision::Failed(error) => {
                trace!(target: "NLS", "Newton iteration {iteration}: {error:?}");
                return Err(error);
            }
        }

        x.copy_from_slice(&work.x1);
        refresh_jacobian(model, x, work)?;
        determinant_zero = solve_step(
            linear_solver,
            casual_tearing_set,
            &mut work.jacobian,
            &work.x_scaling,
            &mut work.y,
            &mut work.dy,
        )?;
    }
    unreachable!("the tracker stops after the maximum number of iterations")
}

/// Solve the scaled linear system `[J | f]` of `jacobian` with
/// `linear_solver` for the scaled step `y`, and scale it back to the
/// unknowns into `dy`. Returns whether the determinant of a casual tearing
/// set vanishes.
fn solve_step(
    linear_solver: &mut impl LinearSolver,
    casual_tearing_set: bool,
    jacobian: &mut Mat<f64>,
    x_scaling: &[f64],
    y: &mut [f64],
    dy: &mut [f64],
) -> Result<bool, Singular> {
    let status = linear_solver.solve(jacobian, casual_tearing_set, y)?;
    scaling::unscale(y, x_scaling, dy);
    Ok(matches!(status, SolveStatus::SolvedDeterminantZero))
}

/// Set up the linear system `[J | f]` of the next Newton step at `x`.
///
/// Expects `F(x)` in `work.f`, as the line search leaves it. Evaluates the
/// Jacobian `J(x)`, with its columns scaled by `work.x_scaling`, into
/// `work.jacobian`, with `f` as its last column, computes `work.res_scaling`
/// from the column-scaled `J` and finally scales the rows. From the end of
/// the loop in `newtonAlgorithm`.
///
/// # Errors
///
/// Returns the [`EvalError`] of the model if `J` can't be evaluated at `x`.
/// `work.jacobian` is unspecified then.
fn refresh_jacobian<M: ModelFunctions>(
    model: &mut M,
    x: &[f64],
    work: &mut NewtonWorkspace,
) -> Result<(), EvalError> {
    let jacobian = work.jacobian.subcols_mut(0, work.jacobian.nrows());

    model.derivative(x, &work.f, &work.x_scaling, jacobian)?;
    work.jacobian
        .col_mut(work.jacobian.nrows())
        .copy_from(ColRef::from_slice(&work.f));

    scaling::residuals(work.jacobian.as_ref(), &mut work.res_scaling);
    scaling::rows(work.jacobian.as_mut(), &mut work.row_maxima);
    Ok(())
}

#[cfg(test)]
mod newton_workspace_tests {
    use super::*;

    #[test]
    /// `NewtonWorkspace::new` allocates zero-initialized buffers of the
    /// documented sizes. Different numbers of unknowns and residuals catch
    /// swapped dimensions.
    fn newton_workspace_new_test() {
        let (num_unknowns, num_residuals) = (3, 2);

        let work = NewtonWorkspace::new(num_unknowns, num_residuals);

        assert_eq!(*work.f, [0.0; 2], "f: one entry per residual");
        assert_eq!(*work.y, [0.0; 3], "y: one entry per unknown");
        assert_eq!(*work.dy, [0.0; 3], "dy: one entry per unknown");
        assert_eq!(*work.x1, [0.0; 3], "x1: one entry per unknown");
        assert_eq!(
            *work.x_scaling, [0.0; 3],
            "x_scaling: one entry per unknown"
        );
        assert_eq!(
            *work.res_scaling, [0.0; 2],
            "res_scaling: one entry per residual"
        );

        // [J | f]: one row per residual, one column per unknown and one for f
        assert_eq!(
            (work.jacobian.nrows(), work.jacobian.ncols()),
            (num_residuals, num_unknowns + 1),
            "jacobian: shape of [J | f]"
        );
        assert_eq!(work.jacobian, Mat::<f64>::zeros(2, 4));
    }
}
