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

//! The strategy of `solveHomotopy` in the C runtime: which solver runs when.

use crate::path_follower::{Direction, NoObserver, PathObserver, follow_path};
use crate::settings::Settings;
use crate::{FixpointHomotopy, NewtonHomotopy};
use log::debug;
use openmodelica_newton::settings::Settings as NewtonSettings;
use openmodelica_newton::{LinearSolver, NewtonError, NewtonWorkspace, TotalPivot};
use openmodelica_nls_model::scaling::{self, scaled_squared_error, squared_error};
use openmodelica_nls_model::{EvalError, ModelFunctions};

/// A start point is already a solution if its squared error is below this
/// factor times `f_tol²`
const SOLVED_AT_START_FACTOR: f64 = 1e-4;
/// The start point moves by these factors of the variable scaling on the
/// first and second retry
const RETRY_SHIFTS: [f64; 2] = [0.01, 0.1];

/// A homotopy the strategy follows after Newton's method failed, in the order
/// of the runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HomotopyRun {
    /// The Newton homotopy, starting towards increasing `λ`
    NewtonPositive,
    /// The Newton homotopy, starting towards decreasing `λ`
    NewtonNegative,
    /// The fixpoint homotopy, starting towards increasing `λ`
    Fixpoint,
}

/// The runs in the order of the C runtime
const RUNS: [HomotopyRun; 3] = [
    HomotopyRun::NewtonPositive,
    HomotopyRun::NewtonNegative,
    HomotopyRun::Fixpoint,
];

/// How [`HomotopySolver::solve`] found the solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Solved {
    /// The start point already solves the system
    StartPoint,
    /// Newton's method from the start point
    Newton,
    /// Newton's method from the end of the path of a homotopy
    Homotopy(HomotopyRun),
    /// The [`Fallback`] solver, after Newton's method failed
    Fallback,
}

/// A solver the strategy tries whenever Newton's method fails, before the
/// next homotopy, like the C runtime tries hybrd of Minpack.
pub trait Fallback {
    /// Solve `model` from `x`, the start point of the failed Newton's method,
    /// and return whether it solved it. `x` holds the solution then, and is
    /// unspecified otherwise.
    fn solve<M: ModelFunctions>(&mut self, model: &mut M, x: &mut [f64]) -> bool;
}

/// A [`Fallback`] that never solves, so the homotopies follow Newton's method
/// directly
pub struct NoFallback;

impl Fallback for NoFallback {
    fn solve<M: ModelFunctions>(&mut self, _: &mut M, _: &mut [f64]) -> bool {
        false
    }
}

/// A closure is a [`Fallback`] that doesn't need the model
impl<F: FnMut(&mut [f64]) -> bool> Fallback for F {
    fn solve<M: ModelFunctions>(&mut self, _: &mut M, x: &mut [f64]) -> bool {
        self(x)
    }
}

/// Which solver solved after Newton's method was started
enum Solver {
    Newton,
    Fallback,
}

/// Why [`HomotopySolver`] found no solution.
#[derive(Debug)]
pub enum HomotopyError {
    /// The model can't be evaluated at the start point, also not at the
    /// moved start points, or it violates a constraint there
    ModelError(EvalError),
    /// Newton's method and all homotopy runs failed
    NotSolved,
}

/// Why the linear system at a point failed.
enum PointError {
    ModelError(EvalError),
    Singular,
}

impl From<EvalError> for PointError {
    fn from(error: EvalError) -> Self {
        PointError::ModelError(error)
    }
}

impl From<NewtonError> for PointError {
    fn from(error: NewtonError) -> Self {
        match error {
            NewtonError::ModelError(error) => PointError::ModelError(error),
            _ => PointError::Singular,
        }
    }
}

/// The start point after its checks.
enum Start {
    /// The start point solves the system
    Solved,
    /// Newton's method and the homotopies can start from it, with the first
    /// Newton step prepared
    Ready,
    /// Like `Ready`, but the first linear system is singular, so Newton's
    /// method can't start
    Singular,
}

/// Statistics of the latest solve of a [`HomotopySolver`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Statistics {
    /// Iterations of all runs of Newton's method, also the failed ones
    pub newton_iterations: usize,
    /// Accepted steps of the homotopy path that led to the solution, 0 without
    /// a homotopy
    pub path_steps: usize,
}

/// The damped Newton solver combined with homotopies, `solveHomotopy` of the
/// C runtime.
///
/// It keeps the buffers and the residual scaling of the latest successful
/// solve between calls, like the C runtime keeps them in its solver data.
pub struct HomotopySolver {
    newton_settings: NewtonSettings,
    settings: Settings,
    newton: NewtonWorkspace,
    /// The start point given by the caller
    x_start: Box<[f64]>,
    /// The start point of Newton's method and the homotopies, maybe moved
    x0: Box<[f64]>,
    /// The start point of the latest Newton's method, for the fallback
    x_newton: Box<[f64]>,
    /// `F(x₀)`
    fx0: Box<[f64]>,
    /// Variable scaling of the `n` unknowns and `λ`, Newton's method uses it
    /// too
    x_scaling: Box<[f64]>,
    /// Residual scaling of the latest successful solve, zero otherwise. Only
    /// the test of an already solved start point uses it.
    res_scaling: Box<[f64]>,
    statistics: Statistics,
}

impl HomotopySolver {
    /// Creates a solver for systems with `size` residuals and unknowns.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    pub fn new(size: usize, newton_settings: NewtonSettings, settings: Settings) -> Self {
        assert!(size > 0, "Invalid size {size}");
        HomotopySolver {
            newton_settings,
            settings,
            newton: NewtonWorkspace::new(size, size),
            x_start: vec![0.0; size].into(),
            x0: vec![0.0; size].into(),
            x_newton: vec![0.0; size].into(),
            fx0: vec![0.0; size].into(),
            x_scaling: vec![1.0; size + 1].into(),
            res_scaling: vec![0.0; size].into(),
            statistics: Statistics::default(),
        }
    }

    /// The statistics of the latest solve.
    pub fn statistics(&self) -> Statistics {
        self.statistics
    }

    /// Solve `F(x) = 0` for `model` from the start point `x`.
    ///
    /// 1. A start point that already solves the system is taken. Otherwise the
    ///    model must be evaluated there and its first linear system must be
    ///    solvable, else the start point is moved by `1%` and then `10%` of
    ///    the variable scaling.
    /// 2. Newton's method from the start point.
    /// 3. If that fails, the paths of the Newton homotopy in both directions
    ///    and of the fixpoint homotopy, each from the start point, until a
    ///    path ends at a point from which Newton's method converges.
    ///
    /// A casual tearing set gives up after the first failure. `x` holds the
    /// solution on success and is unchanged otherwise. From `solveHomotopy`,
    /// without the handling of mixed systems, which belongs to the C
    /// interface. All linear systems are solved with total pivoting.
    /// [`HomotopySolver::solve_with`] adds the hybrd fallback and the linear
    /// solver of the C runtime.
    ///
    /// # Errors
    ///
    /// Returns a [`HomotopyError`] if the model can't be evaluated at the start
    /// point or no solution is found.
    ///
    /// # Panics
    ///
    /// Panics if `model` or `x` don't have the size of the solver.
    pub fn solve<M: ModelFunctions>(
        &mut self,
        model: &mut M,
        x: &mut [f64],
    ) -> Result<Solved, HomotopyError> {
        let mut total_pivot = TotalPivot::new(x.len());
        self.solve_with(model, &mut NoObserver, &mut NoFallback, &mut total_pivot, x)
    }

    /// [`HomotopySolver::solve`], reporting the events of the homotopy paths
    /// to `observer`, trying `fallback` whenever Newton's method fails, except
    /// for a casual tearing set, and solving the linear systems of Newton's
    /// method after its first step with `linear_solver`.
    ///
    /// # Errors
    ///
    /// Like [`HomotopySolver::solve`].
    ///
    /// # Panics
    ///
    /// Like [`HomotopySolver::solve`].
    pub fn solve_with<M: ModelFunctions>(
        &mut self,
        model: &mut M,
        observer: &mut impl PathObserver,
        fallback: &mut impl Fallback,
        linear_solver: &mut impl LinearSolver,
        x: &mut [f64],
    ) -> Result<Solved, HomotopyError> {
        let n = self.x_start.len();
        assert_eq!(model.info().num_unknowns, n, "Model has wrong size");
        assert_eq!(model.info().num_residuals, n, "Model has wrong size");
        assert_eq!(x.len(), n, "Start point has wrong length");

        self.x_start.copy_from_slice(x);
        scaling::variables(&model.info().nominal, x, &mut self.x_scaling[..n]);
        self.x_scaling[n] = 1.0;
        self.newton.set_x_scaling(&self.x_scaling[..n]);
        self.statistics = Statistics::default();

        let result = self.solve_from_start(model, observer, fallback, linear_solver, x);
        match &result {
            Ok(solved) => debug!(target: "NLS", "Solved with {solved:?}"),
            Err(error) => {
                debug!(target: "NLS", "Not solved: {error:?}");
                x.copy_from_slice(&self.x_start);
                self.res_scaling.fill(0.0);
            }
        }
        result
    }

    /// Solve with a model that has its own homotopy, the `homotopy()`
    /// operator of Modelica, from the start point `x`.
    ///
    /// `model` has `λ` as last unknown. Its path is followed from `(x, 0)` in
    /// the `first` direction, and if that fails in the opposite one. The end
    /// of the path is the solution, there is no Newton's method before or
    /// after it. Returns the direction of the successful path. From
    /// `solveHomotopy` with `initHomotopy`.
    ///
    /// # Errors
    ///
    /// Returns [`HomotopyError::NotSolved`] if both paths fail. `x` is
    /// unchanged then.
    ///
    /// # Panics
    ///
    /// Panics if `model` doesn't have the size of the solver with `λ` as
    /// additional unknown, or `x` doesn't have the size of the solver.
    pub fn solve_own_homotopy<M: ModelFunctions>(
        &mut self,
        model: &mut M,
        first: Direction,
        x: &mut [f64],
    ) -> Result<Direction, HomotopyError> {
        self.solve_own_homotopy_with(model, first, &mut NoObserver, x)
    }

    /// [`HomotopySolver::solve_own_homotopy`], reporting the events of the
    /// paths to `observer`.
    ///
    /// # Errors
    ///
    /// Like [`HomotopySolver::solve_own_homotopy`].
    ///
    /// # Panics
    ///
    /// Like [`HomotopySolver::solve_own_homotopy`].
    pub fn solve_own_homotopy_with<M: ModelFunctions>(
        &mut self,
        model: &mut M,
        first: Direction,
        observer: &mut impl PathObserver,
        x: &mut [f64],
    ) -> Result<Direction, HomotopyError> {
        let n = self.x_start.len();
        assert_eq!(model.info().num_residuals, n, "Model has wrong size");
        assert_eq!(
            model.info().num_unknowns,
            n + 1,
            "Model needs λ as last unknown"
        );
        assert_eq!(x.len(), n, "Start point has wrong length");

        self.x_start.copy_from_slice(x);
        scaling::variables(&model.info().nominal[..n], x, &mut self.x_scaling[..n]);
        self.x_scaling[n] = 1.0;
        self.statistics = Statistics::default();

        for direction in [first, first.opposite()] {
            x.copy_from_slice(&self.x_start);
            let path = follow_path(
                model,
                &self.x_scaling,
                direction,
                self.newton_settings.f_tol(),
                &self.settings,
                observer,
                x,
            );
            match path {
                Ok(steps) => {
                    self.statistics.path_steps = steps;
                    return Ok(direction);
                }
                Err(error) => {
                    debug!(target: "NLS", "Homotopy in {direction:?} direction: {error:?}")
                }
            }
        }
        x.copy_from_slice(&self.x_start);
        self.res_scaling.fill(0.0);
        Err(HomotopyError::NotSolved)
    }

    /// The steps of [`HomotopySolver::solve`] after the setup.
    fn solve_from_start<M: ModelFunctions>(
        &mut self,
        model: &mut M,
        observer: &mut impl PathObserver,
        fallback: &mut impl Fallback,
        linear_solver: &mut impl LinearSolver,
        x: &mut [f64],
    ) -> Result<Solved, HomotopyError> {
        let casual_tearing_set = model.info().casual_tearing_set;
        let start = self.check_start(model)?;
        x.copy_from_slice(&self.x0);
        if let Start::Solved = start {
            return Ok(Solved::StartPoint);
        }
        model.start_point_checked();

        let prepared = matches!(start, Start::Ready);
        match self.newton_then_fallback(model, fallback, linear_solver, prepared, x) {
            Some(Solver::Newton) => return Ok(Solved::Newton),
            Some(Solver::Fallback) => return Ok(Solved::Fallback),
            None => {}
        }
        if casual_tearing_set {
            return Err(HomotopyError::NotSolved);
        }

        for run in RUNS {
            x.copy_from_slice(&self.x0);
            let steps = match self.follow(model, run, observer, x) {
                Ok(steps) => steps,
                Err(error) => {
                    debug!(target: "NLS", "Homotopy {run:?}: {error:?}");
                    continue;
                }
            };
            // The C runtime gives up if the end of a path can't start Newton's
            // method
            openmodelica_newton::evaluate(model, x, &mut self.newton)
                .map_err(|_| HomotopyError::NotSolved)?;
            openmodelica_newton::prepare_step(model, x, false, &mut self.newton)
                .map_err(|_| HomotopyError::NotSolved)?;
            match self.newton_then_fallback(model, fallback, linear_solver, true, x) {
                Some(Solver::Newton) => {
                    self.statistics.path_steps = steps;
                    return Ok(Solved::Homotopy(run));
                }
                Some(Solver::Fallback) => return Ok(Solved::Fallback),
                None => {}
            }
        }
        Err(HomotopyError::NotSolved)
    }

    /// Check the start point and move it while the model can't be evaluated
    /// there or its linear system is singular. Leaves the start point to use
    /// in `x0` and its residual in `fx0`.
    fn check_start<M: ModelFunctions>(&mut self, model: &mut M) -> Result<Start, HomotopyError> {
        let n = self.x_start.len();
        let casual_tearing_set = model.info().casual_tearing_set;
        let f_tol_squared = self.newton_settings.f_tol().powi(2);

        for attempt in 0..=RETRY_SHIFTS.len() {
            let last_attempt = attempt == RETRY_SHIFTS.len();
            self.x0.copy_from_slice(&self.x_start);
            if attempt > 0 {
                let shift = RETRY_SHIFTS[attempt - 1];
                for (i, (x0, scaling)) in self.x0.iter_mut().zip(&self.x_scaling).enumerate() {
                    *x0 += scaling * i as f64 / n as f64 * shift;
                }
            }

            let checked = openmodelica_newton::evaluate(model, &self.x0, &mut self.newton)
                .map_err(PointError::from)
                .and_then(|()| {
                    self.fx0.copy_from_slice(self.newton.f());
                    let solved = squared_error(&self.fx0) < f_tol_squared * SOLVED_AT_START_FACTOR
                        || scaled_squared_error(&self.fx0, &self.res_scaling)
                            < f_tol_squared * SOLVED_AT_START_FACTOR;
                    if solved {
                        return Ok(Start::Solved);
                    }
                    openmodelica_newton::prepare_step(model, &self.x0, true, &mut self.newton)?;
                    Ok(Start::Ready)
                });

            match checked {
                Ok(start) => return Ok(start),
                Err(PointError::ModelError(error @ EvalError::ConstraintViolated)) => {
                    return Err(HomotopyError::ModelError(error));
                }
                Err(PointError::ModelError(error)) if casual_tearing_set || last_attempt => {
                    return Err(HomotopyError::ModelError(error));
                }
                Err(PointError::Singular) if casual_tearing_set => {
                    return Err(HomotopyError::NotSolved);
                }
                // Like the C runtime, a singular system at the last moved
                // start point is left to the fallback and the homotopies
                Err(PointError::Singular) if last_attempt => return Ok(Start::Singular),
                Err(_) => {}
            }
        }
        unreachable!("the last attempt returns")
    }

    /// Newton's method from `x` with the step prepared at `x`, and if it fails
    /// or isn't `prepared`, `fallback` from the same start point, except for a
    /// casual tearing set. Returns the solver that solved, `x` holds the
    /// solution then.
    fn newton_then_fallback<M: ModelFunctions>(
        &mut self,
        model: &mut M,
        fallback: &mut impl Fallback,
        linear_solver: &mut impl LinearSolver,
        prepared: bool,
        x: &mut [f64],
    ) -> Option<Solver> {
        self.x_newton.copy_from_slice(x);
        if prepared && self.newton(model, linear_solver, x) {
            return Some(Solver::Newton);
        }
        if model.info().casual_tearing_set {
            return None;
        }
        x.copy_from_slice(&self.x_newton);
        if fallback.solve(model, x) {
            debug!(target: "NLS", "Solved by the fallback");
            return Some(Solver::Fallback);
        }
        None
    }

    /// Newton's method from `x`, with the step prepared at `x`. Keeps its
    /// residual scaling on success.
    fn newton<M: ModelFunctions>(
        &mut self,
        model: &mut M,
        linear_solver: &mut impl LinearSolver,
        x: &mut [f64],
    ) -> bool {
        let result = openmodelica_newton::solve_prepared(
            model,
            x,
            &self.newton_settings,
            linear_solver,
            &mut self.newton,
        );
        self.statistics.newton_iterations += self.newton.iterations();
        match result {
            Ok(_) => {
                self.res_scaling.copy_from_slice(self.newton.res_scaling());
                true
            }
            Err(error) => {
                debug!(target: "NLS", "Newton's method: {error:?}");
                false
            }
        }
    }

    /// Follow the path of the homotopy of `run` from `x₀`, writing its end
    /// into `x`.
    fn follow<M: ModelFunctions>(
        &mut self,
        model: &mut M,
        run: HomotopyRun,
        observer: &mut impl PathObserver,
        x: &mut [f64],
    ) -> Result<usize, crate::PathError> {
        let f_tol = self.newton_settings.f_tol();
        match run {
            HomotopyRun::NewtonPositive | HomotopyRun::NewtonNegative => {
                let direction = if run == HomotopyRun::NewtonPositive {
                    Direction::Positive
                } else {
                    Direction::Negative
                };
                let mut homotopy = NewtonHomotopy::new(model, &self.fx0);
                follow_path(
                    &mut homotopy,
                    &self.x_scaling,
                    direction,
                    f_tol,
                    &self.settings,
                    observer,
                    x,
                )
            }
            HomotopyRun::Fixpoint => {
                let mut homotopy = FixpointHomotopy::new(model, &self.x0);
                follow_path(
                    &mut homotopy,
                    &self.x_scaling,
                    Direction::Positive,
                    f_tol,
                    &self.settings,
                    observer,
                    x,
                )
            }
        }
    }
}
