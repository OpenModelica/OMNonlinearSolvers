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

//! The functions a C host calls.
//!
//! Every function checks its pointers and catches panics, so neither a null
//! pointer nor a panic reaches the host as undefined behaviour.

use crate::model::{CFallback, HomModel, ModelBuffers};
use crate::observer::CObserver;
use crate::settings::HomSettings;
use openmodelica_homotopy::{Direction, HomotopyError, HomotopyRun, HomotopySolver, Solved};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// A solver for systems of one size, created by [`hom_solver_new`] and
/// opaque to C.
pub struct HomSolver {
    solver: HomotopySolver,
    size: usize,
    /// Buffers of the models, kept between solves
    buffers: ModelBuffers,
}

/// Statistics of the latest solve of a [`HomSolver`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HomStatistics {
    /// Iterations of all runs of Newton's method, also the failed ones
    pub newton_iterations: usize,
    /// Accepted steps of the homotopy path that led to the solution, 0 without
    /// a homotopy
    pub path_steps: usize,
    /// Jacobians evaluated by forward differences, in models without a
    /// Jacobian callback. The host counts the calls of its callback itself.
    pub forward_differences: usize,
    /// Time of the forward differences in seconds
    pub forward_differences_time: f64,
}

/// Whether and how a solve ended.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HomStatus {
    /// `x` holds the solution
    Solved,
    /// Newton's method and all homotopies failed, `x` is unchanged
    NotSolved,
    /// The model can't be evaluated at the start point or violates a
    /// constraint, `x` is unchanged
    ModelError,
    /// A pointer is `NULL` or the sizes of the model, the solver and `x`
    /// don't match
    InvalidArgument,
    /// The solver failed unexpectedly, a bug in the solver
    Panic,
}

/// How a solve found the solution.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HomSolvedBy {
    /// The start point already solves the system
    StartPoint,
    /// Newton's method from the start point
    Newton,
    /// Newton's method after the Newton homotopy towards increasing `λ`
    HomotopyNewtonPositive,
    /// Newton's method after the Newton homotopy towards decreasing `λ`
    HomotopyNewtonNegative,
    /// Newton's method after the fixpoint homotopy
    HomotopyFixpoint,
    /// The model's own homotopy towards increasing `λ`
    OwnHomotopyPositive,
    /// The model's own homotopy towards decreasing `λ`
    OwnHomotopyNegative,
    /// The fallback of the model, after Newton's method failed
    Fallback,
}

impl From<Solved> for HomSolvedBy {
    fn from(solved: Solved) -> Self {
        match solved {
            Solved::StartPoint => HomSolvedBy::StartPoint,
            Solved::Newton => HomSolvedBy::Newton,
            Solved::Homotopy(HomotopyRun::NewtonPositive) => HomSolvedBy::HomotopyNewtonPositive,
            Solved::Homotopy(HomotopyRun::NewtonNegative) => HomSolvedBy::HomotopyNewtonNegative,
            Solved::Homotopy(HomotopyRun::Fixpoint) => HomSolvedBy::HomotopyFixpoint,
            Solved::Fallback => HomSolvedBy::Fallback,
        }
    }
}

impl From<HomotopyError> for HomStatus {
    fn from(error: HomotopyError) -> Self {
        match error {
            HomotopyError::ModelError(_) => HomStatus::ModelError,
            HomotopyError::NotSolved => HomStatus::NotSolved,
        }
    }
}

/// Create a solver for systems with `size` residuals and unknowns.
///
/// `settings` may be `NULL` for the defaults, see [`HomSettings`]. Returns
/// `NULL` if `size` is 0 or a setting is invalid. Free the solver with
/// [`hom_solver_free`].
///
/// # Safety
///
/// `settings` must be `NULL` or point to a valid [`HomSettings`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hom_solver_new(
    size: usize,
    settings: *const HomSettings,
) -> *mut HomSolver {
    if size == 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: guaranteed by the caller
    let settings = unsafe { settings.as_ref() }.copied().unwrap_or_default();
    let created = catch_unwind(|| {
        let (newton, homotopy) = settings.to_settings(size);
        HomSolver {
            solver: HomotopySolver::new(size, newton, homotopy),
            size,
            buffers: ModelBuffers::new(size, size),
        }
    });
    match created {
        Ok(solver) => Box::into_raw(Box::new(solver)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a solver of [`hom_solver_new`]. `NULL` is ignored.
///
/// # Safety
///
/// `solver` must be `NULL` or come from [`hom_solver_new`] and not be freed
/// already.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hom_solver_free(solver: *mut HomSolver) {
    if !solver.is_null() {
        // SAFETY: guaranteed by the caller
        drop(unsafe { Box::from_raw(solver) });
    }
}

/// Solve the system `model` from the start point `x`, like `solveHomotopy`
/// of the OpenModelica C runtime: Newton's method, and if that fails Newton's
/// method after each of the homotopies.
///
/// `x` has `size` entries and holds the solution on success. `solved_by` may
/// be `NULL`, otherwise it receives how the solution was found.
///
/// # Safety
///
/// `solver` must come from [`hom_solver_new`], `model` must point to a valid
/// [`HomModel`], with a valid or `NULL` observer, and `x` to `size` values. The
/// callbacks of `model` and its observer are called from this thread only,
/// while the function runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hom_solve(
    solver: *mut HomSolver,
    model: *const HomModel,
    x: *mut f64,
    solved_by: *mut HomSolvedBy,
) -> HomStatus {
    // SAFETY: guaranteed by the caller
    let (Some(solver), Some(model)) = (unsafe { solver.as_mut() }, unsafe { model.as_ref() })
    else {
        return HomStatus::InvalidArgument;
    };
    let size = solver.size;
    if x.is_null() || model.num_residuals != size || model.num_unknowns != size {
        return HomStatus::InvalidArgument;
    }
    let model_observer = model.observer;
    let mut fallback = CFallback::new(model);
    // SAFETY: guaranteed by the caller
    let Some((mut model, mut linear_solver)) = (unsafe { solver.buffers.wrap(model) }) else {
        return HomStatus::InvalidArgument;
    };
    // SAFETY: guaranteed by the caller
    let x = unsafe { std::slice::from_raw_parts_mut(x, size) };
    // SAFETY: guaranteed by the caller
    let mut observer = unsafe { CObserver::new(model_observer) };

    let result = catch_unwind(AssertUnwindSafe(|| {
        solver.solver.solve_with(
            &mut model,
            &mut observer,
            &mut fallback,
            &mut linear_solver,
            x,
        )
    }));
    match result {
        Ok(Ok(solved)) => {
            // SAFETY: guaranteed by the caller
            if let Some(solved_by) = unsafe { solved_by.as_mut() } {
                *solved_by = solved.into();
            }
            HomStatus::Solved
        }
        Ok(Err(error)) => error.into(),
        Err(_) => HomStatus::Panic,
    }
}

/// Solve a system `model` that has its own homotopy, with `λ` as last of its
/// `size + 1` unknowns, from the start point `x`.
///
/// The path starts towards decreasing `λ` if `negative_start` is set,
/// otherwise towards increasing `λ`, and the opposite direction is tried if
/// it fails. `x` has `size` entries and holds the solution on success.
/// `solved_by` may be `NULL`, otherwise it receives the direction of the
/// path.
///
/// # Safety
///
/// Like [`hom_solve`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hom_solve_own_homotopy(
    solver: *mut HomSolver,
    model: *const HomModel,
    x: *mut f64,
    negative_start: bool,
    solved_by: *mut HomSolvedBy,
) -> HomStatus {
    // SAFETY: guaranteed by the caller
    let (Some(solver), Some(model)) = (unsafe { solver.as_mut() }, unsafe { model.as_ref() })
    else {
        return HomStatus::InvalidArgument;
    };
    let size = solver.size;
    if x.is_null() || model.num_residuals != size || model.num_unknowns != size + 1 {
        return HomStatus::InvalidArgument;
    }
    let model_observer = model.observer;
    // SAFETY: guaranteed by the caller
    let Some((mut model, _)) = (unsafe { solver.buffers.wrap(model) }) else {
        return HomStatus::InvalidArgument;
    };
    // SAFETY: guaranteed by the caller
    let x = unsafe { std::slice::from_raw_parts_mut(x, size) };
    let first = if negative_start {
        Direction::Negative
    } else {
        Direction::Positive
    };

    // SAFETY: guaranteed by the caller
    let mut observer = unsafe { CObserver::new(model_observer) };

    let result = catch_unwind(AssertUnwindSafe(|| {
        solver
            .solver
            .solve_own_homotopy_with(&mut model, first, &mut observer, x)
    }));
    match result {
        Ok(Ok(direction)) => {
            // SAFETY: guaranteed by the caller
            if let Some(solved_by) = unsafe { solved_by.as_mut() } {
                *solved_by = match direction {
                    Direction::Positive => HomSolvedBy::OwnHomotopyPositive,
                    Direction::Negative => HomSolvedBy::OwnHomotopyNegative,
                };
            }
            HomStatus::Solved
        }
        Ok(Err(error)) => error.into(),
        Err(_) => HomStatus::Panic,
    }
}

/// The statistics of the latest solve of `solver`, zeros for `NULL`.
///
/// # Safety
///
/// `solver` must be `NULL` or come from [`hom_solver_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hom_solver_statistics(solver: *const HomSolver) -> HomStatistics {
    // SAFETY: guaranteed by the caller
    let Some(solver) = (unsafe { solver.as_ref() }) else {
        return HomStatistics::default();
    };
    let statistics = solver.solver.statistics();
    let derivatives = solver.buffers.statistics();
    HomStatistics {
        newton_iterations: statistics.newton_iterations,
        path_steps: statistics.path_steps,
        forward_differences: derivatives.evaluations,
        forward_differences_time: derivatives.seconds,
    }
}
