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

//! Events of the homotopy paths for a C host, see [`HomObserver`].

use openmodelica_homotopy::{Direction, PathError, PathObserver};
use std::ffi::{c_int, c_void};

/// Why a homotopy path failed, the argument of [`HomPathFailed`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HomPathFailure {
    /// The homotopy or its Jacobian can't be evaluated
    ModelError,
    /// The linear system of the tangent is singular
    Singular,
    /// The homotopy asserts at every predictor point down to the smallest step
    /// size, the value is the last step size divided once more
    PredictorBackedOff,
    /// The corrector neither reached the path nor moved
    CorrectorStuck,
    /// A step failed with the step size already at its minimum
    TauAtMinimum,
    /// A step was tried the maximum number of times, the value
    TooManyTries,
    /// The path went below `λ = −1`, the value is `λ`
    LambdaBelowMinusOne,
    /// The path didn't reach `λ = 1` within the maximum number of steps, the
    /// value
    TooManySteps,
}

/// A path starts from the point `y = (x, 0)` of `size` values towards
/// increasing `λ` for `direction` 1, decreasing for -1.
pub type HomPathStarted =
    Option<unsafe extern "C" fn(user: *mut c_void, direction: c_int, y: *const f64, size: usize)>;
/// A try of a step starts from the latest point `y = (x, λ)` of the path, also
/// before a try fails the path and again for every retry.
pub type HomStepStarted =
    Option<unsafe extern "C" fn(user: *mut c_void, y: *const f64, size: usize)>;
/// A step was accepted, `y = (x, λ)` is the new point of the path.
pub type HomStepAccepted =
    Option<unsafe extern "C" fn(user: *mut c_void, y: *const f64, size: usize)>;
/// The path reached `λ = 1` at `y` after `steps` accepted steps.
pub type HomPathFinished =
    Option<unsafe extern "C" fn(user: *mut c_void, y: *const f64, size: usize, steps: usize)>;
/// The path failed for `failure`, `value` is explained there or 0.
pub type HomPathFailed =
    Option<unsafe extern "C" fn(user: *mut c_void, failure: HomPathFailure, value: f64)>;

/// Callbacks for the events of the homotopy paths of a solve, for example
/// for logging. Every callback may be `NULL`, `λ` is the last of the `size`
/// values of a point. Like the other callbacks, they must return.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HomObserver {
    /// Passed to the callbacks as it is
    pub user: *mut c_void,
    /// See [`HomPathStarted`]
    pub path_started: HomPathStarted,
    /// See [`HomStepStarted`]
    pub step_started: HomStepStarted,
    /// See [`HomStepAccepted`]
    pub step_accepted: HomStepAccepted,
    /// See [`HomPathFinished`]
    pub path_finished: HomPathFinished,
    /// See [`HomPathFailed`]
    pub path_failed: HomPathFailed,
}

/// A [`HomObserver`] as an observer of the solver, or no observer for `NULL`.
pub(crate) struct CObserver<'o> {
    observer: Option<&'o HomObserver>,
}

impl<'o> CObserver<'o> {
    /// # Safety
    ///
    /// `observer` must be `NULL` or point to a valid [`HomObserver`] that stays
    /// valid while the solver runs.
    pub(crate) unsafe fn new(observer: *const HomObserver) -> Self {
        CObserver {
            // SAFETY: guaranteed by the caller
            observer: unsafe { observer.as_ref() },
        }
    }
}

impl PathObserver for CObserver<'_> {
    fn path_started(&mut self, direction: Direction, y: &[f64]) {
        if let Some(observer) = self.observer
            && let Some(callback) = observer.path_started
        {
            let direction = match direction {
                Direction::Positive => 1,
                Direction::Negative => -1,
            };
            // SAFETY: `y` has `y.len()` values
            unsafe { callback(observer.user, direction, y.as_ptr(), y.len()) };
        }
    }

    fn step_started(&mut self, y: &[f64]) {
        if let Some(observer) = self.observer
            && let Some(callback) = observer.step_started
        {
            // SAFETY: `y` has `y.len()` values
            unsafe { callback(observer.user, y.as_ptr(), y.len()) };
        }
    }

    fn step_accepted(&mut self, y: &[f64]) {
        if let Some(observer) = self.observer
            && let Some(callback) = observer.step_accepted
        {
            // SAFETY: `y` has `y.len()` values
            unsafe { callback(observer.user, y.as_ptr(), y.len()) };
        }
    }

    fn path_finished(&mut self, y: &[f64], steps: usize) {
        if let Some(observer) = self.observer
            && let Some(callback) = observer.path_finished
        {
            // SAFETY: `y` has `y.len()` values
            unsafe { callback(observer.user, y.as_ptr(), y.len(), steps) };
        }
    }

    fn path_failed(&mut self, error: &PathError) {
        if let Some(observer) = self.observer
            && let Some(callback) = observer.path_failed
        {
            let (failure, value) = match *error {
                PathError::ModelError(_) => (HomPathFailure::ModelError, 0.0),
                PathError::Singular => (HomPathFailure::Singular, 0.0),
                PathError::PredictorBackedOff { tau } => (HomPathFailure::PredictorBackedOff, tau),
                PathError::CorrectorStuck => (HomPathFailure::CorrectorStuck, 0.0),
                PathError::TauAtMinimum => (HomPathFailure::TauAtMinimum, 0.0),
                PathError::TooManyTries { tries } => (HomPathFailure::TooManyTries, tries as f64),
                PathError::LambdaBelowMinusOne { lambda } => {
                    (HomPathFailure::LambdaBelowMinusOne, lambda)
                }
                PathError::TooManySteps { steps } => (HomPathFailure::TooManySteps, steps as f64),
            };
            // SAFETY: the callback gets values only
            unsafe { callback(observer.user, failure, value) };
        }
    }
}
