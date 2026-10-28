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

//! The line search itself: back off from the full Newton step while the
//! model asserts, test the step with the Armijo condition and damp it with
//! the quadratic and cubic interpolation, as `newtonAlgorithm` in the C
//! runtime does.

use super::TrialPoint;
use super::cubic_interpolation::cubic_interpolation;
use super::quadratic_interpolation::quadratic_interpolation;

use openmodelica_nls_model::EvalError;
use openmodelica_nls_model::scaling::{scaled_squared_error, squared_error};

/// Smallest step factor the line search takes
const LAMBDA_MIN: f64 = 1e-4;

/// Evaluation of the residual `F(x)` along the search direction
///
/// Writes `F(x)` into the second slice, or returns the model's [`EvalError`].
/// Every closure with this signature implements it, for example
/// `|x, f| model.eval(x, f)`.
// A trait alias once `trait_alias` is stable
pub(crate) trait EvalFunction: FnMut(&[f64], &mut [f64]) -> Result<(), EvalError> {}
impl<T: FnMut(&[f64], &mut [f64]) -> Result<(), EvalError>> EvalFunction for T {}

/// Why the line search found no step
#[derive(Debug)]
pub(crate) enum LineSearchError {
    /// The model asserted for every step factor down to [`LAMBDA_MIN`]
    BackedOffBelowLambdaMin,
    /// The model failed at a damped step, where the line search doesn't back
    /// off
    ModelError(EvalError),
}

impl From<EvalError> for LineSearchError {
    fn from(value: EvalError) -> Self {
        LineSearchError::ModelError(value)
    }
}

/// Line search along `dy` from `x`: find a step factor `λ` that decreases
/// the squared error `φ(λ) = ‖F(x + λ·dy)‖²` enough.
///
/// 1. Back off from the full step while the model asserts, see
///    [`back_off_failed_eval`]. Keep that step `λ₁` unless it
///    [`needs_damping`].
/// 2. Otherwise evaluate at the [`quadratic_step`] `λ₂`, and keep it unless
///    the [`needs_cubic_step`] test fails.
/// 3. Otherwise evaluate at the [`cubic_step`] and keep it without testing it.
///
/// `start` holds `φ(0)` and `φ'(0)`, and `res_scaling` the residual scaling
/// of the scaled error. Returns the step factor taken. `x1` then holds the
/// trial point `x + λ·dy` and `f` its residual `F(x1)`.
///
/// # Errors
///
/// * [`LineSearchError::BackedOffBelowLambdaMin`] or
///   [`LineSearchError::ModelError`] from [`back_off_failed_eval`]
/// * [`LineSearchError::ModelError`] if the model fails at the quadratic or
///   cubic step, which aborts without backing off
///
/// After an error `x1` and `f` are unspecified. `x` is never changed.
pub(crate) fn line_search(
    eval: &mut impl EvalFunction,
    x: &[f64],
    dy: &[f64],
    res_scaling: &[f64],
    start: &Start,
    x1: &mut [f64],
    f: &mut [f64],
) -> Result<f64, LineSearchError> {
    let lambda1 = back_off_failed_eval(eval, x, dy, x1, f)?;
    let p1 = TrialPoint {
        lambda: lambda1,
        error: squared_error(f),
    };
    if !needs_damping(start, &p1, scaled_squared_error(f, res_scaling)) {
        return Ok(lambda1);
    }

    let lambda2 = quadratic_step(start, &p1);
    eval_at(eval, x, dy, lambda2, x1, f)?;
    let p2 = TrialPoint {
        lambda: lambda2,
        error: squared_error(f),
    };
    if !needs_cubic_step(start, &p1, &p2) {
        return Ok(lambda2);
    }

    let lambda = cubic_step(start.err, start.g, &p1, &p2);
    eval_at(eval, x, dy, lambda, x1, f)?;
    Ok(lambda)
}

/// Evaluate the trial point `x1 = x + λ·dy` and write `F(x1)` into `f`
fn eval_at(
    eval: &mut impl EvalFunction,
    x: &[f64],
    dy: &[f64],
    lambda: f64,
    x1: &mut [f64],
    f: &mut [f64],
) -> Result<(), EvalError> {
    for ((x1, x), dy) in x1.iter_mut().zip(x).zip(dy) {
        *x1 = x + lambda * dy;
    }
    eval(x1, f)
}

/// Factor by which `λ` shrinks after a failed evaluation.
const LAMBDA_SCALE_DOWN: f64 = 0.655;

/// Back off from the full step `x + dy` until the model can be evaluated.
///
/// Tries `x1 = x + λ·dy` starting at `λ = 1`. While `eval` reports an
/// [`EvalError::Assert`], `λ` is multiplied by [`LAMBDA_SCALE_DOWN`].
///
/// Returns the first `λ` at which `eval` succeeds. `x1` then holds the trial
/// point and `f` its residual `F(x1)`.
///
/// Unlike the C runtime, which evaluates once more at a `λ` below
/// [`LAMBDA_MIN`] and then gives up anyway, it stops before that evaluation.
///
/// # Errors
///
/// * [`LineSearchError::ModelError`] at once if `eval` reports a violated
///   constraint, without backing off
/// * [`LineSearchError::BackedOffBelowLambdaMin`] if `eval` still fails at
///   the smallest `λ` not below [`LAMBDA_MIN`]
///
/// After an error `x1` and `f` are unspecified. `x` is never changed.
fn back_off_failed_eval(
    eval: &mut impl EvalFunction,
    x: &[f64],
    dy: &[f64],
    x1: &mut [f64],
    f: &mut [f64],
) -> Result<f64, LineSearchError> {
    debug_assert_eq!(x1.len(), x.len());
    debug_assert_eq!(x1.len(), dy.len());

    let mut lambda = 1.0;
    while lambda >= LAMBDA_MIN {
        match eval_at(eval, x, dy, lambda, x1, f) {
            Ok(()) => return Ok(lambda),
            Err(EvalError::Assert) => lambda *= LAMBDA_SCALE_DOWN,
            Err(e @ EvalError::ConstraintViolated) => return Err(e.into()),
        }
    }
    Err(LineSearchError::BackedOffBelowLambdaMin)
}

/// Factor `α` of the sufficient decrease in the Armijo test
pub(crate) const ALPHA: f64 = 1e-1;

/// Whether the trial step `λ₁` fails the Armijo test: its error `err_1`
/// didn't decrease by at least the fraction [`ALPHA`] of the slope,
/// `err_1 > err_0 + α·λ₁·g_0`.
fn insufficient_decrease(err_0: f64, err_1: f64, lambda_1: f64, g_0: f64) -> bool {
    err_1 > err_0 + ALPHA * lambda_1 * g_0
}

/// The start of the line search at `λ = 0`: the squared error `φ(0)` of the
/// current iterate and the slope `φ'(0)`, unscaled and with the residual
/// scaling
#[derive(Debug, Clone, Copy)]
pub(crate) struct Start {
    pub(crate) err: f64,
    pub(crate) err_scaled: f64,
    pub(crate) g: f64,
    pub(crate) g_scaled: f64,
}

/// Errors at or below this are too close to the solution to damp the step
const DAMPING_THRESHOLD: f64 = 1e-12;

/// Whether the trial step `p1` is too long and must be damped.
///
/// That is the case if both its error and its scaled error `p1_err_scaled`
/// fail the Armijo test, and the current iterate isn't already within
/// [`DAMPING_THRESHOLD`] of a solution.
fn needs_damping(start: &Start, p1: &TrialPoint, p1_err_scaled: f64) -> bool {
    insufficient_decrease(start.err, p1.error, p1.lambda, start.g)
        && insufficient_decrease(start.err_scaled, p1_err_scaled, p1.lambda, start.g_scaled)
        && start.err > DAMPING_THRESHOLD
        && start.err_scaled > DAMPING_THRESHOLD
}

/// Step factor from the quadratic interpolation through the trial point `p1`,
/// which [`needs_damping`]. The result is at least [`LAMBDA_MIN`].
fn quadratic_step(start: &Start, p1: &TrialPoint) -> f64 {
    quadratic_interpolation(start.err, start.g, p1).max(LAMBDA_MIN)
}

/// Whether the quadratic step `p2` is still too long after damping `p1`.
///
/// Like the C runtime this tests the error of `p1`, not of `p2`, against the
/// Armijo bound of `λ₂`, and doesn't test the scaled error.
fn needs_cubic_step(start: &Start, p1: &TrialPoint, p2: &TrialPoint) -> bool {
    insufficient_decrease(start.err, p1.error, p2.lambda, start.g)
        && start.err > DAMPING_THRESHOLD
        && start.err_scaled > DAMPING_THRESHOLD
}

/// Step factor from the cubic interpolation through the trial points `p1` and
/// `p2`, where `p1` is the first trial point of the line search.
///
/// Like the C runtime it takes half the first step if the cubic has no local
/// minimum, which also covers a non-finite minimum. The result is at least
/// [`LAMBDA_MIN`].
fn cubic_step(err: f64, g: f64, p1: &TrialPoint, p2: &TrialPoint) -> f64 {
    cubic_interpolation(err, g, p1, p2)
        .filter(|lambda| lambda.is_finite())
        .unwrap_or(0.5 * p1.lambda)
        .max(LAMBDA_MIN)
}

#[cfg(test)]
mod line_search_tests {
    use super::*;

    #[test]
    /// `φ(λ) = 1 − 2λ` has neither a cubic nor a quadratic term, so the
    /// minimum of the fit is `+∞`. The step is half the first one instead.
    fn cubic_step_not_finite_test() {
        let err = 1.0;
        let g = -2.0 * err;
        let p1 = TrialPoint {
            lambda: 1.0,
            error: -1.0,
        };
        let p2 = TrialPoint {
            lambda: 0.5,
            error: 0.0,
        };
        assert_eq!(cubic_interpolation(err, g, &p1, &p2), Some(f64::INFINITY));

        assert_eq!(cubic_step(err, g, &p1, &p2), 0.5);
    }

    #[test]
    /// Without a finite minimum the cubic step halves `λ₁`, also below 1:
    /// trial points on the line `φ(λ) = 1 − 2λ` fit `a₂ = a₃ = 0`, the
    /// minimum of the parabola is `+∞`
    fn cubic_step_not_finite_damped_test() {
        let err = 1.0;
        let g = -2.0 * err;
        let p1 = TrialPoint {
            lambda: 0.5,
            error: 0.0,
        };
        let p2 = TrialPoint {
            lambda: 0.25,
            error: 0.5,
        };

        assert_eq!(cubic_step(err, g, &p1, &p2), 0.25);
    }

    /// Start with the error `err` and the Newton slope `g = −2·err`, the same
    /// scaled and unscaled
    fn start(err: f64) -> Start {
        Start {
            err,
            err_scaled: err,
            g: -2.0 * err,
            g_scaled: -2.0 * err,
        }
    }

    #[test]
    /// From `φ(0) = 1` the full step must reach `φ(1) ≤ 0.8`. With 0.9 both
    /// errors fail the Armijo test.
    fn needs_damping_test() {
        let p1 = TrialPoint {
            lambda: 1.0,
            error: 0.9,
        };
        assert!(needs_damping(&start(1.0), &p1, 0.9));
    }

    #[test]
    /// The step is kept if one of the two errors decreased enough
    fn needs_damping_one_error_decreased_test() {
        let p1 = TrialPoint {
            lambda: 1.0,
            error: 0.9,
        };
        assert!(
            !needs_damping(&start(1.0), &p1, 0.8),
            "scaled error decreased"
        );

        let p1 = TrialPoint {
            lambda: 1.0,
            error: 0.8,
        };
        assert!(!needs_damping(&start(1.0), &p1, 0.9), "error decreased");
    }

    #[test]
    /// The Armijo bound scales with `λ₁`: for `λ₁ = 0.5` it is `0.9`
    fn needs_damping_damped_step_test() {
        let p1 = TrialPoint {
            lambda: 0.5,
            error: 0.85,
        };
        assert!(!needs_damping(&start(1.0), &p1, 0.85));
    }

    #[test]
    /// Close to the solution the step is never damped, even if it doubles the
    /// error
    fn needs_damping_near_solution_test() {
        let err = DAMPING_THRESHOLD;
        let p1 = TrialPoint {
            lambda: 1.0,
            error: 2.0 * err,
        };
        assert!(!needs_damping(&start(err), &p1, 2.0 * err));
    }

    #[test]
    /// `φ(λ) = 1 − 2λ + 4λ²` has its minimum at `λ = 0.25`
    fn quadratic_step_test() {
        let p1 = TrialPoint {
            lambda: 1.0,
            error: 3.0,
        };
        assert_eq!(quadratic_step(&start(1.0), &p1), 0.25);
    }

    #[test]
    /// A short trial step with a huge error gives a minimum far below
    /// [`LAMBDA_MIN`], the step is clamped to it
    fn quadratic_step_lambda_min_test() {
        let p1 = TrialPoint {
            lambda: 1e-3,
            error: 1e6,
        };
        assert_eq!(quadratic_step(&start(1.0), &p1), LAMBDA_MIN);
    }

    #[test]
    /// The C runtime tests the error of `p1` against the bound of `λ₂`. From
    /// `φ(0) = 1` with `λ₂ = 0.5` that bound is `0.9`.
    fn needs_cubic_step_test() {
        let p1 = TrialPoint {
            lambda: 1.0,
            error: 0.95,
        };
        let p2 = TrialPoint {
            lambda: 0.5,
            error: 0.0,
        };
        assert!(
            needs_cubic_step(&start(1.0), &p1, &p2),
            "error of p2 is ignored"
        );

        let p1 = TrialPoint {
            lambda: 1.0,
            error: 0.85,
        };
        assert!(!needs_cubic_step(&start(1.0), &p1, &p2));
    }

    /// Result, number of evaluations, `x1` and `f` of a line search
    type Search = (Result<f64, LineSearchError>, usize, [f64; 1], [f64; 1]);

    /// Line search from `x = 0` along the Newton step `dy = 1` of the model
    /// `F(x) = h(x)` with `h(0) = 1` and `h'(0) = −1`, so `x1 = λ`.
    /// `fail(call)` is the error of the evaluation number `call`, counted from
    /// 1, or `None` if it succeeds.
    fn search(h: impl Fn(f64) -> f64, fail: impl Fn(usize) -> Option<EvalError>) -> Search {
        let mut x1 = [0.0];
        let mut f = [0.0];
        let mut calls = 0;
        let mut eval = |x: &[f64], f: &mut [f64]| {
            calls += 1;
            f[0] = h(x[0]);
            fail(calls).map_or(Ok(()), Err)
        };

        let result = line_search(
            &mut eval,
            &[0.0],
            &[1.0],
            &[1.0],
            &start(1.0),
            &mut x1,
            &mut f,
        );
        (result, calls, x1, f)
    }

    #[test]
    /// A linear model reaches the solution with the full step
    fn line_search_full_step_test() {
        let (result, calls, x1, f) = search(|x| 1.0 - x, |_| None);

        assert!(matches!(result, Ok(1.0)), "got {result:?}");
        assert_eq!(calls, 1);
        assert_eq!((x1, f), ([1.0], [0.0]));
    }

    #[test]
    /// `h(λ) = 1 − λ + 0.92·λ²`: the full step `φ(1) = 0.8464` fails the
    /// Armijo bound 0.8, the quadratic step `λ₂ = 1 / (1 + φ(1))` is kept
    /// because `φ(1)` is below its bound `1 − 0.2·λ₂`.
    fn line_search_quadratic_step_test() {
        let h = |x: f64| 1.0 - x + 0.92 * x * x;
        let (result, calls, x1, f) = search(h, |_| None);

        let lambda2 = 1.0 / (1.0 + 0.92 * 0.92);
        let lambda = result.unwrap();
        assert!((lambda - lambda2).abs() < 1e-15, "got {lambda}");
        assert_eq!(calls, 2);
        assert_eq!((x1, f), ([lambda], [h(lambda)]));
    }

    #[test]
    /// `h(λ) = 1 − λ + λ²`: `φ(1) = 1` fails both tests. The cubic through
    /// `φ(1) = 1` and `φ(0.5) = 0.5625` has `a₃ = −0.5`, `a₂ = 2.5` and its
    /// minimum at `λ = 2 / (2.5 + √3.25)`.
    fn line_search_cubic_step_test() {
        let h = |x: f64| 1.0 - x + x * x;
        let (result, calls, x1, f) = search(h, |_| None);

        let expected = 2.0 / (2.5 + 3.25_f64.sqrt());
        let lambda = result.unwrap();
        assert!((lambda - expected).abs() < 1e-15, "got {lambda}");
        assert_eq!(calls, 3);
        assert_eq!((x1, f), ([lambda], [h(lambda)]));
    }

    #[test]
    /// After backing off, the damping starts from the backed off step
    fn line_search_back_off_test() {
        let h = |x: f64| 1.0 - x + x * x;
        let (result, calls, _, _) = search(h, |call| (call == 1).then_some(EvalError::Assert));

        // φ(0.655) = 0.599 is below the Armijo bound 1 − 0.2·0.655 = 0.869
        assert!(
            matches!(result, Ok(l) if l == LAMBDA_SCALE_DOWN),
            "got {result:?}"
        );
        assert_eq!(calls, 2);
    }

    #[test]
    /// A model error at a damped step aborts without backing off
    fn line_search_model_error_test() {
        let h = |x: f64| 1.0 - x + x * x;
        let (result, calls, _, _) = search(h, |call| (call == 2).then_some(EvalError::Assert));

        assert!(
            matches!(result, Err(LineSearchError::ModelError(EvalError::Assert))),
            "got {result:?}"
        );
        assert_eq!(calls, 2);
    }

    /// Result, number of evaluations, `x1` and `f` of a back off
    type BackOff = (Result<f64, LineSearchError>, usize, [f64; 2], [f64; 2]);

    /// Back off from `x = [1, 2]` along `dy = [2, -4]` with the model
    /// `F(x) = x`. `fail(call)` is the error of the evaluation number `call`,
    /// counted from 1, or `None` if it succeeds.
    fn back_off(fail: impl Fn(usize) -> Option<EvalError>) -> BackOff {
        let x = [1.0, 2.0];
        let dy = [2.0, -4.0];
        let mut x1 = [0.0; 2];
        let mut f = [0.0; 2];
        let mut calls = 0;
        let mut eval = |x: &[f64], f: &mut [f64]| {
            calls += 1;
            f.copy_from_slice(x);
            fail(calls).map_or(Ok(()), Err)
        };

        let result = back_off_failed_eval(&mut eval, &x, &dy, &mut x1, &mut f);
        (result, calls, x1, f)
    }

    /// `λ` after `k` back offs, shrunk in the same order as the line search
    fn shrunk(k: usize) -> f64 {
        (0..k).fold(1.0, |lambda, _| lambda * LAMBDA_SCALE_DOWN)
    }

    #[test]
    /// The full step is accepted after one evaluation, `x1 = x + dy`
    fn back_off_full_step_test() {
        let (result, calls, x1, f) = back_off(|_| None);

        assert!(matches!(result, Ok(1.0)), "got {result:?}");
        assert_eq!(calls, 1);
        assert_eq!(x1, [3.0, -2.0]);
        assert_eq!(f, x1, "f must be F(x1)");
    }

    #[test]
    /// Three asserts shrink `λ` three times, the fourth evaluation succeeds
    fn back_off_asserts_test() {
        let (result, calls, x1, f) = back_off(|call| (call <= 3).then_some(EvalError::Assert));

        let lambda = shrunk(3);
        assert!(matches!(result, Ok(l) if l == lambda), "got {result:?}");
        assert_eq!(calls, 4);
        assert_eq!(x1, [1.0 + lambda * 2.0, 2.0 + lambda * -4.0]);
        assert_eq!(f, x1, "f must be F(x1)");
    }

    #[test]
    /// A success at the smallest `λ = 0.655²¹ ≥ λ_min` is accepted
    fn back_off_smallest_lambda_test() {
        let (result, calls, _, _) = back_off(|call| (call <= 21).then_some(EvalError::Assert));

        let lambda = shrunk(21);
        assert!(lambda >= LAMBDA_MIN && lambda * LAMBDA_SCALE_DOWN < LAMBDA_MIN);
        assert!(matches!(result, Ok(l) if l == lambda), "got {result:?}");
        assert_eq!(calls, 22);
    }

    #[test]
    /// A model that always asserts is tried at the 22 step factors from 1 to
    /// `0.655²¹`, then the line search gives up
    fn back_off_below_lambda_min_test() {
        let (result, calls, _, _) = back_off(|_| Some(EvalError::Assert));

        assert!(
            matches!(result, Err(LineSearchError::BackedOffBelowLambdaMin)),
            "got {result:?}"
        );
        assert_eq!(calls, 22);
    }

    #[test]
    /// A violated constraint aborts at once, without backing off
    fn back_off_constraint_test() {
        let (result, calls, _, _) = back_off(|_| Some(EvalError::ConstraintViolated));

        assert!(
            matches!(
                result,
                Err(LineSearchError::ModelError(EvalError::ConstraintViolated))
            ),
            "got {result:?}"
        );
        assert_eq!(calls, 1);
    }

    #[test]
    /// A violated constraint after some asserts still aborts at once
    fn back_off_constraint_after_asserts_test() {
        let (result, calls, _, _) = back_off(|call| match call {
            1 | 2 => Some(EvalError::Assert),
            _ => Some(EvalError::ConstraintViolated),
        });

        assert!(
            matches!(
                result,
                Err(LineSearchError::ModelError(EvalError::ConstraintViolated))
            ),
            "got {result:?}"
        );
        assert_eq!(calls, 3);
    }
}
