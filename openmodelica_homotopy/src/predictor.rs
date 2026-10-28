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

//! The predictor step along the tangent of the homotopy path.

use crate::settings::Settings;
use openmodelica_nls_model::{EvalError, ModelFunctions};

/// A tangent with a `λ` component below this doesn't limit the step size
const MIN_TANGENT_LAMBDA: f64 = 1e-8;

/// Why the predictor found no point.
#[derive(Debug)]
pub(crate) enum PredictorError {
    /// The homotopy asserted for every step size down to `tau_min`, `tau` is
    /// the last step size divided once more, as the C runtime reports it
    BackedOffBelowTauMin { tau: f64 },
    /// The homotopy violated a constraint, which doesn't back off
    ModelError(EvalError),
}

/// Limit the step size `tau` so that the predictor from `λ` along a tangent
/// with the component `t_lambda` doesn't pass `λ = 1`.
///
/// A tangent almost orthogonal to `λ`, `|t_λ| ≤ 10⁻⁸`, leaves `tau` as it is.
/// From the predictor step of `homotopyAlgorithm`, after a new tangent.
pub(crate) fn limit_to_end(tau: f64, lambda: f64, t_lambda: f64) -> f64 {
    if t_lambda.abs() > MIN_TANGENT_LAMBDA {
        tau.min((1.0 - lambda) / t_lambda.abs())
    } else {
        tau
    }
}

/// Predict the next point `y1 = y0 + τ·t` along the tangent `t` and evaluate
/// `h = H(y1)`.
///
/// Starts with the step size `tau`. While the homotopy asserts at `y1`, `τ` is
/// divided by `tau_decreasing_factor_predictor` of the `settings`, as long as
/// it stays above `tau_min`. Returns the step size of `y1`. From the
/// predictor step of `homotopyAlgorithm`.
///
/// # Errors
///
/// * [`PredictorError::BackedOffBelowTauMin`] if the homotopy still asserts at
///   the smallest step size above `tau_min`
/// * [`PredictorError::ModelError`] at once for a violated constraint. The C
///   runtime never follows a path for a casual tearing set, so this doesn't
///   happen there.
///
/// `y1` and `h` are unspecified after an error.
///
/// # Panics
///
/// Panics if `y0`, `t` and `y1` don't have the same length.
pub(crate) fn predictor(
    homotopy: &mut impl ModelFunctions,
    y0: &[f64],
    t: &[f64],
    tau: f64,
    settings: &Settings,
    y1: &mut [f64],
    h: &mut [f64],
) -> Result<f64, PredictorError> {
    assert_eq!(y0.len(), t.len(), "Tangent has wrong length");
    assert_eq!(y0.len(), y1.len(), "Predictor point has wrong length");

    let mut tau = tau;
    loop {
        for ((y1, y0), t) in y1.iter_mut().zip(y0).zip(t) {
            *y1 = y0 + tau * t;
        }
        match homotopy.eval(y1, h) {
            Ok(()) => return Ok(tau),
            Err(EvalError::Assert) => tau /= settings.tau_decreasing_factor_predictor,
            Err(error @ EvalError::ConstraintViolated) => {
                return Err(PredictorError::ModelError(error));
            }
        }
        if tau <= settings.tau_min {
            return Err(PredictorError::BackedOffBelowTauMin { tau });
        }
    }
}

#[cfg(test)]
mod predictor_tests {
    use super::*;
    use openmodelica_nls_model::ClosureModel;

    #[test]
    /// From `λ = 0.5` a tangent with `t_λ = ±2` reaches `λ = 1` after `τ = 0.25`
    fn limit_to_end_test() {
        assert_eq!(limit_to_end(1.0, 0.5, 2.0), 0.25);
        assert_eq!(limit_to_end(1.0, 0.5, -2.0), 0.25, "only |t_λ| counts");
        assert_eq!(limit_to_end(0.2, 0.5, 2.0), 0.2, "a shorter step stays");
    }

    #[test]
    /// A tangent almost orthogonal to `λ` doesn't limit the step
    fn limit_to_end_orthogonal_test() {
        assert_eq!(limit_to_end(5.0, 0.5, 1e-9), 5.0);
    }

    /// Result, number of evaluations, `y1` and `h` of a predictor from
    /// `y0 = (0, 0)` along `t = (1, 0.5)` with `τ = 0.2`, on the linear
    /// homotopy `H = x − 2λ`. `fail(y1)` is the error at `y1`, or `None`.
    fn predict(
        fail: impl Fn(&[f64]) -> Option<EvalError>,
    ) -> (Result<f64, PredictorError>, usize, [f64; 2], [f64; 1]) {
        let mut calls = 0;
        let mut homotopy = ClosureModel::new(1, 2, |y: &[f64], h: &mut [f64]| {
            calls += 1;
            h[0] = y[0] - 2.0 * y[1];
            fail(y).map_or(Ok(()), Err)
        });
        let mut y1 = [0.0; 2];
        let mut h = [f64::NAN];

        let result = predictor(
            &mut homotopy,
            &[0.0, 0.0],
            &[1.0, 0.5],
            0.2,
            &Settings::new(1),
            &mut y1,
            &mut h,
        );
        (result, calls, y1, h)
    }

    #[test]
    /// The full step is taken if the homotopy can be evaluated there
    fn predictor_test() {
        let (result, calls, y1, h) = predict(|_| None);

        assert!(matches!(result, Ok(0.2)), "got {result:?}");
        assert_eq!(calls, 1);
        assert_eq!((y1, h), ([0.2, 0.1], [0.0]));
    }

    #[test]
    /// While the homotopy asserts, the step is halved: 0.2 and 0.1 pass
    /// `x = 0.06`, 0.05 doesn't
    fn predictor_back_off_test() {
        let (result, calls, y1, _) = predict(|y| (y[0] > 0.06).then_some(EvalError::Assert));

        assert!(matches!(result, Ok(0.05)), "got {result:?}");
        assert_eq!(calls, 3);
        assert_eq!(y1, [0.05, 0.025]);
    }

    #[test]
    /// A homotopy that always asserts is tried at `0.2 / 2ᵏ` while that is above
    /// `τ_min = 10⁻⁴`, for `k = 0…10`, then the predictor gives up
    fn predictor_below_tau_min_test() {
        let (result, calls, _, _) = predict(|_| Some(EvalError::Assert));

        assert!(
            matches!(result, Err(PredictorError::BackedOffBelowTauMin { tau }) if tau < 1e-4),
            "got {result:?}"
        );
        assert_eq!(calls, 11);
    }

    #[test]
    /// A violated constraint fails at once, without backing off
    fn predictor_constraint_test() {
        let (result, calls, _, _) = predict(|_| Some(EvalError::ConstraintViolated));

        assert!(
            matches!(
                result,
                Err(PredictorError::ModelError(EvalError::ConstraintViolated))
            ),
            "got {result:?}"
        );
        assert_eq!(calls, 1);
    }
}
