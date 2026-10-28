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

//! Quadratic interpolation of the line search: the first damping step of
//! `newtonAlgorithm` in the C runtime, when the full or backed off step failed
//! the Armijo test.

use super::TrialPoint;

/// Quadratic interpolation on `φ(λ) = ‖F(x + λ·dy)‖²`
///
/// Fits the parabola `φ(λ) ≈ a₂λ² + g·λ + err` through the trial point `p1`
/// and returns the step factor at its minimum.
///
/// * `err`: `φ(0)`, the squared error at the current iterate
/// * `g`: `φ'(0)`, the slope at the current iterate
/// * `p1`: a trial point with positive `lambda` above the tangent,
///   `φ(λ₁) > err + λ₁·g`, so that the parabola opens upwards
///
/// A trial point that failed the Armijo test is above the tangent, and the
/// result is less than `λ₁ / (2·(1 − α))`.
pub(crate) fn quadratic_interpolation(err: f64, g: f64, p1: &TrialPoint) -> f64 {
    let denominator = 2.0 * (p1.error - err - p1.lambda * g);
    debug_assert!(
        p1.lambda > 0.0 && denominator > 0.0,
        "Trial point must have positive lambda and lie above the tangent, got lambda {} and error {}",
        p1.lambda,
        p1.error
    );

    -p1.lambda * p1.lambda * g / denominator
}

#[cfg(test)]
mod quadratic_interpolation_tests {
    use super::super::search::ALPHA;
    use super::*;

    #[test]
    /// `φ(λ) = 1 − 2λ + 4λ²` is a parabola with its minimum at `λ = 0.25`
    fn parabola_test() {
        let p1 = TrialPoint {
            lambda: 1.0,
            error: 3.0,
        };
        assert_eq!(quadratic_interpolation(1.0, -2.0, &p1), 0.25);
    }

    #[test]
    /// For every trial point that fails the Armijo test the step shrinks to
    /// less than `λ₁ / (2·(1 − α))`
    fn bounded_step_test() {
        let err = 1.0;
        let g = -2.0 * err;
        for lambda in [1.0, 0.5, 1e-2, 1e-4] {
            let armijo_bound = err + ALPHA * lambda * g;
            for excess in [1e-12, 1e-6, 1e-2, 1.0, 1e6] {
                let p1 = TrialPoint {
                    lambda,
                    error: armijo_bound + excess,
                };
                let lambda2 = quadratic_interpolation(err, g, &p1);
                assert!(
                    0.0 < lambda2 && lambda2 < lambda / (2.0 * (1.0 - ALPHA)),
                    "lambda {lambda}, excess {excess}: got {lambda2}"
                );
            }
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "lie above the tangent")]
    /// A trial point on the tangent `1 − 2λ` gives no parabola
    fn on_tangent_test() {
        let p1 = TrialPoint {
            lambda: 1.0,
            error: -1.0,
        };
        quadratic_interpolation(1.0, -2.0, &p1);
    }
}
