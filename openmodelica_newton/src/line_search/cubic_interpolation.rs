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

//! Cubic interpolation of the line search: the last damping step of
//! `newtonAlgorithm` in the C runtime, when the quadratic step was still too
//! long.

use super::TrialPoint;

/// Cubic interpolation on `φ(λ) = ‖F(x + λ·dy)‖²`
///
/// Fits `φ(λ) ≈ a₃λ³ + a₂λ² + g·λ + err` through the two trial points `p1`
/// and `p2` and returns the step factor at its local minimum. If the fit has
/// no cubic term, that is the minimum of the parabola.
///
/// * `err`: `φ(0)`, the squared error at the current iterate
/// * `g`: `φ'(0)`, the slope at the current iterate
/// * `p1`, `p2`: two trial points with different, positive `lambda`
///
/// Returns `None` if the cubic has no local minimum. The result isn't checked
/// for being finite: a fit without cubic and quadratic term returns `+∞`.
pub(crate) fn cubic_interpolation(
    err: f64,
    g: f64,
    p1: &TrialPoint,
    p2: &TrialPoint,
) -> Option<f64> {
    debug_assert!(
        p1.lambda > 0.0 && p2.lambda > 0.0,
        "Trial points must have positive lambda, got {} and {}",
        p1.lambda,
        p2.lambda
    );
    debug_assert!(
        (p1.lambda - p2.lambda).abs() > f64::EPSILON,
        "Two trial points must have different lambda, got {} and {}",
        p1.lambda,
        p2.lambda
    );

    let rhs1 = p1.error - g * p1.lambda - err;
    let rhs2 = p2.error - g * p2.lambda - err;

    let a3 =
        (rhs1 / (p1.lambda * p1.lambda) - rhs2 / (p2.lambda * p2.lambda)) / (p1.lambda - p2.lambda);
    let a2 = (-p2.lambda * rhs1 / (p1.lambda * p1.lambda)
        + p1.lambda * rhs2 / (p2.lambda * p2.lambda))
        / (p1.lambda - p2.lambda);

    // Only exactly zero needs its own case: for a tiny a3 and a2 > 0 the
    // conjugate form below tends to the minimum of the parabola.
    if a3 == 0.0 {
        return Some(-g / (2.0 * a2));
    }

    let d = a2 * a2 - 3.0 * a3 * g;
    if d <= 0.0 {
        return None;
    }

    // The root (−a₂ + √d) / (3a₃), multiplied by its conjugate, so that
    // −a₂ + √d doesn't cancel for a₂ > 0
    if a2 > 0.0 {
        return Some(-g / (a2 + d.sqrt()));
    }

    Some((-a2 + d.sqrt()) / (3.0 * a3))
}

#[cfg(test)]
mod cubic_interpolation_tests {
    use super::*;

    /// Interpolates `φ(λ) = 1 − 2λ + a₂λ² + a₃λ³` through `λ₁` and `λ₂`
    fn interpolate(a2: f64, a3: f64, lambda1: f64, lambda2: f64) -> Option<f64> {
        let phi = |lambda: f64| 1.0 - 2.0 * lambda + a2 * lambda * lambda + a3 * lambda.powi(3);
        let err = phi(0.0);
        let g = -2.0 * err;
        let p1 = TrialPoint {
            lambda: lambda1,
            error: phi(lambda1),
        };
        let p2 = TrialPoint {
            lambda: lambda2,
            error: phi(lambda2),
        };

        cubic_interpolation(err, g, &p1, &p2)
    }

    #[test]
    /// `φ(λ) = 1 − 2λ + 4λ²` is a parabola with its minimum at `λ = 0.25`
    fn parabola_test() {
        assert_eq!(interpolate(4.0, 0.0, 1.0, 0.5), Some(0.25));
    }

    #[test]
    /// `φ(λ) = 1 − 2λ − λ² + 4λ³` has its local minimum at `λ = 0.5`, with
    /// `a₂ ≤ 0`
    fn cubic_test() {
        assert_eq!(interpolate(-1.0, 4.0, 1.0, 0.25), Some(0.5));
    }

    #[test]
    /// `φ(λ) = 1 − 2λ + 2.75λ² − λ³` has its local minimum at `λ = 0.5`, with
    /// `a₂ > 0`
    fn cubic_conjugate_test() {
        assert_eq!(interpolate(2.75, -1.0, 1.0, 0.25), Some(0.5));
    }

    #[test]
    /// `φ(λ) = 1 − 2λ + 2λ² − λ³` decreases everywhere, `d = 4 − 6 < 0`
    fn cubic_no_minimum_test() {
        assert_eq!(interpolate(2.0, -1.0, 1.0, 0.5), None);
    }

    #[test]
    /// The order of the trial points doesn't matter
    fn trial_point_order_test() {
        assert_eq!(interpolate(-1.0, 4.0, 0.25, 1.0), Some(0.5));
    }

    #[test]
    /// The fits of [`parabola_test`], [`cubic_test`] and
    /// [`cubic_conjugate_test`] from trial points below 1, where dividing and
    /// multiplying by `λ²` differ
    fn trial_points_below_one_test() {
        assert_eq!(interpolate(4.0, 0.0, 0.5, 0.125), Some(0.25));
        assert_eq!(interpolate(-1.0, 4.0, 0.5, 0.125), Some(0.5));
        assert_eq!(interpolate(2.75, -1.0, 0.75, 0.25), Some(0.5));
    }

    #[test]
    /// `φ(λ) = 1 − 2λ + λ² + 10⁻¹²λ³` has its minimum at
    /// `λ = 1 − 1.5·10⁻¹² + O(10⁻²⁴)`. With `a₂ > 0` and a tiny `a₃`, the
    /// root `(−a₂ + √d)/(3a₃)` loses all but four digits to cancellation, its
    /// conjugate form `−g/(a₂ + √d)` keeps them.
    fn cubic_conjugate_cancellation_test() {
        let lambda = interpolate(1.0, 1e-12, 1.0, 0.5).unwrap();

        assert!((lambda - (1.0 - 1.5e-12)).abs() < 1e-14, "got {lambda}");
    }
}
