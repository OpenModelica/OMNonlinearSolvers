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

//! Tests of the Newton solver on small systems.

use faer::MatMut;
use openmodelica_newton::settings::Settings;
use openmodelica_newton::{NewtonError, NewtonStatus, NewtonWorkspace, solve};
use openmodelica_nls_model::{ClosureModel, EvalError, ModelFunctions, SystemInfo};

/// Solve `model` from `x0` with the default settings
fn newton(
    model: &mut impl ModelFunctions,
    x0: &[f64],
) -> (Result<NewtonStatus, NewtonError>, Vec<f64>) {
    let n = model.info().num_unknowns;
    let mut x = x0.to_vec();
    let mut work = NewtonWorkspace::new(n, model.info().num_residuals);
    let result = solve(model, &mut x, &Settings::new(n), &mut work);
    (result, x)
}

/// Whether `x` is within `1e-10` of `expected`, relative to its magnitude
fn close(x: &[f64], expected: &[f64]) -> bool {
    x.iter()
        .zip(expected)
        .all(|(x, e)| (x - e).abs() <= 1e-10 * (1.0 + e.abs()))
}

#[test]
/// A linear system is solved by the first full Newton step
fn linear_test() {
    // x + 2y = 5, 3x − y = 1  ⇒  x = 1, y = 2
    let mut model = ClosureModel::new(2, 2, |x: &[f64], res: &mut [f64]| {
        res[0] = x[0] + 2.0 * x[1] - 5.0;
        res[1] = 3.0 * x[0] - x[1] - 1.0;
        Ok(())
    });

    let (result, x) = newton(&mut model, &[0.0, 0.0]);

    assert!(matches!(result, Ok(NewtonStatus::Solved)), "got {result:?}");
    assert!(close(&x, &[1.0, 2.0]), "got {x:?}");
}

#[test]
/// `x² = 2` converges quadratically to `√2`
fn square_root_test() {
    let mut model = ClosureModel::new(1, 1, |x: &[f64], res: &mut [f64]| {
        res[0] = x[0] * x[0] - 2.0;
        Ok(())
    });

    let (result, x) = newton(&mut model, &[1.0]);

    assert!(matches!(result, Ok(NewtonStatus::Solved)), "got {result:?}");
    assert!(close(&x, &[2.0_f64.sqrt()]), "got {x:?}");
}

#[test]
/// Rosenbrock's function `F = (10·(y − x²), 1 − x)` from the classic start
/// point `(−1.2, 1)` has its root at `(1, 1)`
fn rosenbrock_test() {
    let mut model = ClosureModel::new(2, 2, |x: &[f64], res: &mut [f64]| {
        res[0] = 10.0 * (x[1] - x[0] * x[0]);
        res[1] = 1.0 - x[0];
        Ok(())
    });

    let (result, x) = newton(&mut model, &[-1.2, 1.0]);

    assert!(matches!(result, Ok(NewtonStatus::Solved)), "got {result:?}");
    assert!(close(&x, &[1.0, 1.0]), "got {x:?}");
}

#[test]
/// Unknowns of very different magnitudes are scaled with their nominal values
fn nominal_test() {
    // x = 1e6, y = 1e-6 with residuals of comparable size
    let mut model = ClosureModel::new(2, 2, |x: &[f64], res: &mut [f64]| {
        res[0] = x[0] * 1e-6 - 1.0;
        res[1] = x[1] * 1e6 - 1.0;
        Ok(())
    })
    .with_nominal(vec![1e6, 1e-6]);

    let (result, x) = newton(&mut model, &[2e6, 2e-6]);

    assert!(matches!(result, Ok(NewtonStatus::Solved)), "got {result:?}");
    assert!(close(&x, &[1e6, 1e-6]), "got {x:?}");
}

#[test]
/// `ln(x) = 1` asserts for `x ≤ 0`. The full step from `x = 10` lands at
/// `x ≈ −3`, so the line search backs off, and the iteration still reaches `e`.
fn assert_back_off_test() {
    let mut model = ClosureModel::new(1, 1, |x: &[f64], res: &mut [f64]| {
        if x[0] <= 0.0 {
            return Err(EvalError::Assert);
        }
        res[0] = x[0].ln() - 1.0;
        Ok(())
    });

    let (result, x) = newton(&mut model, &[10.0]);

    assert!(matches!(result, Ok(NewtonStatus::Solved)), "got {result:?}");
    assert!(close(&x, &[std::f64::consts::E]), "got {x:?}");
}

#[test]
/// A model that can't be evaluated at the start point fails at once, `x` is
/// unchanged
fn start_point_assert_test() {
    let mut model = ClosureModel::new(1, 1, |_: &[f64], _: &mut [f64]| Err(EvalError::Assert));

    let (result, x) = newton(&mut model, &[3.0]);

    assert!(
        matches!(result, Err(NewtonError::ModelError(EvalError::Assert))),
        "got {result:?}"
    );
    assert_eq!(x, [3.0]);
}

#[test]
/// A casual tearing set that violates its constraint at the first trial point
/// aborts without backing off
fn constraint_violated_test() {
    let mut model = ClosureModel::new(1, 1, |x: &[f64], res: &mut [f64]| {
        res[0] = x[0] - 5.0;
        if x[0] > 2.0 {
            return Err(EvalError::ConstraintViolated);
        }
        Ok(())
    })
    .with_casual_tearing_set();

    let (result, x) = newton(&mut model, &[0.0]);

    assert!(
        matches!(
            result,
            Err(NewtonError::ModelError(EvalError::ConstraintViolated))
        ),
        "got {result:?}"
    );
    assert_eq!(x, [0.0]);
}

#[test]
/// `x² + 1 = 0` has no real root, the iteration fails instead of looping
fn no_solution_test() {
    let mut model = ClosureModel::new(1, 1, |x: &[f64], res: &mut [f64]| {
        res[0] = x[0] * x[0] + 1.0;
        Ok(())
    });

    let (result, _) = newton(&mut model, &[0.5]);

    assert!(result.is_err(), "got {result:?}");
}

#[test]
/// A start point that already solves the system is accepted
fn already_solved_test() {
    let mut model = ClosureModel::new(1, 1, |x: &[f64], res: &mut [f64]| {
        res[0] = x[0] - 4.0;
        Ok(())
    });

    let (result, x) = newton(&mut model, &[4.0]);

    assert!(matches!(result, Ok(NewtonStatus::Solved)), "got {result:?}");
    assert_eq!(x, [4.0]);
}

#[test]
/// `x + y = 1` and `x + y = 3` are parallel lines: the Jacobian is singular
/// and the linear system inconsistent, so the total pivot solver fails
fn singular_test() {
    let mut model = ClosureModel::new(2, 2, |x: &[f64], res: &mut [f64]| {
        res[0] = x[0] + x[1] - 1.0;
        res[1] = x[0] + x[1] - 3.0;
        Ok(())
    });

    let (result, x) = newton(&mut model, &[0.0, 0.0]);

    assert!(
        matches!(result, Err(NewtonError::Singular)),
        "got {result:?}"
    );
    assert_eq!(x, [0.0, 0.0]);
}

#[test]
/// A residual that is NaN without an assert makes the linear system not
/// finite, which the total pivot solver rejects
fn not_finite_test() {
    let mut model = ClosureModel::new(1, 1, |x: &[f64], res: &mut [f64]| {
        res[0] = x[0].sqrt() - 1.0;
        Ok(())
    });

    let (result, _) = newton(&mut model, &[-4.0]);

    assert!(
        matches!(result, Err(NewtonError::Singular)),
        "got {result:?}"
    );
}

/// `F(x) = atan(x)` with its analytical Jacobian `1/(1 + x²)`
struct Atan {
    info: SystemInfo,
}

impl ModelFunctions for Atan {
    fn info(&self) -> &SystemInfo {
        &self.info
    }

    fn eval(&mut self, x: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
        res[0] = x[0].atan();
        Ok(())
    }

    fn derivative(
        &mut self,
        x: &[f64],
        _: &[f64],
        x_scaling: &[f64],
        mut jacobian: MatMut<'_, f64>,
    ) -> Result<(), EvalError> {
        jacobian[(0, 0)] = x_scaling[0] / (1.0 + x[0] * x[0]);
        Ok(())
    }
}

#[test]
/// The first step of `atan(x) = 0` from `x₀ = 1.28` is damped with the
/// quadratic step of the line search.
///
/// The Newton step `dy = −atan(x₀)·(1 + x₀²) ≈ −2.39` overshoots to `−1.11`,
/// where `φ(1) = atan(x₀ + dy)² ≈ 0.856·err` with `err = atan(x₀)²`. That is
/// above the Armijo bound `err + 0.1·g = 0.8·err` for the slope `g = −2·err`,
/// also for the scaled errors, which have the same ratio. The quadratic step
/// is `λ₂ = −g/(2(φ(1) − err − g)) = err/(φ(1) + err) ≈ 0.539`. Like in C, the
/// cubic step only follows if `φ(1)` is above the Armijo bound at `λ₂`,
/// `err + 0.1·λ₂·g ≈ 0.892·err`, which it isn't. One iteration stops at
/// `x₀ + λ₂·dy`.
fn damped_step_test() {
    let mut model = Atan {
        info: SystemInfo::new(1, 1),
    };
    let x0: f64 = 1.28;
    let dy = -x0.atan() * (1.0 + x0 * x0);
    let err = x0.atan().powi(2);
    let phi1 = (x0 + dy).atan().powi(2);
    let expected = x0 + err / (phi1 + err) * dy;
    let mut x = [x0];
    let mut work = NewtonWorkspace::new(1, 1);

    let settings = Settings::new(1).with_maximum_iterations(1);
    let result = solve(&mut model, &mut x, &settings, &mut work);

    assert!(
        matches!(result, Err(NewtonError::MaxIterations)),
        "got {result:?}"
    );
    assert!(
        (x[0] - expected).abs() < 1e-12,
        "got {x:?}, expected {expected}"
    );
}
