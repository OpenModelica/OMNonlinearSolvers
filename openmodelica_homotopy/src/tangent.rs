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

//! The tangent of the homotopy path, the direction of the predictor step.

use faer::Mat;
use openmodelica_nls_model::{EvalError, ModelFunctions, scaling};
use openmodelica_total_pivot_solver::{SolveError, Workspace};

/// Why the tangent can't be computed.
#[derive(Debug)]
pub(crate) enum TangentError {
    /// The Jacobian of the homotopy can't be evaluated
    ModelError(EvalError),
    /// The linear system of the tangent has no solution
    Singular,
}

impl From<EvalError> for TangentError {
    fn from(error: EvalError) -> Self {
        TangentError::ModelError(error)
    }
}

impl From<SolveError> for TangentError {
    fn from(_: SolveError) -> Self {
        TangentError::Singular
    }
}

/// Buffers of [`tangent`] for a homotopy with `n` residuals and `n + 1`
/// unknowns.
pub(crate) struct TangentWorkspace {
    /// The Jacobian `[∂H/∂x | ∂H/∂λ]`, `n×(n+1)`
    jacobian: Mat<f64>,
    /// Buffers of the total pivot solver
    pivot: Workspace,
    /// The maxima of the row scaling
    row_maxima: Box<[f64]>,
}

impl TangentWorkspace {
    /// Allocates the buffers for a homotopy with `n` residuals.
    pub(crate) fn new(n: usize) -> Self {
        TangentWorkspace {
            jacobian: Mat::zeros(n, n + 1),
            pivot: Workspace::new(n),
            row_maxima: vec![0.0; n].into(),
        }
    }
}

/// Compute the tangent `t` of the path `H(y) = 0` at `y = (x, λ)`.
///
/// The tangent solves `[∂H/∂x | ∂H/∂λ]·t = 0`. The `n×(n+1)` Jacobian is
/// evaluated into `work` with `h = H(y)` and its columns scaled with
/// `x_scaling`, then its rows are scaled with their largest entry. The total pivot solver
/// fixes the entry of `t` it chooses to 1, and `t` is scaled back to the
/// unknowns. It is neither normalized nor oriented, see [`orient`].
///
/// Returns the coordinate the solver fixed, the corrector keeps it fixed.
/// From the predictor step of `homotopyAlgorithm`.
///
/// # Errors
///
/// Returns a [`TangentError`] if the Jacobian can't be evaluated or the linear
/// system can't be solved. `t` is unspecified then.
///
/// # Panics
///
/// Panics if the sizes of `y`, `h`, `x_scaling`, `work` and `t` don't match
/// the homotopy.
pub(crate) fn tangent(
    homotopy: &mut impl ModelFunctions,
    y: &[f64],
    h: &[f64],
    x_scaling: &[f64],
    work: &mut TangentWorkspace,
    t: &mut [f64],
) -> Result<usize, TangentError> {
    let n = homotopy.info().num_residuals;
    assert_eq!(
        homotopy.info().num_unknowns,
        n + 1,
        "Homotopy needs λ as last unknown"
    );

    homotopy.derivative(y, h, x_scaling, work.jacobian.as_mut())?;
    scaling::rows(work.jacobian.as_mut(), &mut work.row_maxima);

    let solution = openmodelica_total_pivot_solver::solve(
        &work.jacobian,
        None,
        homotopy.info().casual_tearing_set,
        &mut work.pivot,
    )?;
    scaling::unscale(solution.x, x_scaling, t);

    Ok(solution.fixed_column)
}

/// Flip the tangent `t` if it points against the `previous` tangent, so the
/// path is followed in one direction.
///
/// On the first step the previous tangent is `(0, …, 0, ±1)` and can be
/// orthogonal to the path, then `flip_if_orthogonal` decides, set for a
/// negative start direction. From the predictor step of `homotopyAlgorithm`.
///
/// # Panics
///
/// Panics if `t` and `previous` don't have the same length.
pub(crate) fn orient(t: &mut [f64], previous: &[f64], flip_if_orthogonal: bool) {
    assert_eq!(t.len(), previous.len(), "Previous tangent has wrong length");

    let along_previous: f64 = t.iter().zip(previous).map(|(t, p)| t * p).sum();
    let orthogonal = along_previous.abs() < f64::EPSILON;
    if along_previous < 0.0 || (orthogonal && flip_if_orthogonal) {
        t.iter_mut().for_each(|t| *t = -*t);
    }
}

#[cfg(test)]
mod tangent_tests {
    use super::*;
    use crate::NewtonHomotopy;
    use faer::MatMut;
    use openmodelica_nls_model::{ClosureModel, SystemInfo};

    /// The linear homotopy `H(x, λ) = x − c·λ`, its path is `x = c·λ`
    struct Linear {
        c: f64,
        info: SystemInfo,
    }

    impl Linear {
        fn new(c: f64) -> Self {
            Linear {
                c,
                info: SystemInfo::new(1, 2),
            }
        }
    }

    impl ModelFunctions for Linear {
        fn info(&self) -> &SystemInfo {
            &self.info
        }

        fn eval(&mut self, y: &[f64], h: &mut [f64]) -> Result<(), EvalError> {
            h[0] = y[0] - self.c * y[1];
            Ok(())
        }

        fn derivative(
            &mut self,
            _: &[f64],
            _: &[f64],
            y_scaling: &[f64],
            mut jacobian: MatMut<'_, f64>,
        ) -> Result<(), EvalError> {
            jacobian[(0, 0)] = y_scaling[0];
            jacobian[(0, 1)] = -self.c * y_scaling[1];
            Ok(())
        }
    }

    /// The tangent of `homotopy` at `y` with the scaling `x_scaling`
    fn tangent_of(
        homotopy: &mut impl ModelFunctions,
        y: &[f64],
        x_scaling: &[f64],
    ) -> (Result<usize, TangentError>, Vec<f64>) {
        let info = homotopy.info();
        let (n, m) = (info.num_residuals, info.num_unknowns);
        let mut h = vec![0.0; n];
        homotopy.eval(y, &mut h).unwrap();
        let mut t = vec![0.0; m];
        let fixed = tangent(
            homotopy,
            y,
            &h,
            x_scaling,
            &mut TangentWorkspace::new(n),
            &mut t,
        );
        (fixed, t)
    }

    #[test]
    /// `H = x − 2λ`: the solver pivots on the larger `∂H/∂λ = −2` and fixes
    /// `x`, so `t = (1, 0.5)` along the path `x = 2λ`
    fn fixes_x_test() {
        let (fixed, t) = tangent_of(&mut Linear::new(2.0), &[0.0, 0.0], &[1.0, 1.0]);

        assert_eq!(fixed.unwrap(), 0);
        assert_eq!(t, [1.0, 0.5]);
    }

    #[test]
    /// `H = x − 0.5·λ`: the solver pivots on `∂H/∂x = 1` and fixes `λ`, so
    /// `t = (0.5, 1)`
    fn fixes_lambda_test() {
        let (fixed, t) = tangent_of(&mut Linear::new(0.5), &[0.0, 0.0], &[1.0, 1.0]);

        assert_eq!(fixed.unwrap(), 1);
        assert_eq!(t, [0.5, 1.0]);
    }

    #[test]
    /// With `x` scaled by 10, `∂H/∂x` becomes the pivot of `H = x − 2λ`, `λ` is
    /// fixed in the scaled system and `t = (2, 1)` after scaling back
    fn scaling_test() {
        let (fixed, t) = tangent_of(&mut Linear::new(2.0), &[0.0, 0.0], &[10.0, 1.0]);

        assert_eq!(fixed.unwrap(), 1);
        assert_eq!(t, [2.0, 1.0]);
    }

    #[test]
    /// A tangent against the previous one is flipped, one along it is kept
    fn orient_test() {
        let mut t = [1.0, 0.5];
        orient(&mut t, &[-1.0, -1.0], false);
        assert_eq!(t, [-1.0, -0.5]);

        orient(&mut t, &[-1.0, -1.0], false);
        assert_eq!(t, [-1.0, -0.5]);
    }

    #[test]
    /// A tangent orthogonal to the previous one is only flipped if asked to
    fn orient_orthogonal_test() {
        // (1, 0.5) · (0.5, −1) = 0
        let mut t = [1.0, 0.5];
        orient(&mut t, &[0.5, -1.0], false);
        assert_eq!(t, [1.0, 0.5]);

        orient(&mut t, &[0.5, -1.0], true);
        assert_eq!(t, [-1.0, -0.5]);
    }

    #[test]
    /// On the Newton homotopy of a nonlinear system the tangent solves
    /// `[∂H/∂x | ∂H/∂λ]·t = 0`
    fn newton_homotopy_test() {
        // F(x) = (x₀² + x₁ − 3, x₀·x₁ − 2), start at x₀ = (2, 1)
        let mut model = ClosureModel::new(2, 2, |x: &[f64], res: &mut [f64]| {
            res[0] = x[0] * x[0] + x[1] - 3.0;
            res[1] = x[0] * x[1] - 2.0;
            Ok(())
        });
        let mut homotopy = NewtonHomotopy::new(&mut model, &[2.0, 0.0]);
        let y = [2.0, 1.0, 0.0];

        let (_, mut t) = tangent_of(&mut homotopy, &y, &[2.0, 1.0, 1.0]);
        orient(&mut t, &[0.0, 0.0, 1.0], false);

        // ∂H/∂x = J(2, 1) = [[4, 1], [1, 2]], ∂H/∂λ = F(x₀) = (2, 0)
        let residual = [4.0 * t[0] + t[1] + 2.0 * t[2], t[0] + 2.0 * t[1]];
        assert!(residual.iter().all(|r| r.abs() < 1e-6), "{residual:?}");
        assert!(t[2] > 0.0, "oriented along the previous tangent");
    }

    #[test]
    /// A Jacobian that can't be evaluated fails the tangent
    fn model_error_test() {
        let mut model = ClosureModel::new(1, 2, |y: &[f64], h: &mut [f64]| {
            if y[0] > 0.5 {
                return Err(EvalError::Assert);
            }
            h[0] = y[0] - y[1];
            Ok(())
        });

        // The forward difference steps past x = 0.5
        let mut h = [0.0];
        model.eval(&[0.5, 0.0], &mut h).unwrap();
        let mut t = [0.0; 2];
        let result = tangent(
            &mut model,
            &[0.5, 0.0],
            &h,
            &[1.0, 1.0],
            &mut TangentWorkspace::new(1),
            &mut t,
        );

        assert!(
            matches!(result, Err(TangentError::ModelError(EvalError::Assert))),
            "got {result:?}"
        );
    }

    #[test]
    /// A Jacobian with NaN can't be solved
    fn singular_test() {
        let mut model = ClosureModel::new(1, 2, |y: &[f64], h: &mut [f64]| {
            h[0] = y[0].sqrt() - y[1];
            Ok(())
        });

        let (result, _) = tangent_of(&mut model, &[-1.0, 0.0], &[1.0, 1.0]);

        assert!(
            matches!(result, Err(TangentError::Singular)),
            "got {result:?}"
        );
    }
}
