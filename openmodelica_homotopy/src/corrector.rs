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

//! The corrector step back to the homotopy path, strategy 1 of the C runtime:
//! Newton's method with one coordinate fixed.

use faer::{ColRef, Mat, MatRef};
use openmodelica_nls_model::scaling::{self, scaled_squared_error, squared_error};
use openmodelica_nls_model::{EvalError, ModelFunctions};
use openmodelica_total_pivot_solver::{SolveError, Workspace};

/// How a corrector ended without an error.
#[derive(Debug, PartialEq)]
pub(crate) enum Correction {
    /// `‖H‖` or its scaled norm is below the tolerance
    Converged,
    /// The maximum number of Newton steps is reached
    NotConverged,
}

/// Why the corrector failed.
#[derive(Debug)]
pub(crate) enum CorrectorError {
    /// The homotopy or its Jacobian can't be evaluated
    ModelError,
    /// The linear system of a Newton step has no solution
    Singular,
}

impl From<EvalError> for CorrectorError {
    fn from(_: EvalError) -> Self {
        CorrectorError::ModelError
    }
}

impl From<SolveError> for CorrectorError {
    fn from(_: SolveError) -> Self {
        CorrectorError::Singular
    }
}

/// Buffers of the corrector, sized for a homotopy with `n` residuals and
/// `n + 1` unknowns.
pub(crate) struct CorrectorWorkspace {
    /// Variable scaling of the `n + 1` unknowns, `λ` last
    x_scaling: Box<[f64]>,
    /// The linear system of a Newton step, `n×(n+1)`
    jacobian: Mat<f64>,
    /// Buffers of the total pivot solver
    pivot: Workspace,
    /// The maxima of the row scaling
    row_maxima: Box<[f64]>,
    /// Newton step
    dy: Box<[f64]>,
    /// Residual scaling of the latest Jacobian
    res_scaling: Box<[f64]>,
}

impl CorrectorWorkspace {
    /// Allocates the buffers for a homotopy whose unknowns are scaled with
    /// `x_scaling`, `λ` last.
    pub(crate) fn new(x_scaling: &[f64]) -> Self {
        let m = x_scaling.len();
        CorrectorWorkspace {
            x_scaling: x_scaling.into(),
            jacobian: Mat::zeros(m - 1, m),
            pivot: Workspace::new(m - 1),
            row_maxima: vec![0.0; m - 1].into(),
            dy: vec![0.0; m].into(),
            res_scaling: vec![1.0; m - 1].into(),
        }
    }
}

/// Correct the predicted point `y` back to the path `H(y) = 0`, keeping the
/// coordinate `fixed` as it is.
///
/// `h` holds `H(y)` on entry and on exit. Each Newton step solves the scaled
/// Jacobian with `H` in the column of `fixed`, which gives the step of the
/// other coordinates. The corrector stops once `‖H‖` or the norm of `H`
/// divided by the residual scaling is below `h_eps`, or after `max_steps`
/// steps. Before the first Jacobian `H` counts as scaled already. Like in the
/// C runtime the test comes before each step, so a corrector that only
/// converges with its last step is [`Correction::NotConverged`]. From the
/// corrector step of `homotopyAlgorithm` with `homBacktraceStrategy = 1`.
///
/// # Errors
///
/// Returns a [`CorrectorError`] if the homotopy or its Jacobian can't be
/// evaluated, or a Newton step can't be solved. After a
/// [`CorrectorError::Singular`] `y` and `h` are the latest corrected point,
/// after a [`CorrectorError::ModelError`] they are unspecified.
///
/// # Panics
///
/// Panics if the sizes of `y`, `h` and `work` don't match the homotopy, or
/// `fixed` is not a coordinate of `y`.
pub(crate) fn corrector(
    homotopy: &mut impl ModelFunctions,
    fixed: usize,
    h_eps: f64,
    max_steps: usize,
    work: &mut CorrectorWorkspace,
    y: &mut [f64],
    h: &mut [f64],
) -> Result<Correction, CorrectorError> {
    let n = homotopy.info().num_residuals;
    assert_eq!(y.len(), n + 1, "Point has wrong length");
    assert!(fixed <= n, "Invalid fixed coordinate {fixed}");

    work.res_scaling.fill(1.0);
    for _ in 0..max_steps {
        let converged = squared_error(h).sqrt() < h_eps
            || scaled_squared_error(h, &work.res_scaling).sqrt() < h_eps;
        if converged {
            return Ok(Correction::Converged);
        }

        homotopy.derivative(y, h, &work.x_scaling, work.jacobian.as_mut())?;
        row_sums(work.jacobian.as_ref(), &mut work.res_scaling);

        work.jacobian
            .col_mut(fixed)
            .copy_from(ColRef::from_slice(h));
        scaling::rows(work.jacobian.as_mut(), &mut work.row_maxima);
        let solution = openmodelica_total_pivot_solver::solve(
            &work.jacobian,
            Some(fixed),
            homotopy.info().casual_tearing_set,
            &mut work.pivot,
        )?;
        scaling::unscale(solution.x, &work.x_scaling, &mut work.dy);
        work.dy[fixed] = 0.0;

        for (y, dy) in y.iter_mut().zip(&work.dy) {
            *y += dy;
        }
        homotopy.eval(y, h)?;
    }
    Ok(Correction::NotConverged)
}

/// Write the sum of the absolute entries of each row of `matrix` into `sums`,
/// over all columns. From `matVecMultAbs` with a vector of ones.
fn row_sums(matrix: MatRef<'_, f64>, sums: &mut [f64]) {
    for (sum, row) in sums.iter_mut().zip(matrix.row_iter()) {
        *sum = row.norm_l1();
    }
}

#[cfg(test)]
mod corrector_tests {
    use super::*;
    use faer::mat;
    use openmodelica_nls_model::ClosureModel;

    /// Result, point and `H` after correcting `y` of the one-dimensional
    /// `homotopy` with the coordinate `fixed`
    fn correct(
        homotopy: impl FnMut(&[f64], &mut [f64]) -> Result<(), EvalError>,
        y: [f64; 2],
        fixed: usize,
        max_steps: usize,
    ) -> (Result<Correction, CorrectorError>, [f64; 2], [f64; 1]) {
        let mut homotopy = ClosureModel::new(1, 2, homotopy);
        let mut y = y;
        let mut h = [0.0];
        homotopy.eval(&y, &mut h).unwrap();
        let mut work = CorrectorWorkspace::new(&[1.0, 1.0]);

        let result = corrector(
            &mut homotopy,
            fixed,
            1e-10,
            max_steps,
            &mut work,
            &mut y,
            &mut h,
        );
        (result, y, h)
    }

    /// The linear homotopy `H = x − 2λ`
    fn linear(y: &[f64], h: &mut [f64]) -> Result<(), EvalError> {
        h[0] = y[0] - 2.0 * y[1];
        Ok(())
    }

    /// `H = c·(x² − λ)`, scaled by `c`
    fn scaled_parabola(c: f64) -> impl FnMut(&[f64], &mut [f64]) -> Result<(), EvalError> {
        move |y: &[f64], h: &mut [f64]| {
            h[0] = c * (y[0] * y[0] - y[1]);
            Ok(())
        }
    }

    /// The unit circle `H = x² + λ² − 1`
    fn circle(y: &[f64], h: &mut [f64]) -> Result<(), EvalError> {
        h[0] = y[0] * y[0] + y[1] * y[1] - 1.0;
        Ok(())
    }

    #[test]
    /// A point on the path is accepted without a Newton step
    fn on_path_test() {
        let (result, y, _) = correct(linear, [0.2, 0.1], 1, 0);

        assert!(
            matches!(result, Ok(Correction::NotConverged)),
            "no steps allowed: {result:?}"
        );
        let (result, y2, _) = correct(linear, [0.2, 0.1], 1, 20);
        assert!(
            matches!(result, Ok(Correction::Converged)),
            "got {result:?}"
        );
        assert_eq!((y, y2), ([0.2, 0.1], [0.2, 0.1]));
    }

    #[test]
    /// With `λ` fixed, one Newton step moves `x` onto the linear path
    fn fixed_lambda_test() {
        let (result, y, h) = correct(linear, [0.3, 0.1], 1, 20);

        assert!(
            matches!(result, Ok(Correction::Converged)),
            "got {result:?}"
        );
        assert!((y[0] - 0.2).abs() < 1e-10 && y[1] == 0.1, "{y:?}");
        assert!(h[0].abs() < 1e-10);
    }

    #[test]
    /// With `x` fixed, the corrector moves `λ` instead
    fn fixed_x_test() {
        let (result, y, _) = correct(linear, [0.3, 0.1], 0, 20);

        assert!(
            matches!(result, Ok(Correction::Converged)),
            "got {result:?}"
        );
        assert!(y[0] == 0.3 && (y[1] - 0.15).abs() < 1e-10, "{y:?}");
    }

    #[test]
    /// On the unit circle with `λ = 0.6` fixed, Newton converges from `x = 1`
    /// to `x = 0.8`
    fn circle_test() {
        let (result, y, _) = correct(circle, [1.0, 0.6], 1, 20);

        assert!(
            matches!(result, Ok(Correction::Converged)),
            "got {result:?}"
        );
        assert!((y[0] - 0.8).abs() < 1e-9 && y[1] == 0.6, "{y:?}");
    }

    #[test]
    /// One Newton step is not enough on the circle
    fn not_converged_test() {
        let (result, y, _) = correct(circle, [1.0, 0.6], 1, 1);

        assert!(
            matches!(result, Ok(Correction::NotConverged)),
            "got {result:?}"
        );
        assert!(
            (y[0] - 0.82).abs() < 1e-6,
            "one step from x = 1 goes to 0.82: {y:?}"
        );
    }

    #[test]
    /// A homotopy that asserts at the corrected point fails the corrector
    fn model_error_test() {
        let asserting = |y: &[f64], h: &mut [f64]| {
            if y[0] < 0.9 {
                return Err(EvalError::Assert);
            }
            circle(y, h)
        };

        let (result, _, _) = correct(asserting, [1.0, 0.6], 1, 20);

        assert!(
            matches!(result, Err(CorrectorError::ModelError)),
            "got {result:?}"
        );
    }

    #[test]
    /// `H = λ − 1` doesn't depend on `x`: with `λ = 0.5` fixed no step of `x`
    /// reaches the path
    fn singular_test() {
        let constant = |y: &[f64], h: &mut [f64]| {
            h[0] = y[1] - 1.0;
            Ok(())
        };

        let (result, y, _) = correct(constant, [0.0, 0.5], 1, 20);

        assert!(
            matches!(result, Err(CorrectorError::Singular)),
            "got {result:?}"
        );
        assert_eq!(y, [0.0, 0.5], "the latest corrected point");
    }

    #[test]
    /// The residual scaling sums over all columns, `∂H/∂λ` included
    fn row_sums_test() {
        let mut sums = [0.0; 2];
        row_sums(mat![[1.0, -2.0, 3.0], [0.0, 4.0, -5.0]].as_ref(), &mut sums);

        assert_eq!(sums, [6.0, 9.0]);
    }

    #[test]
    /// The plain or the scaled residual below `h_eps = 10⁻¹⁰` is enough. With
    /// `λ = 1` fixed, Newton's method for `c·(x² − 1)` from `x = 1.1` leaves
    /// `x² − 1 ≈ 1.1·10⁻¹⁰` after 3 steps and `≈ 2·10⁻⁵` after 2. The residual
    /// scaling is the row sum `|c|·(2x + 1) ≈ 3|c|` of the latest Jacobian.
    ///
    /// * `c = 10⁶`: after 3 steps the plain residual is `1.1·10⁻⁴`, the scaled
    ///   one `3.5·10⁻¹¹` converged.
    /// * `c = 10⁻⁶`: after 2 steps the plain residual is `2·10⁻¹¹` converged,
    ///   the scaled one `7·10⁻⁶`.
    fn plain_or_scaled_residual_test() {
        let (result, _, _) = correct(scaled_parabola(1e6), [1.1, 1.0], 1, 4);
        assert!(
            matches!(result, Ok(Correction::Converged)),
            "scaled: got {result:?}"
        );

        let (result, _, _) = correct(scaled_parabola(1e-6), [1.1, 1.0], 1, 3);
        assert!(
            matches!(result, Ok(Correction::Converged)),
            "plain: got {result:?}"
        );
    }
}
