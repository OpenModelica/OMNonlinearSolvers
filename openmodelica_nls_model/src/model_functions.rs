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

//! The interface of a nonlinear system and the error of a failed evaluation.

use faer::MatMut;

use crate::SystemInfo;

/// Why evaluating a model failed.
#[derive(Debug)]
pub enum EvalError {
    /// The model raised an assert, e.g. a function outside its domain
    Assert,
    /// A constraint of a casual tearing set is violated
    ConstraintViolated,
}

/// A nonlinear system `F(x) = 0`.
///
/// Implemented by the type of each concrete system, its fields hold the data
/// needed to evaluate it. The facts about the system, its sizes, nominal
/// values, bounds and tearing kind, are a [`SystemInfo`] returned by
/// [`info`](Self::info). A system with an additional unknown, like a homotopy
/// `H(x, λ)` with `n` residuals and `n+1` unknowns, is non-square.
///
/// [`info`](Self::info) and [`eval`](Self::eval) are required,
/// [`derivative`](Self::derivative) defaults to a numerical Jacobian and
/// [`start_point_checked`](Self::start_point_checked) to doing nothing.
pub trait ModelFunctions {
    /// Facts about the system: sizes, nominal values, bounds and tearing kind.
    fn info(&self) -> &SystemInfo;

    /// Evaluate the residual `res := F(x)`.
    ///
    /// `x` has `info().num_unknowns` entries, `res` has `info().num_residuals`
    /// entries.
    ///
    /// # Errors
    ///
    /// Returns an [`EvalError`] if `F` can't be evaluated at `x`.
    fn eval(&mut self, x: &[f64], res: &mut [f64]) -> Result<(), EvalError>;

    /// Evaluate the scaled Jacobian `jacobian := J(x)·diag(x_scaling)` with
    /// `J = ∂F/∂x`.
    ///
    /// `fx` is the residual `F(x)` at the same `x`, the caller has computed it
    /// already. `x_scaling` has one factor per unknown, the solvers pass their
    /// variable scaling, see [`scaling::variables`](crate::scaling::variables).
    /// `jacobian` is a `num_residuals × num_unknowns` view of a column-major
    /// matrix.
    ///
    /// The default computes [`forward_differences`], one evaluation of `F`
    /// per column, with the step `h = sqrt(20·ε)·(1 + |xⱼ|)` of the C runtime.
    /// When `x+h` reaches maximum bound backwards differences are used
    /// instead. It allocates the perturbed point, models that are solved
    /// often override this method and pass a buffer.
    ///
    /// Systems with an analytical Jacobian override it and multiply each
    /// column `j` by `x_scaling[j]`. The scaling is part of the evaluation,
    /// so that the entries are rounded like in the C runtime, which scales
    /// inside its Jacobian functions.
    ///
    /// # Errors
    ///
    /// Returns an [`EvalError`] if `F` can't be evaluated at a perturbed `x`.
    ///
    /// # Panics
    ///
    /// Panics if the shape of `jacobian` doesn't match `x` and `fx`, or
    /// `x_scaling` doesn't have one factor per unknown.
    fn derivative(
        &mut self,
        x: &[f64],
        fx: &[f64],
        x_scaling: &[f64],
        jacobian: MatMut<'_, f64>,
    ) -> Result<(), EvalError> {
        forward_differences(self, x, fx, x_scaling, &mut x.to_vec(), jacobian)
    }

    /// Called by the homotopy solver when it has checked its start point,
    /// evaluated `F` and the Jacobian there, before Newton's method and the
    /// homotopies start. Not called if the start point solves the system.
    ///
    /// The OpenModelica C runtime freezes the relations of a model with
    /// events from here on, `solveContinuous` in `solveHomotopy`.
    fn start_point_checked(&mut self) {}
}

/// The scaled Jacobian `J(x)·diag(x_scaling)` of `model` by forward
/// differences, the default of [`ModelFunctions::derivative`].
///
/// One evaluation of `F` per column, with the step `h = sqrt(20·ε)·(1 + |xⱼ|)`
/// of the C runtime, backwards where `x + h` reaches the upper bound. `fx` is
/// `F(x)`. Models that override `derivative` can fall back to it. Each
/// difference is multiplied by `1/h·x_scaling[j]`, like in
/// `getNumericalJacobianHomotopy`. `x_perturbed` is a buffer for the
/// perturbed points, one entry per unknown.
///
/// # Errors
///
/// Returns the [`EvalError`] of the model if `F` can't be evaluated at a
/// perturbed point.
///
/// # Panics
///
/// Panics if the shape of `jacobian` doesn't match `x` and `fx`, or
/// `x_scaling` or `x_perturbed` don't have one entry per unknown.
pub fn forward_differences<M: ModelFunctions + ?Sized>(
    model: &mut M,
    x: &[f64],
    fx: &[f64],
    x_scaling: &[f64],
    x_perturbed: &mut [f64],
    jacobian: MatMut<'_, f64>,
) -> Result<(), EvalError> {
    assert_eq!(x.len(), jacobian.ncols());
    assert_eq!(fx.len(), jacobian.nrows());
    assert_eq!(x_scaling.len(), x.len(), "Scaling vector has wrong length");
    assert_eq!(x_perturbed.len(), x.len(), "Buffer has wrong length");

    let delta = (f64::EPSILON * 20.0).sqrt();

    x_perturbed.copy_from_slice(x);

    for (j, col) in jacobian.col_iter_mut().enumerate() {
        let mut h = delta * (1.0 + x[j].abs());

        // step backwards if x + h would reach the upper bound
        if x[j] + h >= model.info().max[j] {
            h *= -1.0;
        }

        x_perturbed[j] += h;

        // col := F(x + h·eⱼ)
        let column: &mut [f64] = col.try_as_col_major_mut().unwrap().as_slice_mut();
        model.eval(x_perturbed, column)?;

        // col := (F(x + h·eⱼ) − F(x)) / h · sⱼ
        let factor = 1.0 / h * x_scaling[j];
        for i in 0..column.len() {
            column[i] = (column[i] - fx[i]) * factor;
        }

        x_perturbed[j] = x[j]; // undo the perturbation
    }

    Ok(())
}

#[cfg(test)]
mod numeric_jacobian_tests {
    use super::*;
    use equator::assert;
    use faer::utils::approx::{ApproxEq, CwiseMat};
    use faer::{Mat, mat};

    /// Equal within the accuracy of a forward difference
    fn approx_eq_fd_tol() -> CwiseMat<ApproxEq<f64>> {
        CwiseMat(ApproxEq {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
        })
    }

    /// Evaluate `F(x)` and the numerical Jacobian of `model` at `x`
    fn jacobian_at(model: &mut impl ModelFunctions, x: &[f64]) -> Result<Mat<f64>, EvalError> {
        let info = model.info();
        let mut f_x = vec![0.0; info.num_residuals];
        let mut jac = Mat::zeros(info.num_residuals, info.num_unknowns);
        model.eval(x, &mut f_x)?;
        model.derivative(x, &f_x, &vec![1.0; x.len()], jac.as_mut())?;
        Ok(jac)
    }

    #[test]
    /// Evaluate a Jacobian of a non-square system with 2 residuals and 3
    /// unknowns, shaped like a homotopy `H(x, λ)` with `λ` as additional
    /// unknown:
    /// ```text
    /// H(x, y, λ) = ( x² + y - λ )
    ///              ( x·y·λ - 2  )
    /// ```
    /// at `(x, y, λ) = (1, 2, 1)` and `(2, -1, 0.5)`
    fn numeric_jacobian_test() {
        struct H {
            info: SystemInfo,
        }
        impl ModelFunctions for H {
            fn info(&self) -> &SystemInfo {
                &self.info
            }

            fn eval(&mut self, x_vec: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
                let [x, y, lambda] = [x_vec[0], x_vec[1], x_vec[2]];

                res[0] = x * x + y - lambda;
                res[1] = x * y * lambda - 2.0;
                Ok(())
            }
        }
        let mut model = H {
            info: SystemInfo::new(2, 3),
        };

        assert!(jacobian_at(&mut model, &[1.0, 2.0, 1.0]).unwrap() :approx_eq_fd_tol(): mat![[2.0, 1.0, -1.0], [2.0, 1.0, 2.0]]);
        assert!(jacobian_at(&mut model, &[2.0, -1.0, 0.5]).unwrap() :approx_eq_fd_tol(): mat![[4.0, 1.0, -1.0], [-0.5, 1.0, -2.0]]);
    }

    /// `F: ℝ² → ℝ, (x,y) → x² + y²`,
    /// `J(x, y) = [2x, 2y]`
    struct F {
        info: SystemInfo,
    }
    impl ModelFunctions for F {
        fn info(&self) -> &SystemInfo {
            &self.info
        }

        fn eval(&mut self, x_vec: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
            let [x, y] = [x_vec[0], x_vec[1]];
            if !(-1.0..=1.0).contains(&x) || !(-1.0..=1.0).contains(&y) {
                return Err(EvalError::Assert);
            }
            res[0] = x.powi(2) + y.powi(2);
            Ok(())
        }
    }

    #[test]
    /// The difference quotient is multiplied by `1/h·sⱼ` like in
    /// `getNumericalJacobianHomotopy`, for `F(x, y) = x² + y²` at `(0.5, 0.2)`
    /// with the scaling `(7, 3)`. Dividing by `h` and multiplying by `sⱼ`
    /// afterwards rounds differently there.
    fn numeric_jacobian_scaling_test() {
        let mut model = F {
            info: SystemInfo::new(1, 2),
        };
        let (x, x_scaling) = ([0.5, 0.2], [7.0, 3.0]);
        let mut fx = [0.0];
        model.eval(&x, &mut fx).unwrap();
        let mut jac = Mat::zeros(1, 2);

        model.derivative(&x, &fx, &x_scaling, jac.as_mut()).unwrap();

        let h = (f64::EPSILON * 20.0).sqrt() * (1.0 + x[0]);
        let mut f_h = [0.0];
        model.eval(&[x[0] + h, x[1]], &mut f_h).unwrap();
        assert!(jac[(0, 0)] == (f_h[0] - fx[0]) * (1.0 / h * x_scaling[0]));
        assert!(jac :approx_eq_fd_tol(): mat![[2.0 * 0.5 * 7.0, 2.0 * 0.2 * 3.0]]);
    }

    #[test]
    /// Test Jacobian evaluation right at upper bound of allowed range.
    fn numeric_jacobian_bounds_test() {
        let mut model = F {
            info: SystemInfo::new(1, 2)
                .with_min([-1.0, -1.0])
                .with_max([1.0, 1.0]),
        };
        // Test at the bound
        assert!(jacobian_at(&mut model, &[1.0, 0.0]).unwrap() :approx_eq_fd_tol(): mat![[2.0, 0.0]]);
        // Test just below the bound
        assert!(jacobian_at(&mut model, &[0.0, 1.0 - 1e-8]).unwrap() :approx_eq_fd_tol(): mat![[0.0, 2.0]]);
        // Test at lower bound
        assert!(jacobian_at(&mut model, &[-1.0, -1.0]).unwrap() :approx_eq_fd_tol(): mat![[-2.0, -2.0]]);
    }

    #[test]
    /// Test system missing max bounds triggering assert
    /// Missing max bounds in `F` must resulting in forward difference used and triggering assert
    fn numeric_jacobian_bounds_panic_test() {
        let mut model = F {
            info: SystemInfo::new(1, 2),
        };

        // Test just below the bound
        let result = jacobian_at(&mut model, &[0.0, 1.0 - 1e-8]);
        assert!(matches!(result, Err(EvalError::Assert)));
    }
}
