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

//! The homotopy functions `H(x, λ)` that deform a system `F(x) = 0` into a
//! system that is easy to solve at `λ = 0` and is `F` again at `λ = 1`.
//!
//! Each is a model itself, with `λ` as an additional, last unknown, so the
//! path following can treat all of them alike. A model with its own
//! homotopy, the `homotopy()` operator of Modelica, needs no wrapper: it
//! already has `λ` as unknown.

use faer::MatMut;
use faer::reborrow::ReborrowMut;
use openmodelica_nls_model::{EvalError, ModelFunctions, SystemInfo};

/// The Newton homotopy `H(x, λ) = F(x) − (1 − λ)·F(x₀)`.
///
/// Its solution at `λ = 0` is the start point `x₀`. From
/// `wrapper_fvec_homotopy_newton` and `wrapper_fvec_homotopy_newton_der` of
/// the C runtime.
pub struct NewtonHomotopy<'m, M> {
    model: EvaluatedModel<'m, M>,
    /// `F(x₀)` at the start point
    fx0: Box<[f64]>,
    /// The model's info with `λ` as additional unknown
    info: SystemInfo,
}

impl<'m, M: ModelFunctions> NewtonHomotopy<'m, M> {
    /// Creates the Newton homotopy of `model` from the residual `fx0 = F(x₀)`
    /// at the start point.
    ///
    /// # Panics
    ///
    /// Panics if `fx0` doesn't have one entry per residual of `model`.
    pub fn new(model: &'m mut M, fx0: &[f64]) -> Self {
        assert_eq!(
            fx0.len(),
            model.info().num_residuals,
            "Start residual has wrong length"
        );
        NewtonHomotopy {
            info: with_lambda(model.info()),
            model: EvaluatedModel::new(model),
            fx0: fx0.into(),
        }
    }
}

impl<M: ModelFunctions> ModelFunctions for NewtonHomotopy<'_, M> {
    fn info(&self) -> &SystemInfo {
        &self.info
    }

    fn eval(&mut self, y: &[f64], h: &mut [f64]) -> Result<(), EvalError> {
        let (x, lambda) = split(y);
        let f = self.model.eval(x)?;
        for ((h, f), fx0) in h.iter_mut().zip(f).zip(&self.fx0) {
            *h = f - (1.0 - lambda) * fx0;
        }
        Ok(())
    }

    /// `[∂H/∂x | ∂H/∂λ]·S = [J(x)·Sₓ | F(x₀)·s_λ]`
    fn derivative(
        &mut self,
        y: &[f64],
        _hy: &[f64],
        y_scaling: &[f64],
        jacobian: MatMut<'_, f64>,
    ) -> Result<(), EvalError> {
        let (x, _) = split(y);
        let (x_scaling, lambda_scaling) = split(y_scaling);
        let (jx, jlambda) = jacobian.split_at_col_mut(x.len());
        self.model.derivative(x, x_scaling, jx)?;
        for (dh, fx0) in jlambda.col_mut(0).iter_mut().zip(&self.fx0) {
            *dh = fx0 * lambda_scaling;
        }
        Ok(())
    }
}

/// The fixpoint homotopy `H(x, λ) = λ·F(x) + (1 − λ)·(x − x₀)`.
///
/// Its solution at `λ = 0` is the start point `x₀`. It needs as many
/// residuals as unknowns. From `wrapper_fvec_homotopy_fixpoint` and
/// `wrapper_fvec_homotopy_fixpoint_der` of the C runtime.
///
/// Its Jacobian reproduces a scaling of the C runtime: C scales the columns
/// of `J` by the variable scaling `S` before adding the identity, so the path
/// following solves with `λ·J·S + (1 − λ)·I` instead of the consistently
/// scaled `(λ·J + (1 − λ)·I)·S`. The two only agree where `S = I`.
pub struct FixpointHomotopy<'m, M> {
    model: EvaluatedModel<'m, M>,
    /// The start point `x₀`
    x0: Box<[f64]>,
    /// The model's info with `λ` as additional unknown
    info: SystemInfo,
}

impl<'m, M: ModelFunctions> FixpointHomotopy<'m, M> {
    /// Creates the fixpoint homotopy of `model` from the start point `x0`.
    ///
    /// # Panics
    ///
    /// Panics if `model` doesn't have as many residuals as unknowns, or `x0`
    /// doesn't have one entry per unknown.
    pub fn new(model: &'m mut M, x0: &[f64]) -> Self {
        let info = model.info();
        assert_eq!(
            info.num_residuals, info.num_unknowns,
            "Fixpoint homotopy needs a square system"
        );
        assert_eq!(x0.len(), info.num_unknowns, "Start point has wrong length");
        FixpointHomotopy {
            info: with_lambda(info),
            model: EvaluatedModel::new(model),
            x0: x0.into(),
        }
    }
}

impl<M: ModelFunctions> ModelFunctions for FixpointHomotopy<'_, M> {
    fn info(&self) -> &SystemInfo {
        &self.info
    }

    fn eval(&mut self, y: &[f64], h: &mut [f64]) -> Result<(), EvalError> {
        let (x, lambda) = split(y);
        let f = self.model.eval(x)?;
        for (((h, f), x), x0) in h.iter_mut().zip(f).zip(x).zip(&self.x0) {
            *h = lambda * f + (1.0 - lambda) * (x - x0);
        }
        Ok(())
    }

    /// `[λ·J(x)·Sₓ + (1 − λ)·I | (F(x) − (x − x₀))·s_λ]`, C's scaled
    /// Jacobian with the identity part unscaled, see [`FixpointHomotopy`].
    fn derivative(
        &mut self,
        y: &[f64],
        _hy: &[f64],
        y_scaling: &[f64],
        jacobian: MatMut<'_, f64>,
    ) -> Result<(), EvalError> {
        let (x, lambda) = split(y);
        let (x_scaling, lambda_scaling) = split(y_scaling);
        let (mut jx, jlambda) = jacobian.split_at_col_mut(x.len());
        self.model.derivative(x, x_scaling, jx.rb_mut())?;

        jx *= faer::Scale(lambda);
        for d in jx.diagonal_mut().column_vector_mut().iter_mut() {
            *d += 1.0 - lambda;
        }
        let f = self.model.f();
        for (((dh, f), x), x0) in jlambda.col_mut(0).iter_mut().zip(f).zip(x).zip(&self.x0) {
            *dh = (f - (x - x0)) * lambda_scaling;
        }
        Ok(())
    }
}

/// A model together with the point it was last evaluated at, so that the
/// Jacobian of a homotopy doesn't evaluate `F` again at the same point.
struct EvaluatedModel<'m, M> {
    model: &'m mut M,
    /// The point of the latest successful evaluation, NaN before the first
    x: Box<[f64]>,
    /// `F(x)` at that point
    f: Box<[f64]>,
}

impl<'m, M: ModelFunctions> EvaluatedModel<'m, M> {
    fn new(model: &'m mut M) -> Self {
        let info = model.info();
        EvaluatedModel {
            x: vec![f64::NAN; info.num_unknowns].into(),
            f: vec![0.0; info.num_residuals].into(),
            model,
        }
    }

    /// Evaluates `F(x)`.
    fn eval(&mut self, x: &[f64]) -> Result<&[f64], EvalError> {
        self.x.fill(f64::NAN);
        self.model.eval(x, &mut self.f)?;
        self.x.copy_from_slice(x);
        Ok(&self.f)
    }

    /// Evaluates the scaled Jacobian `J(x)·diag(x_scaling)`, and `F(x)` first
    /// unless `x` is the point of the latest evaluation.
    fn derivative(
        &mut self,
        x: &[f64],
        x_scaling: &[f64],
        jacobian: MatMut<'_, f64>,
    ) -> Result<(), EvalError> {
        if *self.x != *x {
            self.eval(x)?;
        }
        self.model.derivative(x, &self.f, x_scaling, jacobian)
    }

    /// `F(x)` of the latest evaluation.
    fn f(&self) -> &[f64] {
        &self.f
    }
}

/// Splits `y = (x, λ)` into `x` and `λ`.
fn split(y: &[f64]) -> (&[f64], f64) {
    let (lambda, x) = y.split_last().expect("y contains λ");
    (x, *lambda)
}

/// The info of a homotopy of a system with `info`: `λ` is an additional,
/// last unknown with nominal value 1 and no bounds.
fn with_lambda(info: &SystemInfo) -> SystemInfo {
    let append = |values: &[f64], lambda: f64| -> Box<[f64]> {
        values.iter().copied().chain([lambda]).collect()
    };
    let mut lifted = SystemInfo::new(info.num_residuals, info.num_unknowns + 1)
        .with_nominal(append(&info.nominal, 1.0))
        .with_min(append(&info.min, f64::NEG_INFINITY))
        .with_max(append(&info.max, f64::INFINITY));
    lifted.casual_tearing_set = info.casual_tearing_set;
    lifted
}

#[cfg(test)]
mod homotopy_function_tests {
    use super::*;
    use faer::Mat;
    use openmodelica_nls_model::ClosureModel;
    use std::cell::Cell;

    /// The start point of the homotopies
    const X0: [f64; 2] = [2.0, 1.0];

    /// `F(x) = (x₀² + x₁ − 3, x₀·x₁ − 2)` with the root `(1, 2)`, counting its
    /// evaluations in `calls`
    fn model(calls: &Cell<usize>) -> impl ModelFunctions + '_ {
        ClosureModel::new(2, 2, move |x: &[f64], res: &mut [f64]| {
            calls.set(calls.get() + 1);
            res[0] = x[0] * x[0] + x[1] - 3.0;
            res[1] = x[0] * x[1] - 2.0;
            Ok(())
        })
    }

    /// `F` of [`model`] at `x`
    fn f(x: &[f64]) -> [f64; 2] {
        let mut res = [0.0; 2];
        model(&Cell::new(0)).eval(x, &mut res).unwrap();
        res
    }

    /// `H(y)` of `homotopy`
    fn h(homotopy: &mut impl ModelFunctions, y: &[f64]) -> [f64; 2] {
        let mut res = [0.0; 2];
        homotopy.eval(y, &mut res).unwrap();
        res
    }

    /// The Jacobian of `homotopy` at `y`, evaluated after `H(y)` as the
    /// solvers do
    fn jacobian(homotopy: &mut impl ModelFunctions, y: &[f64]) -> Mat<f64> {
        let hy = h(homotopy, y);
        let mut jac = Mat::zeros(2, 3);
        homotopy
            .derivative(y, &hy, &[1.0; 3], jac.as_mut())
            .unwrap();
        jac
    }

    /// The Jacobian of `homotopy` at `y` by central differences
    fn central_differences(homotopy: &mut impl ModelFunctions, y: &[f64]) -> Mat<f64> {
        let step = 1e-6;
        let mut jac = Mat::zeros(2, 3);
        for j in 0..3 {
            let mut y_plus = y.to_vec();
            let mut y_minus = y.to_vec();
            y_plus[j] += step;
            y_minus[j] -= step;
            let (h_plus, h_minus) = (h(homotopy, &y_plus), h(homotopy, &y_minus));
            for i in 0..2 {
                jac[(i, j)] = (h_plus[i] - h_minus[i]) / (2.0 * step);
            }
        }
        jac
    }

    /// Whether all entries of `a` and `b` differ by at most `tol`
    fn close(a: &Mat<f64>, b: &Mat<f64>, tol: f64) -> bool {
        (0..a.nrows()).all(|i| (0..a.ncols()).all(|j| (a[(i, j)] - b[(i, j)]).abs() <= tol))
    }

    #[test]
    /// `H(x₀, 0) = 0`: the start point solves the Newton homotopy at `λ = 0`
    fn newton_start_test() {
        let calls = Cell::new(0);
        let mut model = model(&calls);
        let mut homotopy = NewtonHomotopy::new(&mut model, &f(&X0));

        assert_eq!(h(&mut homotopy, &[X0[0], X0[1], 0.0]), [0.0, 0.0]);
    }

    #[test]
    /// `H(x, 1) = F(x)`
    fn newton_end_test() {
        let calls = Cell::new(0);
        let mut model = model(&calls);
        let mut homotopy = NewtonHomotopy::new(&mut model, &f(&X0));

        assert_eq!(h(&mut homotopy, &[0.5, -1.0, 1.0]), f(&[0.5, -1.0]));
    }

    #[test]
    /// `[J(x) | F(x₀)]` matches central differences of `H`
    fn newton_derivative_test() {
        let calls = Cell::new(0);
        let mut model = model(&calls);
        let mut homotopy = NewtonHomotopy::new(&mut model, &f(&X0));
        let y = [1.5, 0.5, 0.3];

        let jac = jacobian(&mut homotopy, &y);
        assert!(
            close(&jac, &central_differences(&mut homotopy, &y), 1e-6),
            "{jac:?}"
        );
        assert_eq!((jac[(0, 2)], jac[(1, 2)]), (2.0, 0.0), "∂H/∂λ = F(x₀)");
    }

    #[test]
    /// `H(x, 0) = x − x₀`, so the start point solves the fixpoint homotopy at
    /// `λ = 0`
    fn fixpoint_start_test() {
        let calls = Cell::new(0);
        let mut model = model(&calls);
        let mut homotopy = FixpointHomotopy::new(&mut model, &X0);

        assert_eq!(h(&mut homotopy, &[X0[0], X0[1], 0.0]), [0.0, 0.0]);
        assert_eq!(h(&mut homotopy, &[3.0, -1.0, 0.0]), [1.0, -2.0]);
    }

    #[test]
    /// `H(x, 1) = F(x)`
    fn fixpoint_end_test() {
        let calls = Cell::new(0);
        let mut model = model(&calls);
        let mut homotopy = FixpointHomotopy::new(&mut model, &X0);

        assert_eq!(h(&mut homotopy, &[0.5, -1.0, 1.0]), f(&[0.5, -1.0]));
    }

    #[test]
    /// Without scaling, `[λ·J(x) + (1 − λ)·I | F(x) − (x − x₀)]` matches
    /// central differences of `H`
    fn fixpoint_derivative_test() {
        let calls = Cell::new(0);
        let mut model = model(&calls);
        let mut homotopy = FixpointHomotopy::new(&mut model, &X0);
        let y = [1.5, 0.5, 0.3];

        let jac = jacobian(&mut homotopy, &y);
        assert!(
            close(&jac, &central_differences(&mut homotopy, &y), 1e-6),
            "{jac:?}"
        );
    }

    #[test]
    /// With the scaling `S = diag(2, 4, 1)` the Jacobian is C's
    /// `λ·J·S + (1 − λ)·I`
    fn fixpoint_derivative_scaling_test() {
        let calls = Cell::new(0);
        let mut model = model(&calls);
        let mut homotopy = FixpointHomotopy::new(&mut model, &X0);
        let y = [1.5, 0.5, 0.25];

        let hy = h(&mut homotopy, &y);
        let mut jac = Mat::zeros(2, 3);
        homotopy
            .derivative(&y, &hy, &[2.0, 4.0, 1.0], jac.as_mut())
            .unwrap();

        // J(1.5, 0.5) = [[3, 1], [0.5, 1.5]], λ = 0.25
        let expected = [
            [0.25 * 3.0 * 2.0 + 0.75, 0.25 * 1.0 * 4.0],
            [0.25 * 0.5 * 2.0, 0.25 * 1.5 * 4.0 + 0.75],
        ];
        for (i, row) in expected.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert!((jac[(i, j)] - value).abs() < 1e-6, "({i}, {j}): {jac:?}");
            }
        }
    }

    #[test]
    #[should_panic = "Fixpoint homotopy needs a square system"]
    fn fixpoint_not_square_test() {
        let mut model = ClosureModel::new(1, 2, |_: &[f64], _: &mut [f64]| Ok(()));
        let _ = FixpointHomotopy::new(&mut model, &[0.0, 0.0]);
    }

    #[test]
    /// `λ` is an additional, last unknown with nominal value 1 and no bounds,
    /// the rest of the model's info is kept
    fn info_test() {
        let mut model = ClosureModel::new(2, 2, |_: &[f64], _: &mut [f64]| Ok(()))
            .with_nominal(vec![10.0, 20.0])
            .with_min(vec![0.0, -1.0])
            .with_casual_tearing_set();
        let homotopy = NewtonHomotopy::new(&mut model, &[0.0, 0.0]);
        let info = homotopy.info();

        assert_eq!((info.num_residuals, info.num_unknowns), (2, 3));
        assert_eq!(*info.nominal, [10.0, 20.0, 1.0]);
        assert_eq!(*info.min, [0.0, -1.0, f64::NEG_INFINITY]);
        assert_eq!(*info.max, [f64::INFINITY; 3]);
        assert!(info.casual_tearing_set);
    }

    #[test]
    /// The Jacobian reuses `F` of the latest evaluation at the same point, and
    /// evaluates `F` first at any other point
    fn reuse_evaluation_test() {
        let calls = Cell::new(0);
        let mut model = model(&calls);
        let mut homotopy = NewtonHomotopy::new(&mut model, &[0.0, 0.0]);
        let mut jac = Mat::zeros(2, 3);
        let y = [1.5, 0.5, 0.3];

        h(&mut homotopy, &y);
        calls.set(0);
        homotopy
            .derivative(&y, &[0.0; 2], &[1.0; 3], jac.as_mut())
            .unwrap();
        // One evaluation per column of the forward differences
        assert_eq!(calls.get(), 2, "F reused");

        calls.set(0);
        homotopy
            .derivative(&[1.0, 1.0, 0.3], &[0.0; 2], &[1.0; 3], jac.as_mut())
            .unwrap();
        assert_eq!(calls.get(), 3, "F evaluated first");
    }

    #[test]
    /// A failing model fails the homotopy and isn't reused afterwards
    fn model_error_test() {
        let mut model = ClosureModel::new(1, 1, |x: &[f64], res: &mut [f64]| {
            if x[0] < 0.0 {
                return Err(EvalError::Assert);
            }
            res[0] = x[0];
            Ok(())
        });
        let mut homotopy = FixpointHomotopy::new(&mut model, &[1.0]);
        let mut res = [0.0];
        let mut jac = Mat::zeros(1, 2);

        assert!(matches!(
            homotopy.eval(&[-1.0, 0.5], &mut res),
            Err(EvalError::Assert)
        ));
        assert!(matches!(
            homotopy.derivative(&[-1.0, 0.5], &res, &[1.0; 2], jac.as_mut()),
            Err(EvalError::Assert)
        ));
    }
}
