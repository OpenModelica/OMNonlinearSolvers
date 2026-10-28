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

//! Tests of the scaling functions for variables, residuals and the rows of a
//! linear system, and of the errors measured with them.

use equator::assert;
use faer::mat;
use faer::utils::approx::{ApproxEq, CwiseMat};
use openmodelica_nls_model::scaling::{
    make_finite, residuals, rows, scaled_squared_error, squared_error, unscale, variables,
};

/// Equal up to rounding errors, `128·ε` absolute and relative
fn approx_eq_eps() -> CwiseMat<ApproxEq<f64>> {
    CwiseMat(ApproxEq::eps())
}

#[test]
/// Each unknown is scaled with its nominal value or its magnitude, whichever
/// is larger.
fn variable_scaling_test() {
    let nominal = vec![1.0, 100.0, 1.0];
    let x = [-10.0, 25.0, 10.0];
    let mut factor = vec![0.0; 3];

    variables(&nominal, &x, &mut factor);
    assert_eq!(factor, [10.0, 100.0, 10.0]);
}

#[test]
/// Test that for a n×(n+1) matrix the extra column is ignored.
/// Otherwise the L1 norm on the rows would result in [13, 27].
fn residual_scaling_test() {
    let m = mat![[1.0, -2.0, 10.0], [-3.0, 4.0, 20.0]];
    let mut factor = vec![1.0; 2];

    residuals(m.as_ref(), &mut factor);
    assert_eq!(factor, [3.0, 7.0]);
}

#[test]
/// Test scaling on 3×4 matrix.
/// Last column shouldn't affect the scaling factor
fn row_scaling_test() {
    let mut m = mat![
        [0.0, 0.0, 0.0, 10.0],
        [1.0, 2.0, 4.0, -3.0],
        [5.0, -6.0, 7.0, -8.0]
    ];

    rows(m.as_mut(), &mut [0.0; 3]);
    assert!(m :approx_eq_eps(): mat![[0.0, 0.0, 0.0, 10.0], [1.0/4.0, 2.0/4.0, 4.0/4.0, -3.0/4.0], [5.0/7.0, -6.0/7.0, 7.0/7.0, -8.0/7.0]]);
}

#[test]
/// Rows are divided by their maximum like in C. Multiplying by its reciprocal
/// rounds differently: `5.0 * (1.0 / 7.0) != 5.0 / 7.0`.
fn row_scaling_divides_test() {
    let mut m = mat![[7.0, 5.0]];

    rows(m.as_mut(), &mut [0.0; 1]);
    assert!(m[(0, 1)] == 5.0 / 7.0);
}

#[test]
/// A NaN is no maximum, like in `scaleMatrixRows`: the row is divided by the
/// largest of the other entries of `J`
fn row_scaling_nan_test() {
    let mut m = mat![[f64::NAN, 2.0, 4.0], [1.0, 1.0, 1.0]];

    rows(m.as_mut(), &mut [0.0; 2]);
    assert!(m[(0, 0)].is_nan() && m[(0, 1)] == 1.0 && m[(0, 2)] == 2.0);
}

#[test]
#[should_panic = "Matrix has wrong shape"]
/// `rows` needs the right-hand side as last column, passing the square
/// Jacobian `J` alone is a mistake
fn row_scaling_square_matrix_test() {
    let mut m = mat![[1.0, 2.0], [3.0, 4.0]];

    rows(m.as_mut(), &mut [0.0; 2]);
}

#[test]
/// Multiplying by the absolute factor undoes the column scaling, a factor of
/// zero keeps the entry
fn unscale_test() {
    let mut x = [0.0; 3];

    unscale(&[1.0, 2.0, 3.0], &[2.0, -3.0, 0.0], &mut x);
    assert_eq!(x, [2.0, 6.0, 3.0]);
}

#[test]
/// Infinite and NaN factors become 1, finite ones stay
fn make_finite_test() {
    let mut factor = [2.0, f64::INFINITY, f64::NAN, f64::NEG_INFINITY, 0.0];

    make_finite(&mut factor);
    assert_eq!(factor, [2.0, 1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn squared_error_test() {
    assert_eq!(squared_error(&[3.0, -4.0]), 25.0);
}

#[test]
/// Entries are divided by the absolute scaling, a scaling of zero keeps the
/// entry unscaled
fn scaled_squared_error_test() {
    assert_eq!(
        scaled_squared_error(&[2.0, 3.0, 5.0], &[2.0, -3.0, 0.0]),
        27.0
    );
}
