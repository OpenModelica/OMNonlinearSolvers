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

//! Tests for the determinant `solve` computes as the product of the pivots.
//!
//! The elimination itself is stable, but the product of `n` harmless pivots
//! can leave the range of `f64`. `solve` handles it like C: a NaN determinant
//! is an error, an overflow to infinity is ignored.

use faer::Mat;
use openmodelica_total_pivot_solver::{SolveStatus, Workspace, solve};

/// Diagonal `n×(n+1)` system with the given diagonal and a zero last
/// column
fn diagonal_system(diagonal: &[f64]) -> Mat<f64> {
    let n = diagonal.len();
    Mat::from_fn(n, n + 1, |i, j| if i == j { diagonal[i] } else { 0.0 })
}

/// Assert that a log record with `level` and target `NLS` was written
fn assert_logged(logs: &[testing_logger::CapturedLog], level: log::Level) {
    assert!(
        logs.iter().any(|l| l.level == level && l.target == "NLS"),
        "no {level} log with target NLS, logs (first line each): {:?}",
        logs.iter()
            .map(|l| (l.level, &l.target, l.body.lines().next().unwrap_or("")))
            .collect::<Vec<_>>()
    );
}

#[test]
/// 100 pivots of `1e4`: the determinant `1e400` overflows to infinity,
/// although the system is perfectly well-conditioned. Like C, `solve` doesn't
/// check for an infinite determinant: the solution doesn't depend on it, so
/// the system is solved normally, without warning or error.
fn determinant_overflow_test() {
    testing_logger::setup();
    let matrix = diagonal_system(&[1e4; 100]);

    let mut work = Workspace::new(matrix.nrows());
    let solution = solve(&matrix, Some(100), false, &mut work).unwrap();

    // The right-hand side is zero, so all unknowns are zero
    let mut expected = vec![0.0; 101];
    expected[100] = 1.0;
    assert_eq!(*solution.x, *expected);
    assert!(matches!(solution.status, SolveStatus::Solved));
    testing_logger::validate(|logs| {
        assert!(
            !logs.iter().any(|l| l.level <= log::Level::Warn),
            "an overflow must not be logged as warning or error"
        );
    });
}

#[test]
/// Two huge pivots and a zero row: the product overflows to infinity
/// after the two pivots, the zero diagonal entry beyond the rank then
/// gives `∞ · 0 = NaN`. `solve` logs an error and fails, like C.
fn determinant_nan_test() {
    testing_logger::setup();
    let matrix = diagonal_system(&[1e200, 1e200, 0.0]);

    let mut work = Workspace::new(matrix.nrows());
    let result = solve(&matrix, Some(3), false, &mut work);

    assert!(result.is_err(), "solve must fail for a NaN determinant");
    testing_logger::validate(|logs| assert_logged(logs, log::Level::Error));
}
