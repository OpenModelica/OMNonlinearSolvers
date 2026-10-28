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

use equator::assert;
use faer::utils::approx::{ApproxEq, CwiseMat};
use faer::{ColRef, col, mat};
use openmodelica_total_pivot_solver::{SolveError, SolveStatus, Workspace, solve};

/// Equal up to rounding errors, `128·ε` absolute and relative
fn approx_eq_eps() -> CwiseMat<ApproxEq<f64>> {
    CwiseMat(ApproxEq::eps())
}

#[test]
/// Solve:
/// ```text
/// 2·x₁ + 1·x₂ = 5
/// 1·x₁ + 3·x₂ = 10
/// ```
///represented as
/// ```text
///        x₁   x₂   last column
/// A = [   2    1    -5  ]
///     [   1    3   -10  ]
/// ```
/// Solution: `x = [1, 3]`, with the fixed last entry `[1, 3, 1]`.
fn total_pivot_test_1() {
    let matrix = mat![[2.0, 1.0, -5.0], [1.0, 3.0, -10.0],];

    let mut work = Workspace::new(matrix.nrows());
    let solution = solve(&matrix, Some(2), false, &mut work).unwrap();

    assert!(ColRef::from_slice(solution.x) :approx_eq_eps(): col![1.0, 3.0, 1.0]);
    assert!(solution.rank == 2);
    assert!(solution.fixed_column == 2);
    assert!(matches!(solution.status, SolveStatus::Solved));
}

#[test]
/// Solve a system with zeros, including on the diagonal:
/// ```text
/// 0·x₁ + 2·x₂ + 0·x₃ = 4
/// 3·x₁ + 0·x₂ + 1·x₃ = 6
/// 0·x₁ + 1·x₂ + 4·x₃ = 14
/// ```
/// represented as
/// ```text
///        x₁   x₂   x₃   last column
/// A = [   0    2    0    -4  ]
///     [   3    0    1    -6  ]
///     [   0    1    4   -14  ]
/// ```
/// Solution: `x = [1, 2, 3]`, with the fixed last entry `[1, 2, 3, 1]`.
///
/// Without pivoting the first step would divide by `A[0, 0] = 0`. Total
/// pivoting starts with the largest entry, `4` at `(2, 2)`.
fn total_pivot_test_zeros() {
    let matrix = mat![
        [0.0, 2.0, 0.0, -4.0],
        [3.0, 0.0, 1.0, -6.0],
        [0.0, 1.0, 4.0, -14.0],
    ];

    let mut work = Workspace::new(matrix.nrows());
    let solution = solve(&matrix, Some(3), false, &mut work).unwrap();

    assert!(ColRef::from_slice(solution.x) :approx_eq_eps(): col![1.0, 2.0, 3.0, 1.0]);
    assert!(solution.rank == 3);
    assert!(solution.fixed_column == 3);
    assert!(matches!(solution.status, SolveStatus::Solved));
}

#[test]
/// Example from <https://en.wikipedia.org/wiki/Gaussian_elimination>:
/// ```text
/// 1·x₁ +  3·x₂ + 1·x₃ = 9
/// 1·x₁ +  1·x₂ - 1·x₃ = 1
/// 3·x₁ + 11·x₂ + 5·x₃ = 35
/// ```
/// The matrix has rank 2, the system is consistent and every
/// `x = [-3 + 2·x₃, 4 - x₃, x₃]` is a solution. The unknown without a
/// pivot is set to 0, with the pivot order of the C implementation that is
/// `x₁`, which gives `x = [0, 2.5, 1.5]`. The determinant vanishes, but the
/// system is no casual tearing set, so the status is `Solved`.
fn total_pivot_test_rank_deficient() {
    let matrix = mat![
        [1.0, 3.0, 1.0, -9.0],
        [1.0, 1.0, -1.0, -1.0],
        [3.0, 11.0, 5.0, -35.0],
    ];

    let mut work = Workspace::new(matrix.nrows());
    let solution = solve(&matrix, Some(3), false, &mut work).unwrap();

    assert!(ColRef::from_slice(solution.x) :approx_eq_eps(): col![0.0, 2.5, 1.5, 1.0]);
    assert!(solution.rank == 2);
    assert!(matches!(solution.status, SolveStatus::Solved));
}

#[test]
/// The rank-deficient system from above as a casual tearing set: the solution
/// is the same, but the vanishing determinant is reported.
fn total_pivot_test_rank_deficient_casual() {
    let matrix = mat![
        [1.0, 3.0, 1.0, -9.0],
        [1.0, 1.0, -1.0, -1.0],
        [3.0, 11.0, 5.0, -35.0],
    ];

    let mut work = Workspace::new(matrix.nrows());
    let solution = solve(&matrix, Some(3), true, &mut work).unwrap();

    assert!(ColRef::from_slice(solution.x) :approx_eq_eps(): col![0.0, 2.5, 1.5, 1.0]);
    assert!(matches!(
        solution.status,
        SolveStatus::SolvedDeterminantZero
    ));
}

#[test]
/// A regular casual tearing set: the determinant `5` does not vanish, the
/// status stays `Solved`.
fn total_pivot_test_regular_casual() {
    let matrix = mat![[2.0, 1.0, -5.0], [1.0, 3.0, -10.0],];

    let mut work = Workspace::new(matrix.nrows());
    let solution = solve(&matrix, Some(2), true, &mut work).unwrap();

    assert!(matches!(solution.status, SolveStatus::Solved));
}

#[test]
/// Rank-deficient and inconsistent:
/// ```text
/// x₁ + x₂ = 1
/// x₁ + x₂ = 2
/// ```
fn total_pivot_test_inconsistent() {
    let matrix = mat![[1.0, 1.0, -1.0], [1.0, 1.0, -2.0]];

    let mut work = Workspace::new(matrix.nrows());
    let result = solve(&matrix, Some(2), false, &mut work);

    assert!(matches!(result, Err(SolveError::UnderDetermined)));
}

#[test]
/// Homotopy tangent mode, `pos = None`: one equation `y₁ - 2·y₂ = 0` in two
/// unknowns. The pivot is the largest entry `-2` in column 1, so column 0
/// is the fixed one: `y = [1, 0.5]`.
fn total_pivot_test_tangent_1x2() {
    let matrix = mat![[1.0, -2.0]];

    let mut work = Workspace::new(matrix.nrows());
    let solution = solve(&matrix, None, false, &mut work).unwrap();

    assert!(ColRef::from_slice(solution.x) :approx_eq_eps(): col![1.0, 0.5]);
    assert!(solution.fixed_column == 0);
}

#[test]
/// Homotopy tangent mode, `pos = None`, with two equations in three
/// unknowns:
/// ```text
/// 1·y₁ + 2·y₂ + 3·y₃ = 0
/// 0·y₁ + 1·y₂ + 4·y₃ = 0
/// ```
/// The pivots are in columns 2 and 1, so column 0 is the fixed one:
/// `y = [1, -0.8, 0.2]`.
fn total_pivot_test_tangent_2x3() {
    let matrix = mat![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0]];

    let mut work = Workspace::new(matrix.nrows());
    let solution = solve(&matrix, None, false, &mut work).unwrap();

    assert!(ColRef::from_slice(solution.x) :approx_eq_eps(): col![1.0, -0.8, 0.2]);
    assert!(solution.rank == 2);
    assert!(solution.fixed_column == 0);
}

#[test]
/// A matrix with NaN is rejected before the elimination
fn nan_test() {
    let m = mat![
        [0.0, 1.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, f64::NAN],
        [0.0, 1.0, 4.0, -3.0]
    ];

    let mut work = Workspace::new(m.nrows());
    let result = solve(&m, Some(3), false, &mut work);

    assert!(matches!(result, Err(SolveError::MatrixNotFinite)));
}

#[test]
/// A workspace is reused for the next system: the permutations of the first
/// solve don't leak into the second
fn workspace_reuse_test() {
    let mut work = Workspace::new(2);
    let first = mat![[0.0, 1.0, -2.0], [1.0, 0.0, -3.0]];
    let second = mat![[1.0, 0.0, -3.0], [0.0, 1.0, -2.0]];

    let x_first = solve(&first, Some(2), false, &mut work).unwrap().x.to_vec();
    let solution = solve(&second, Some(2), false, &mut work).unwrap();

    assert!(ColRef::from_slice(&x_first) :approx_eq_eps(): col![3.0, 2.0, 1.0]);
    assert!(ColRef::from_slice(solution.x) :approx_eq_eps(): col![3.0, 2.0, 1.0]);
}
