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

use crate::pivot_search::pivot_search;
use openmodelica_permuted_matrix::PermutedMatrix;

/// Forward elimination with complete pivoting
///
/// Brings `matrix` to upper triangular form in logical order. Pivots are
/// searched in the first `n_pivot_columns` logical columns only, see
/// [`pivot_search`]. Stops early if the remaining part is (numerically) zero.
///
/// Returns the rank of `matrix`.
pub(crate) fn forward_elimination(
    matrix: &mut PermutedMatrix<'_, f64>,
    n_pivot_columns: usize,
) -> usize {
    let n_rows = matrix.nrows(); // n
    let mut rank: usize = n_rows;

    for stage in 0..n_rows {
        let pivot = pivot_search(matrix, stage, n_pivot_columns);

        if pivot.abs_value < f64::EPSILON {
            rank = stage;
            break;
        }

        // Swap
        matrix.swap_rows(stage, pivot.row);
        matrix.swap_columns(stage, pivot.column);

        // Gauss elimination
        eliminate_below_pivot(matrix, stage);
    }

    rank
}

/// Single forward elimination step
///
/// Eliminates the entries below the pivot `(i,i)` by subtracting multiples of
/// row `i`.
fn eliminate_below_pivot(matrix: &mut PermutedMatrix<'_, f64>, i: usize) {
    for k in i + 1..matrix.nrows() {
        let multiplier = -matrix[(k, i)] / matrix[(i, i)];
        let ncols = matrix.ncols();
        matrix.add_scaled_row(k, i, multiplier, i + 1..ncols);

        matrix[(k, i)] = 0.0;
    }
}

#[cfg(test)]
mod forward_elimination_tests {
    use super::*;
    use equator::assert;
    use faer::mat;
    use faer::utils::approx::{ApproxEq, CwiseMat};

    use openmodelica_permuted_matrix::{Permutations, PermutedMatrix};

    /// Equal up to rounding errors, `128·ε` absolute and relative
    fn approx_eq_eps() -> CwiseMat<ApproxEq<f64>> {
        CwiseMat(ApproxEq::eps())
    }

    #[test]
    /// Example from Wiki: https://en.wikipedia.org/wiki/Gaussian_elimination
    fn gauss_elimination_test() {
        let m = mat![
            [1.0, 3.0, 1.0, -9.0],
            [1.0, 1.0, -1.0, -1.0],
            [3.0, 11.0, 5.0, -35.0]
        ];
        let mut permutations = Permutations::new(m.nrows(), m.ncols());
        let mut values = crate::values_of(&m);
        let mut p = PermutedMatrix::new(&mut values, &mut permutations);

        // Step 0: subtract 1·row 0 from row 1 and 3·row 0 from row 2
        let expected_result = mat![
            [1.0, 3.0, 1.0, -9.0],
            [0.0, -2.0, -2.0, 8.0],
            [0.0, 2.0, 2.0, -8.0]
        ];
        eliminate_below_pivot(&mut p, 0);
        assert_eq!(p.to_mat(), expected_result, "Wrong result after step 0");

        // Step 1: add row 1 to row 2, which becomes zero (the matrix has rank 2)
        let expected_result = mat![
            [1.0, 3.0, 1.0, -9.0],
            [0.0, -2.0, -2.0, 8.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        eliminate_below_pivot(&mut p, 1);
        assert_eq!(p.to_mat(), expected_result, "Wrong result after step 1");

        // Step 2: no rows below row 2, nothing changes
        eliminate_below_pivot(&mut p, 2);
        assert_eq!(p.to_mat(), expected_result, "Wrong result after step 2");
    }

    #[test]
    /// Forward elimination of
    /// ```text
    /// 0·x₁ + 2·x₂ + 0·x₃ = 4
    /// 3·x₁ + 0·x₂ + 1·x₃ = 6
    /// 0·x₁ + 1·x₂ + 4·x₃ = 14
    /// ```
    /// with the last column `-b`. All four columns are pivot candidates, as in
    /// the C implementation with `pos = -1`.
    ///
    /// | Stage | Pivot                   | Swaps                     |
    /// |-------|-------------------------|---------------------------|
    /// | 0     | `-14` at `(2, 3)`       | rows 0 ↔ 2, columns 0 ↔ 3 |
    /// | 1     | `3` at logical `(1, 1)` | none                      |
    /// | 2     | `12/7` at `(2, 2)`      | none                      |
    ///
    /// The result is upper triangular in logical order:
    /// ```text
    /// [ -14   0     1     4  ]
    /// [   0   3   -3/7  -5/7 ]
    /// [   0   0   12/7  -8/7 ]
    /// ```
    fn forward_elimination_test() {
        let m = mat![
            [0.0, 2.0, 0.0, -4.0],
            [3.0, 0.0, 1.0, -6.0],
            [0.0, 1.0, 4.0, -14.0]
        ];
        let n_pivot_columns = m.ncols();
        let mut permutations = Permutations::new(m.nrows(), m.ncols());
        let mut values = crate::values_of(&m);
        let mut p = PermutedMatrix::new(&mut values, &mut permutations);

        let rank = forward_elimination(&mut p, n_pivot_columns);

        assert_eq!(rank, 3, "Wrong rank");
        assert_eq!(p.row_permutation(), &[2, 1, 0], "Wrong row permutation");
        assert_eq!(
            p.column_permutation(),
            &[3, 0, 1, 2],
            "Wrong column permutation"
        );

        let expected_result = mat![
            [-14.0, 0.0, 1.0, 4.0],
            [0.0, 3.0, -3.0 / 7.0, -5.0 / 7.0],
            [0.0, 0.0, 12.0 / 7.0, -8.0 / 7.0]
        ];
        assert!(p.to_mat() :approx_eq_eps(): expected_result);
    }
}
