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

use crate::solve::SolveError;
use openmodelica_permuted_matrix::PermutedMatrix;

/// Back substitution on a matrix after [`forward_elimination`].
///
/// Solves the upper triangular system in logical order from the last row up
/// and writes `x` with `n+1` entries in the original (physical) column order.
/// The entry of the fixed column (logical column `n`) is set to 1.
///
/// Rows from `rank` on have no pivot. If their right-hand side is (almost)
/// zero, `|rhs| <= 1e-6`, the system is consistent and the unknown of that row
/// is set to 0, otherwise the system has no solution.
///
/// # Errors
///
/// Returns [`SolveError::UnderDetermined`] if a row without pivot has a
/// right-hand side larger than `1e-6`.
///
/// # Panics
///
/// Panics if `x` doesn't have one entry per column of `matrix`.
///
/// [`forward_elimination`]: crate::forward_elimination::forward_elimination
pub(crate) fn backward_elimination(
    matrix: &PermutedMatrix<'_, f64>,
    rank: usize,
    x: &mut [f64],
) -> Result<(), SolveError> {
    let n_rows = matrix.nrows(); // n
    assert_eq!(x.len(), matrix.ncols(), "Solution has wrong length");

    for i in (0..n_rows).rev() {
        if i >= rank {
            if matrix[(i, n_rows)].abs() > 1e-6 {
                return Err(SolveError::UnderDetermined);
            } else {
                let col_index_i = matrix.column_permutation()[i];
                x[col_index_i] = 0.0;
            }
        } else {
            let col_index_i = matrix.column_permutation()[i];
            x[col_index_i] = -matrix[(i, n_rows)];

            for j in (i + 1..n_rows).rev() {
                let col_index_j = matrix.column_permutation()[j];
                x[col_index_i] -= matrix[(i, j)] * x[col_index_j];
            }
            x[col_index_i] /= matrix[(i, i)];
        }
    }
    let col_index_n = matrix.column_permutation()[n_rows];
    x[col_index_n] = 1.0;

    Ok(())
}

#[cfg(test)]
mod backward_elimination_tests {
    use super::*;
    use equator::assert;
    use faer::mat;
    use faer::utils::approx::{ApproxEq, CwiseMat};
    use faer::{Col, ColRef};

    use openmodelica_permuted_matrix::{Permutations, PermutedMatrix};

    /// Equal up to rounding errors, `128·ε` absolute and relative
    fn approx_eq_eps() -> CwiseMat<ApproxEq<f64>> {
        CwiseMat(ApproxEq::eps())
    }

    #[test]
    fn backward_elimination_test() {
        let m = mat![
            [1.0, 3.0, 1.0, -9.0],
            [0.0, -2.0, -2.0, 8.0],
            [0.0, 0.0, 1.0, 2.0]
        ];
        let mut permutations = Permutations::new(m.nrows(), m.ncols());
        let mut values = crate::values_of(&m);
        let p = PermutedMatrix::new(&mut values, &mut permutations);

        let mut x = [0.0; 4];
        backward_elimination(&p, 3, &mut x).unwrap();

        let residual = &m * ColRef::from_slice(&x);
        assert!(residual :approx_eq_eps(): Col::<f64>::zeros(3));
    }

    #[test]
    /// Rank-deficient but consistent: the Wikipedia example
    /// <https://en.wikipedia.org/wiki/Gaussian_elimination> after the forward
    /// elimination, with rank 2. The last row is zero, including the right-hand
    /// side.
    ///
    /// The unknown without a pivot, `x₃`, is set to 0. That gives
    /// `x = [-3, 4, 0]`, the solution `[-3 + 2·x₃, 4 - x₃, x₃]` of the
    /// original system for `x₃ = 0`.
    fn backward_elimination_rank_deficient_test() {
        let m = mat![
            [1.0, 3.0, 1.0, -9.0],
            [0.0, -2.0, -2.0, 8.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        let mut permutations = Permutations::new(m.nrows(), m.ncols());
        let mut values = crate::values_of(&m);
        let p = PermutedMatrix::new(&mut values, &mut permutations);

        let mut x = [0.0; 4];
        backward_elimination(&p, 2, &mut x).unwrap();

        assert!(x == [-3.0, 4.0, 0.0, 1.0]);

        // x also solves the original system before the elimination
        let original = mat![
            [1.0, 3.0, 1.0, -9.0],
            [1.0, 1.0, -1.0, -1.0],
            [3.0, 11.0, 5.0, -35.0]
        ];
        let residual = &original * ColRef::from_slice(&x);
        assert!(residual :approx_eq_eps(): Col::<f64>::zeros(3));
    }

    #[test]
    /// Rank-deficient and inconsistent: the same matrix, but the zero row has
    /// a right-hand side of 5. Its equation reads `0 = 5`, so there is no
    /// solution.
    fn backward_elimination_under_determined_test() {
        let m = mat![
            [1.0, 3.0, 1.0, -9.0],
            [0.0, -2.0, -2.0, 8.0],
            [0.0, 0.0, 0.0, 5.0]
        ];
        let mut permutations = Permutations::new(m.nrows(), m.ncols());
        let mut values = crate::values_of(&m);
        let p = PermutedMatrix::new(&mut values, &mut permutations);

        let result = backward_elimination(&p, 2, &mut [0.0; 4]);

        assert!(matches!(result, Err(SolveError::UnderDetermined)));
    }
}
