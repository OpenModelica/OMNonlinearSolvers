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

use faer::Mat;
use log::{error, trace, warn};
use openmodelica_permuted_matrix::{Permutations, PermutedMatrix};

use crate::backward_elimination::backward_elimination;
use crate::forward_elimination::forward_elimination;

/// Buffers of [`solve`] for `n×(n+1)` systems: the matrix it eliminates, the
/// permutations of the pivot search and the solution.
///
/// Allocated once and reused, so a solve doesn't allocate.
#[derive(Debug)]
pub struct Workspace {
    /// The matrix, column after column
    values: Box<[f64]>,
    permutations: Permutations,
    x: Box<[f64]>,
}

impl Workspace {
    /// Allocates the buffers for `n×(n+1)` systems.
    pub fn new(n: usize) -> Self {
        Workspace {
            values: vec![0.0; n * (n + 1)].into(),
            permutations: Permutations::new(n, n + 1),
            x: vec![0.0; n + 1].into(),
        }
    }
}

/// Result of a successful [`solve`].
#[derive(Debug)]
pub struct Solution<'w> {
    /// Solution, `n+1` entries, the entry at `fixed_column` is 1. It lives in
    /// the [`Workspace`] of the solve.
    pub x: &'w [f64],
    /// Numerical rank of the matrix, `n` for a regular system
    pub rank: usize,
    /// Column of `x` fixed to 1: `pos` if it was given, otherwise the column
    /// chosen by the solver. The homotopy corrector passes the column chosen
    /// for the predictor back in as `pos`.
    pub fixed_column: usize,
    /// Solver status
    pub status: SolveStatus,
}

/// How a successful [`solve`] went.
///
/// Both variants come with a usable solution, the caller decides how to treat
/// a vanishing determinant.
#[derive(Debug)]
pub enum SolveStatus {
    /// The system was solved.
    /// For a rank-deficient but consistent system the unknowns without a pivot are set to 0.
    Solved,
    /// The system was solved.
    /// But `is_casual_tearing_set` is set and the determinant vanishes (|det| < 1e-9).
    /// The caller should fail unless the current point already is the solution.
    SolvedDeterminantZero,
}

/// Why [`solve`] could not solve the system.
///
/// No solution is available in these cases.
#[derive(Debug)]
pub enum SolveError {
    /// The matrix contains NaN or infinite values. Checked before the
    /// elimination, the C implementation fails later with a NaN determinant.
    MatrixNotFinite,
    /// The determinant is NaN: the product of the pivots overflowed to
    /// infinity and was multiplied with a zero diagonal entry beyond the rank
    NanDeterminant,
    /// The matrix is rank-deficient and the system is inconsistent, a row
    /// without pivot has a right-hand side larger than `1e-6`
    UnderDetermined,
}

/// Gaussian elimination with complete ("total") pivoting
///
/// Solve a linear equation system
/// ```tex
/// A · x = 0,   with one entry of x fixed to 1
/// ```
/// where `A` is a `n×(n+1)` matrix, `x` a vector of size `n+1`. The
/// elimination works on a copy of `matrix` in `work`, as a slice column after
/// column, which is faster than indexing a faer matrix.
///
/// - `pos`: the entry of `x` fixed to 1. `Some(n)` fixes the last column and
///   solves `J·x = -b` for `A = [J | -b]`. `None` lets the solver choose the
///   column, the one left over after pivoting, which is returned in
///   [`Solution::fixed_column`].
/// - `is_casual_tearing_set`: the system is the casual set of a dynamic
///   tearing. Only then a vanishing determinant (`|det| < 1e-9`) is reported
///   as [`SolveStatus::SolvedDeterminantZero`].
/// - `work`: the buffers, for systems of the size of `matrix`.
///
/// From C function `solveSystemWithTotalPivotSearch`. Like C, a rank-deficient
/// but consistent system is solved with the unknowns without pivot set to 0,
/// and an overflow of the determinant to infinity is not checked.
///
/// # Errors
///
/// Returns a [`SolveError`] if `matrix` contains NaN or infinite values, if the
/// determinant is NaN, or if the system is rank-deficient and inconsistent.
///
/// # Panics
///
/// Panics if `matrix` is not `n×(n+1)`, `pos` is not a valid column index or
/// `work` has another size.
pub fn solve<'w>(
    matrix: &Mat<f64>,
    pos: Option<usize>,
    is_casual_tearing_set: bool,
    work: &'w mut Workspace,
) -> Result<Solution<'w>, SolveError> {
    let mut status = SolveStatus::Solved;

    let n_rows = matrix.nrows(); // n
    let n_cols = matrix.ncols(); // n+1

    assert!(n_rows + 1 == n_cols, "Wrong matrix shape {n_rows}x{n_cols}");
    assert!(
        pos.unwrap_or(0) < n_cols,
        "Position argument `pos` out of range!"
    );

    if !matrix.is_all_finite() {
        return Err(SolveError::MatrixNotFinite);
    }

    assert_eq!(work.x.len(), n_cols, "Workspace has wrong size");
    for (values, column) in work.values.chunks_exact_mut(n_rows).zip(matrix.col_iter()) {
        for (value, entry) in values.iter_mut().zip(column.iter()) {
            *value = *entry;
        }
    }
    let mut matrix = PermutedMatrix::new(&mut work.values, &mut work.permutations);

    // The fixed column is no pivot candidate
    let n_pivot_columns = if pos.is_some() { n_rows } else { n_cols };

    if let Some(pos) = pos {
        matrix.swap_columns(n_rows, pos);
    }

    let rank = forward_elimination(&mut matrix, n_pivot_columns);
    if rank < n_rows {
        warn!(target: "NLS", "Matrix (nearly) singular. Continuing anyway.");
        trace!(target: "NLS", "rank = {rank}");
        trace!(target: "NLS", "position = {:?}", pos);
    }
    trace!(target: "NLS", "Linear System Matrix after forward elimination:\n{matrix:10.4}");

    // Check determinant for consistency. Like C, an overflow to infinity is not
    // checked: the solution doesn't depend on the determinant.
    let determinant_abs = matrix.diagonal().product::<f64>().abs();
    trace!(target: "NLS", "Determinant = {determinant_abs}");
    if determinant_abs.is_nan() {
        error!(target: "NLS", "Jacobian determinant is NaN."); // TODO: Should be printed by caller
        return Err(SolveError::NanDeterminant);
    }
    if determinant_abs < 1e-9 && is_casual_tearing_set {
        trace!(target: "DT", "The determinant of the casual tearing set is vanishing, let's fail if this is not the solution..."); // TODO: Should be printed by caller
        status = SolveStatus::SolvedDeterminantZero;
    }

    // Solve even singular matrices
    backward_elimination(&matrix, rank, &mut work.x)?;

    if pos.is_none() {
        trace!(target: "NLS", "Position of largest value = {0}", matrix.column_permutation()[n_rows]);
    }

    let fixed_column = match pos {
        None => matrix.column_permutation()[n_rows],
        Some(p) => p,
    };
    Ok(Solution {
        x: &work.x,
        rank,
        fixed_column,
        status,
    })
}
