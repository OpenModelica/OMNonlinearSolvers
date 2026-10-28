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

//! Linear solver with complete ("total") pivoting for the homotopy and Newton
//! solvers.
//!
//! Port of `solveSystemWithTotalPivotSearch` from the OpenModelica C runtime
//! (`nonlinearSolverHomotopy.c`). It solves `A·x = 0` for an `n×(n+1)` matrix
//! `A` with one entry of `x` fixed to 1. With the last column fixed this is the
//! ordinary system `J·x = -b`; letting the solver choose the fixed column
//! computes the tangent of a homotopy path.

// int linearSolverWrapper(DATA *data, int n, double* x, double* A, int* indRow, int* indCol, int *pos, int *rank, int method, int casualTearingSet)

// int solveSystemWithTotalPivotSearch(DATA *data, int n, double* x, double* A, int* indRow, int* indCol, int *pos, int *rank, int casualTearingSet)

/// Back substitution on the eliminated matrix
mod backward_elimination;
/// Forward elimination with complete pivoting
mod forward_elimination;
/// Search for the pivot element of one elimination step
mod pivot_search;
/// Solving the linear system, the public entry point
mod solve;

pub use solve::{Solution, SolveError, SolveStatus, Workspace, solve};

/// The values of `m`, column after column, for a `PermutedMatrix` in tests
#[cfg(test)]
fn values_of(m: &faer::Mat<f64>) -> Vec<f64> {
    m.col_iter()
        .flat_map(|column| column.iter().copied().collect::<Vec<_>>())
        .collect()
}
/// Pivot element found by [`pivot_search`]
struct Pivot {
    /// Logical row of the pivot element
    row: usize,
    /// Logical column of the pivot element
    column: usize,
    /// Absolute value of the pivot element
    abs_value: f64,
}
