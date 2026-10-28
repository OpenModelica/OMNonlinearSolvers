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

//! The linear solver of the Newton steps after the first.

use faer::Mat;
use openmodelica_total_pivot_solver::{SolveStatus, Workspace};

/// Solves the linear system of a Newton step after the first iteration.
///
/// Like `linearSolverWrapper` of the C runtime, which solves with the linear
/// solver the runtime is set to, LAPACK by default. The first step, at the
/// start point or the end of a homotopy path, is always solved with total
/// pivoting, like in C. [`TotalPivot`] solves the later steps alike.
pub trait LinearSolver {
    /// Solve the scaled system `[J | f]` of `system` for the step `y` with
    /// `J·y = −f`.
    ///
    /// `system` is the `n×(n+1)` matrix with `f` as last column, its columns
    /// scaled with the variable scaling and its rows with their maximum. It
    /// may be overwritten. `y` has `n` entries.
    ///
    /// Returns [`SolveStatus::SolvedDeterminantZero`] for a casual tearing
    /// set whose determinant vanishes, Newton's method fails then unless it
    /// converges in this step.
    ///
    /// # Errors
    ///
    /// Returns [`Singular`] if the system can't be solved.
    fn solve(
        &mut self,
        system: &mut Mat<f64>,
        casual_tearing_set: bool,
        y: &mut [f64],
    ) -> Result<SolveStatus, Singular>;
}

/// The linear system of a Newton step can't be solved.
#[derive(Debug)]
pub struct Singular;

/// Gaussian elimination with complete pivoting, see
/// [`openmodelica_total_pivot_solver::solve`], with its buffers.
pub struct TotalPivot {
    work: Workspace,
}

impl TotalPivot {
    /// Allocates the buffers for systems with `n` unknowns.
    pub fn new(n: usize) -> Self {
        TotalPivot {
            work: Workspace::new(n),
        }
    }
}

impl LinearSolver for TotalPivot {
    fn solve(
        &mut self,
        system: &mut Mat<f64>,
        casual_tearing_set: bool,
        y: &mut [f64],
    ) -> Result<SolveStatus, Singular> {
        let n = y.len();
        let solution = openmodelica_total_pivot_solver::solve(
            system,
            Some(n),
            casual_tearing_set,
            &mut self.work,
        )
        .map_err(|_| Singular)?;
        y.copy_from_slice(&solution.x[..n]);
        Ok(solution.status)
    }
}
