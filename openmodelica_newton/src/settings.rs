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

//! Settings of the Newton-Raphson algorithm: tolerances and the maximum number
//! of iterations.

/// Default step tolerance, `newtonXTol` of the C runtime
const X_TOL_DEFAULT: f64 = 1e-12;
/// Default residual tolerance, `newtonFTol` of the C runtime
const F_TOL_DEFAULT: f64 = 1e-12;
/// Default maximum number of iterations per unknown, the C runtime allows
/// `100·n` iterations for a system with `n` unknowns
const MAX_ITERATIONS_DEFAULT: usize = 100;

/// Settings of the Newton-Raphson algorithm.
///
/// [`Settings::new`] creates the defaults for a system of a given size, the
/// `with_*` methods change them.
///
/// # Example
///
/// ```
/// use openmodelica_newton::settings::Settings;
///
/// // A system with 3 unknowns
/// let settings = Settings::new(3).with_f_tol(1e-10).with_maximum_iterations(50);
/// ```
pub struct Settings {
    /// Step tolerance: the iteration has converged once the norm of the
    /// Newton step is below it
    pub(crate) x_tol: f64,
    /// Residual tolerance: the iteration has converged once the norm of the
    /// residual is below it
    pub(crate) f_tol: f64,
    /// Maximum number of Newton iterations before giving up
    pub(crate) maximum_iterations: usize,
}

impl Settings {
    /// Creates the default settings for a system with `size` unknowns: step
    /// and residual tolerance of `1e-12` and at most `100·size` iterations,
    /// like the C runtime.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "Invalid size {size}");
        Settings {
            x_tol: X_TOL_DEFAULT,
            f_tol: F_TOL_DEFAULT,
            maximum_iterations: size * MAX_ITERATIONS_DEFAULT,
        }
    }

    /// The residual tolerance `f_tol`.
    pub fn f_tol(&self) -> f64 {
        self.f_tol
    }

    /// Sets the step tolerance, the iteration has converged once the norm of
    /// the Newton step is below `x_tol`.
    ///
    /// # Panics
    ///
    /// Panics if `x_tol` is not positive.
    pub fn with_x_tol(mut self, x_tol: f64) -> Self {
        assert!(x_tol > 0.0, "Invalid tolerance {x_tol}");
        self.x_tol = x_tol;
        self
    }

    /// Sets the residual tolerance, the iteration has converged once the norm
    /// of the residual is below `f_tol`.
    ///
    /// # Panics
    ///
    /// Panics if `f_tol` is not positive.
    pub fn with_f_tol(mut self, f_tol: f64) -> Self {
        assert!(f_tol > 0.0, "Invalid tolerance {f_tol}");
        self.f_tol = f_tol;
        self
    }

    /// Sets the maximum number of Newton iterations before giving up,
    /// independent of the size of the system.
    ///
    /// # Panics
    ///
    /// Panics if `maximum_iterations` is zero.
    pub fn with_maximum_iterations(mut self, maximum_iterations: usize) -> Self {
        assert!(
            maximum_iterations > 0,
            "Invalid maximum number of iterations {maximum_iterations}"
        );
        self.maximum_iterations = maximum_iterations;
        self
    }
}

#[cfg(test)]
mod settings_tests {
    use super::*;

    #[test]
    /// `Settings::new` has the default tolerances and 100 iterations per
    /// unknown
    fn settings_defaults() {
        let settings = Settings::new(2);

        assert_eq!(settings.x_tol, X_TOL_DEFAULT);
        assert_eq!(settings.f_tol, F_TOL_DEFAULT);
        assert_eq!(settings.maximum_iterations, 2 * MAX_ITERATIONS_DEFAULT);
    }

    #[test]
    /// The `with_*` methods set the tolerances and the maximum iterations
    fn settings_with() {
        let settings = Settings::new(2)
            .with_x_tol(2.0)
            .with_f_tol(3.0)
            .with_maximum_iterations(10);

        assert_eq!(settings.x_tol, 2.0);
        assert_eq!(settings.f_tol, 3.0);
        assert_eq!(settings.maximum_iterations, 10);
    }

    #[test]
    #[should_panic = "Invalid tolerance"]
    /// The residual tolerance must be positive
    fn invalid_tolerance_test() {
        let _ = Settings::new(2).with_f_tol(0.0);
    }
}
