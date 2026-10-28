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

//! The settings of the solver as a C struct.

use openmodelica_homotopy::settings::Settings;
use openmodelica_newton::settings::Settings as NewtonSettings;

/// Settings of [`crate::HomSolver`], the `newton*` and `hom*` flags of the
/// OpenModelica C runtime.
///
/// A value of 0 keeps the default, so `HomSettings settings = {0};` gives the
/// defaults of the C runtime and a host only sets what it changes. No setting
/// can be 0 itself.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HomSettings {
    /// Step tolerance of Newton's method, `newtonXTol`
    pub newton_x_tol: f64,
    /// Residual tolerance of Newton's method, `newtonFTol`
    pub newton_f_tol: f64,
    /// Maximum number of Newton iterations, default `100·size`
    pub newton_max_iterations: usize,
    /// `homTauStart`
    pub tau_start: f64,
    /// `homTauMin`
    pub tau_min: f64,
    /// `homTauMax`
    pub tau_max: f64,
    /// `homTauDecreasingFactor`
    pub tau_decreasing_factor: f64,
    /// `homTauDecreasingFactorPredictor`
    pub tau_decreasing_factor_predictor: f64,
    /// `homTauIncreasingFactor`
    pub tau_increasing_factor: f64,
    /// `homTauIncreasingThreshold`
    pub tau_increasing_threshold: f64,
    /// `homAdaptBend`
    pub adapt_bend: f64,
    /// `homHEps`
    pub h_eps: f64,
    /// `homMaxNewtonSteps`
    pub max_corrector_steps: usize,
    /// `homMaxTries`
    pub max_tries: usize,
    /// `homMaxLambdaSteps`, default `100·size`
    pub max_lambda_steps: usize,
}

impl HomSettings {
    /// The settings of Newton's method and the homotopy for a system with
    /// `size` unknowns.
    ///
    /// # Panics
    ///
    /// Panics if a value that isn't 0 is invalid, see [`NewtonSettings`] and
    /// [`Settings`].
    pub(crate) fn to_settings(self, size: usize) -> (NewtonSettings, Settings) {
        let newton = NewtonSettings::new(size);
        let newton = or_default(newton, self.newton_x_tol, NewtonSettings::with_x_tol);
        let newton = or_default(newton, self.newton_f_tol, NewtonSettings::with_f_tol);
        let newton = or_default(
            newton,
            self.newton_max_iterations,
            NewtonSettings::with_maximum_iterations,
        );

        let homotopy = Settings::new(size);
        let homotopy = or_default(homotopy, self.tau_start, Settings::with_tau_start);
        let homotopy = or_default(homotopy, self.tau_min, Settings::with_tau_min);
        let homotopy = or_default(homotopy, self.tau_max, Settings::with_tau_max);
        let homotopy = or_default(
            homotopy,
            self.tau_decreasing_factor,
            Settings::with_tau_decreasing_factor,
        );
        let homotopy = or_default(
            homotopy,
            self.tau_decreasing_factor_predictor,
            Settings::with_tau_decreasing_factor_predictor,
        );
        let homotopy = or_default(
            homotopy,
            self.tau_increasing_factor,
            Settings::with_tau_increasing_factor,
        );
        let homotopy = or_default(
            homotopy,
            self.tau_increasing_threshold,
            Settings::with_tau_increasing_threshold,
        );
        let homotopy = or_default(homotopy, self.adapt_bend, Settings::with_adapt_bend);
        let homotopy = or_default(homotopy, self.h_eps, Settings::with_h_eps);
        let homotopy = or_default(
            homotopy,
            self.max_corrector_steps,
            Settings::with_max_corrector_steps,
        );
        let homotopy = or_default(homotopy, self.max_tries, Settings::with_max_tries);
        let homotopy = or_default(
            homotopy,
            self.max_lambda_steps,
            Settings::with_max_lambda_steps,
        );

        (newton, homotopy)
    }
}

/// `settings` with `value` set by `with`, unless `value` is 0, which keeps
/// the default.
fn or_default<S, T: Default + PartialEq>(settings: S, value: T, with: fn(S, T) -> S) -> S {
    if value == T::default() {
        settings
    } else {
        with(settings, value)
    }
}
