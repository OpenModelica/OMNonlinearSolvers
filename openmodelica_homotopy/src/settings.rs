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

//! Settings of the homotopy path following, with the defaults of the `hom*`
//! flags of the C runtime.

/// Default step size `τ` at the start of the path, `homTauStart`
const TAU_START_DEFAULT: f64 = 0.2;
/// Default smallest step size, `homTauMin`
const TAU_MIN_DEFAULT: f64 = 1e-4;
/// Default largest step size, `homTauMax`
const TAU_MAX_DEFAULT: f64 = 10.0;
/// Default factor `τ` shrinks by after a rejected step, `homTauDecreasingFactor`
const TAU_DECREASING_FACTOR_DEFAULT: f64 = 10.0;
/// Default factor `τ` shrinks by while the predictor point can't be
/// evaluated, `homTauDecreasingFactorPredictor`
const TAU_DECREASING_FACTOR_PREDICTOR_DEFAULT: f64 = 2.0;
/// Default factor `τ` grows by after a step with little bending,
/// `homTauIncreasingFactor`
const TAU_INCREASING_FACTOR_DEFAULT: f64 = 2.0;
/// Default threshold for growing `τ`, `homTauIncreasingThreshold`
const TAU_INCREASING_THRESHOLD_DEFAULT: f64 = 10.0;
/// Default largest bending of an accepted step, `homAdaptBend`
const ADAPT_BEND_DEFAULT: f64 = 0.5;
/// Default residual tolerance of the corrector, `homHEps`
const H_EPS_DEFAULT: f64 = 1e-5;
/// Default maximum number of Newton steps of one corrector, `homMaxNewtonSteps`
const MAX_CORRECTOR_STEPS_DEFAULT: usize = 20;
/// Default maximum number of tries of one step along the path, `homMaxTries`
const MAX_TRIES_DEFAULT: usize = 10;
/// Default maximum number of accepted steps along the path per unknown. Like
/// the maximum number of Newton iterations, `100·n` for `n` unknowns, which
/// the C runtime uses while `homMaxLambdaSteps` is 0.
const MAX_LAMBDA_STEPS_DEFAULT: usize = 100;

/// Settings of the homotopy path following.
///
/// [`Settings::new`] creates the defaults of the C runtime for a system of a
/// given size, the `with_*` methods change them.
///
/// # Example
///
/// ```
/// use openmodelica_homotopy::settings::Settings;
///
/// // A system with 3 unknowns
/// let settings = Settings::new(3).with_tau_start(0.1).with_max_tries(20);
/// ```
#[derive(Debug, Clone)]
pub struct Settings {
    /// Step size `τ` at the start of the path
    pub(crate) tau_start: f64,
    /// Smallest step size, a step that fails with it ends the path following
    pub(crate) tau_min: f64,
    /// Largest step size
    pub(crate) tau_max: f64,
    /// Factor `τ` shrinks by when the corrector fails or the path bends too
    /// much
    pub(crate) tau_decreasing_factor: f64,
    /// Factor `τ` shrinks by while the model can't be evaluated at the
    /// predictor point
    pub(crate) tau_decreasing_factor_predictor: f64,
    /// Factor `τ` grows by after a step that bent less than
    /// `adapt_bend / tau_increasing_threshold`
    pub(crate) tau_increasing_factor: f64,
    /// Threshold for growing `τ`, see `tau_increasing_factor`
    pub(crate) tau_increasing_threshold: f64,
    /// Largest bending of the path, the ratio of the corrector to the
    /// predictor step, to accept a step
    pub(crate) adapt_bend: f64,
    /// Residual tolerance of the homotopy function `H` in the corrector
    pub(crate) h_eps: f64,
    /// Maximum number of Newton steps of one corrector
    pub(crate) max_corrector_steps: usize,
    /// Maximum number of tries of one step along the path
    pub(crate) max_tries: usize,
    /// Maximum number of accepted steps along the path
    pub(crate) max_lambda_steps: usize,
}

impl Settings {
    /// Creates the default settings of the C runtime for a system with `size`
    /// unknowns, at most `100·size` steps along the path.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "Invalid size {size}");
        Settings {
            tau_start: TAU_START_DEFAULT,
            tau_min: TAU_MIN_DEFAULT,
            tau_max: TAU_MAX_DEFAULT,
            tau_decreasing_factor: TAU_DECREASING_FACTOR_DEFAULT,
            tau_decreasing_factor_predictor: TAU_DECREASING_FACTOR_PREDICTOR_DEFAULT,
            tau_increasing_factor: TAU_INCREASING_FACTOR_DEFAULT,
            tau_increasing_threshold: TAU_INCREASING_THRESHOLD_DEFAULT,
            adapt_bend: ADAPT_BEND_DEFAULT,
            h_eps: H_EPS_DEFAULT,
            max_corrector_steps: MAX_CORRECTOR_STEPS_DEFAULT,
            max_tries: MAX_TRIES_DEFAULT,
            max_lambda_steps: size * MAX_LAMBDA_STEPS_DEFAULT,
        }
    }

    /// Sets the step size `τ` at the start of the path, `homTauStart`.
    ///
    /// # Panics
    ///
    /// Panics if `tau_start` is not positive.
    pub fn with_tau_start(mut self, tau_start: f64) -> Self {
        assert!(
            tau_start > 0.0,
            "Invalid tau_start (homTauStart) {tau_start}, must be positive"
        );
        self.tau_start = tau_start;
        self
    }

    /// Sets the smallest step size, `homTauMin`.
    ///
    /// # Panics
    ///
    /// Panics if `tau_min` is not positive.
    pub fn with_tau_min(mut self, tau_min: f64) -> Self {
        assert!(
            tau_min > 0.0,
            "Invalid tau_min (homTauMin) {tau_min}, must be positive"
        );
        self.tau_min = tau_min;
        self
    }

    /// Sets the largest step size, `homTauMax`.
    ///
    /// # Panics
    ///
    /// Panics if `tau_max` is not positive.
    pub fn with_tau_max(mut self, tau_max: f64) -> Self {
        assert!(
            tau_max > 0.0,
            "Invalid tau_max (homTauMax) {tau_max}, must be positive"
        );
        self.tau_max = tau_max;
        self
    }

    /// Sets the factor `τ` shrinks by when the corrector fails or the path
    /// bends too much, `homTauDecreasingFactor`.
    ///
    /// # Panics
    ///
    /// Panics if `factor` is not greater than 1.
    pub fn with_tau_decreasing_factor(mut self, factor: f64) -> Self {
        assert!(
            factor > 1.0,
            "Invalid tau_decreasing_factor (homTauDecreasingFactor) {factor}, must be greater than 1"
        );
        self.tau_decreasing_factor = factor;
        self
    }

    /// Sets the factor `τ` shrinks by while the model can't be evaluated at
    /// the predictor point, `homTauDecreasingFactorPredictor`.
    ///
    /// # Panics
    ///
    /// Panics if `factor` is not greater than 1.
    pub fn with_tau_decreasing_factor_predictor(mut self, factor: f64) -> Self {
        assert!(
            factor > 1.0,
            "Invalid tau_decreasing_factor_predictor (homTauDecreasingFactorPredictor) {factor}, must be greater than 1"
        );
        self.tau_decreasing_factor_predictor = factor;
        self
    }

    /// Sets the factor `τ` grows by after a step with little bending,
    /// `homTauIncreasingFactor`.
    ///
    /// # Panics
    ///
    /// Panics if `factor` is not greater than 1.
    pub fn with_tau_increasing_factor(mut self, factor: f64) -> Self {
        assert!(
            factor > 1.0,
            "Invalid tau_increasing_factor (homTauIncreasingFactor) {factor}, must be greater than 1"
        );
        self.tau_increasing_factor = factor;
        self
    }

    /// Sets the threshold for growing `τ`: it grows after a step that bent
    /// less than `adapt_bend / threshold`, `homTauIncreasingThreshold`.
    ///
    /// # Panics
    ///
    /// Panics if `threshold` is not positive.
    pub fn with_tau_increasing_threshold(mut self, threshold: f64) -> Self {
        assert!(
            threshold > 0.0,
            "Invalid tau_increasing_threshold (homTauIncreasingThreshold) {threshold}, must be positive"
        );
        self.tau_increasing_threshold = threshold;
        self
    }

    /// Sets the largest bending of the path to accept a step, `homAdaptBend`.
    ///
    /// # Panics
    ///
    /// Panics if `adapt_bend` is not positive.
    pub fn with_adapt_bend(mut self, adapt_bend: f64) -> Self {
        assert!(
            adapt_bend > 0.0,
            "Invalid adapt_bend (homAdaptBend) {adapt_bend}, must be positive"
        );
        self.adapt_bend = adapt_bend;
        self
    }

    /// Sets the residual tolerance of the corrector, `homHEps`.
    ///
    /// # Panics
    ///
    /// Panics if `h_eps` is not positive.
    pub fn with_h_eps(mut self, h_eps: f64) -> Self {
        assert!(
            h_eps > 0.0,
            "Invalid h_eps (homHEps) {h_eps}, must be positive"
        );
        self.h_eps = h_eps;
        self
    }

    /// Sets the maximum number of Newton steps of one corrector,
    /// `homMaxNewtonSteps`.
    ///
    /// # Panics
    ///
    /// Panics if `max_corrector_steps` is zero.
    pub fn with_max_corrector_steps(mut self, max_corrector_steps: usize) -> Self {
        assert!(
            max_corrector_steps > 0,
            "Invalid max_corrector_steps (homMaxNewtonSteps) {max_corrector_steps}, must not be zero"
        );
        self.max_corrector_steps = max_corrector_steps;
        self
    }

    /// Sets the maximum number of tries of one step along the path,
    /// `homMaxTries`.
    ///
    /// # Panics
    ///
    /// Panics if `max_tries` is zero.
    pub fn with_max_tries(mut self, max_tries: usize) -> Self {
        assert!(
            max_tries > 0,
            "Invalid max_tries (homMaxTries) {max_tries}, must not be zero"
        );
        self.max_tries = max_tries;
        self
    }

    /// Sets the maximum number of accepted steps along the path, independent
    /// of the size of the system, `homMaxLambdaSteps`.
    ///
    /// # Panics
    ///
    /// Panics if `max_lambda_steps` is zero.
    pub fn with_max_lambda_steps(mut self, max_lambda_steps: usize) -> Self {
        assert!(
            max_lambda_steps > 0,
            "Invalid max_lambda_steps (homMaxLambdaSteps) {max_lambda_steps}, must not be zero"
        );
        self.max_lambda_steps = max_lambda_steps;
        self
    }
}

#[cfg(test)]
mod settings_tests {
    use super::*;

    #[test]
    /// `Settings::new` has the defaults of the C runtime and 100 steps along
    /// the path per unknown
    fn settings_defaults_test() {
        let settings = Settings::new(3);

        assert_eq!(settings.tau_start, 0.2);
        assert_eq!(settings.tau_min, 1e-4);
        assert_eq!(settings.tau_max, 10.0);
        assert_eq!(settings.tau_decreasing_factor, 10.0);
        assert_eq!(settings.tau_decreasing_factor_predictor, 2.0);
        assert_eq!(settings.tau_increasing_factor, 2.0);
        assert_eq!(settings.tau_increasing_threshold, 10.0);
        assert_eq!(settings.adapt_bend, 0.5);
        assert_eq!(settings.h_eps, 1e-5);
        assert_eq!(settings.max_corrector_steps, 20);
        assert_eq!(settings.max_tries, 10);
        assert_eq!(settings.max_lambda_steps, 300);
    }

    #[test]
    /// The `with_*` methods set every value
    fn settings_with_test() {
        let settings = Settings::new(3)
            .with_tau_start(0.1)
            .with_tau_min(1e-6)
            .with_tau_max(5.0)
            .with_tau_decreasing_factor(4.0)
            .with_tau_decreasing_factor_predictor(3.0)
            .with_tau_increasing_factor(1.5)
            .with_tau_increasing_threshold(20.0)
            .with_adapt_bend(0.25)
            .with_h_eps(1e-8)
            .with_max_corrector_steps(5)
            .with_max_tries(7)
            .with_max_lambda_steps(42);

        assert_eq!(settings.tau_start, 0.1);
        assert_eq!(settings.tau_min, 1e-6);
        assert_eq!(settings.tau_max, 5.0);
        assert_eq!(settings.tau_decreasing_factor, 4.0);
        assert_eq!(settings.tau_decreasing_factor_predictor, 3.0);
        assert_eq!(settings.tau_increasing_factor, 1.5);
        assert_eq!(settings.tau_increasing_threshold, 20.0);
        assert_eq!(settings.adapt_bend, 0.25);
        assert_eq!(settings.h_eps, 1e-8);
        assert_eq!(settings.max_corrector_steps, 5);
        assert_eq!(settings.max_tries, 7);
        assert_eq!(settings.max_lambda_steps, 42);
    }

    #[test]
    #[should_panic = "Invalid tau_start (homTauStart) 0, must be positive"]
    fn invalid_tau_test() {
        let _ = Settings::new(1).with_tau_start(0.0);
    }

    #[test]
    #[should_panic = "Invalid tau_decreasing_factor (homTauDecreasingFactor) 1, must be greater than 1"]
    /// A factor of 1 would never change the step size
    fn invalid_factor_test() {
        let _ = Settings::new(1).with_tau_decreasing_factor(1.0);
    }

    #[test]
    #[should_panic = "Invalid max_tries (homMaxTries) 0, must not be zero"]
    fn invalid_count_test() {
        let _ = Settings::new(1).with_max_tries(0);
    }

    #[test]
    #[should_panic = "Invalid size"]
    fn invalid_size_test() {
        let _ = Settings::new(0);
    }
}
