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

//! The step size control of the homotopy path following.

use crate::settings::Settings;

/// How the corrector of a step ended, the input of [`control`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Outcome {
    /// The corrector reached the path
    Converged,
    /// The corrector used all its Newton steps without reaching the path
    NotConverged,
    /// The corrector failed, the model or a linear system couldn't be solved
    Failed,
}

/// Why the path following ends without reaching `λ = 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum StepFailure {
    /// The corrector neither reached the path nor moved away from the
    /// predicted point
    CorrectorStuck,
    /// A step failed with the step size already at `tau_min`
    TauAtMinimum,
    /// The maximum number of `tries` of one step is used up
    TooManyTries { tries: usize },
}

/// What the path following does after a step, see [`control`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Control {
    /// Accept the corrected point and go on with the step size `tau`
    Accept { tau: f64 },
    /// Reject the step and try again from the same point with the smaller
    /// step size `tau`, as try number `tries`
    Retry { tau: f64, tries: usize },
    /// Give up the path
    Fail(StepFailure),
}

/// The bending `‖y1 − yt‖ / ‖yt − y0‖` of a step from `y0` to the predicted
/// point `yt` and the corrected point `y1`: the length of the corrector
/// relative to the length of the predictor step.
///
/// # Panics
///
/// Panics if `y0`, `yt` and `y1` don't have the same length.
pub(crate) fn bend(y0: &[f64], yt: &[f64], y1: &[f64]) -> f64 {
    assert_eq!(y0.len(), yt.len(), "Predicted point has wrong length");
    assert_eq!(y0.len(), y1.len(), "Corrected point has wrong length");

    let distance = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    };
    distance(y1, yt) / distance(yt, y0)
}

/// Decide after the corrector with the `outcome` and the `bend` of the step
/// whether to accept it, and which step size `tau` to use next.
///
/// * A step is accepted if its corrector converged and it bent at most
///   `adapt_bend`. Then `τ` grows by `tau_increasing_factor`, up to `tau_max`,
///   if it bent less than `adapt_bend / tau_increasing_threshold`.
/// * Otherwise `τ` shrinks by `tau_decreasing_factor`, down to `tau_min`, and
///   `tries` counts up. The path following fails once `max_tries` tries are
///   reached, or at once if `τ` was already at `tau_min`. In the C runtime
///   that step uses up all tries, and it reports the two cases differently.
/// * A corrector that didn't fail, didn't converge and stayed at the
///   predicted point (`bend < ε`) fails the path following.
///
/// `bend` is ignored for a [`Outcome::Failed`] corrector. After an accepted
/// step the caller starts counting the tries of the next step at 0. From the
/// end of the step in `homotopyAlgorithm` and the test of `homMaxTries` at its
/// start.
pub(crate) fn control(
    outcome: Outcome,
    bend: f64,
    tau: f64,
    tries: usize,
    settings: &Settings,
) -> Control {
    if outcome == Outcome::Converged && bend <= settings.adapt_bend {
        let tau = if bend < settings.adapt_bend / settings.tau_increasing_threshold {
            settings.tau_max.min(tau * settings.tau_increasing_factor)
        } else {
            tau
        };
        return Control::Accept { tau };
    }

    if outcome == Outcome::NotConverged && bend < f64::EPSILON {
        return Control::Fail(StepFailure::CorrectorStuck);
    }

    let smaller_tau = settings.tau_min.max(tau / settings.tau_decreasing_factor);
    if smaller_tau == tau {
        return Control::Fail(StepFailure::TauAtMinimum);
    }
    let tries = tries + 1;
    if tries >= settings.max_tries {
        return Control::Fail(StepFailure::TooManyTries { tries });
    }
    Control::Retry {
        tau: smaller_tau,
        tries,
    }
}

#[cfg(test)]
mod step_control_tests {
    use super::*;

    /// [`control`] with the default settings
    fn control_default(outcome: Outcome, bend: f64, tau: f64, tries: usize) -> Control {
        control(outcome, bend, tau, tries, &Settings::new(1))
    }

    #[test]
    /// The corrector `(0, 0.1)` relative to the predictor `(0.2, 0.1)`
    fn bend_test() {
        let bend = bend(&[0.0, 0.0], &[0.2, 0.1], &[0.2, 0.2]);

        assert!((bend - 0.1 / 0.05_f64.sqrt()).abs() < 1e-15, "{bend}");
    }

    #[test]
    /// A converged step that bent between `0.05` and `0.5` keeps `τ`
    fn accept_test() {
        assert_eq!(
            control_default(Outcome::Converged, 0.1, 0.2, 3),
            Control::Accept { tau: 0.2 }
        );
        assert_eq!(
            control_default(Outcome::Converged, 0.5, 0.2, 0),
            Control::Accept { tau: 0.2 },
            "a bend of exactly adapt_bend is accepted"
        );
    }

    #[test]
    /// A step that bent less than `0.5 / 10` doubles `τ`, up to `τ_max = 10`
    fn accept_grow_test() {
        assert_eq!(
            control_default(Outcome::Converged, 0.01, 0.2, 0),
            Control::Accept { tau: 0.4 }
        );
        assert_eq!(
            control_default(Outcome::Converged, 0.01, 8.0, 0),
            Control::Accept { tau: 10.0 }
        );
    }

    #[test]
    /// A step that bent too much, or whose corrector didn't converge or
    /// failed, is retried with a tenth of `τ`
    fn retry_test() {
        let retry = Control::Retry {
            tau: 0.02,
            tries: 1,
        };
        assert_eq!(control_default(Outcome::Converged, 0.6, 0.2, 0), retry);
        assert_eq!(control_default(Outcome::NotConverged, 0.1, 0.2, 0), retry);
        assert_eq!(
            control_default(Outcome::Failed, 0.0, 0.2, 0),
            retry,
            "bend is ignored"
        );
    }

    #[test]
    /// A corrector that didn't converge and didn't move gets stuck
    fn corrector_stuck_test() {
        assert_eq!(
            control_default(Outcome::NotConverged, 0.0, 0.2, 0),
            Control::Fail(StepFailure::CorrectorStuck)
        );
    }

    #[test]
    /// `τ` shrinks to `τ_min = 10⁻⁴`, a rejected step at `τ_min` gives up
    fn tau_min_test() {
        assert_eq!(
            control_default(Outcome::Failed, 0.0, 5e-4, 0),
            Control::Retry {
                tau: 1e-4,
                tries: 1
            }
        );
        assert_eq!(
            control_default(Outcome::Failed, 0.0, 1e-4, 1),
            Control::Fail(StepFailure::TauAtMinimum)
        );
    }

    #[test]
    /// The 10th rejected try of a step gives up
    fn too_many_tries_test() {
        assert_eq!(
            control_default(Outcome::Failed, 0.0, 0.2, 8),
            Control::Retry {
                tau: 0.02,
                tries: 9
            }
        );
        assert_eq!(
            control_default(Outcome::Failed, 0.0, 0.2, 9),
            Control::Fail(StepFailure::TooManyTries { tries: 10 })
        );
    }
}
