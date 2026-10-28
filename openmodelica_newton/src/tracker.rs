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

//! Progress of a Newton iteration and the decision when to stop it, from the
//! counters and exits of `newtonAlgorithm` in the C runtime.

use crate::error_history::ErrorHistory;
use crate::settings::Settings;
use crate::solve::{NewtonError, NewtonStatus};

/// The error grew strongly if it is more than this factor above the previous
/// error
const GROWTH_FACTOR: f64 = 10.0;
/// The error hovers while it stays below this factor times the hover level
const HOVER_WINDOW: f64 = 10.0;
/// A step is heavily damped if its step factor is below this value
const DAMPED_STEP_FACTOR: f64 = 1e-3;
/// An error below this factor times `f_tol²` is inside the "less accurate"
/// band, close enough to a solution that slow progress is no divergence
const LESS_ACCURATE_BAND: f64 = 1e6;
/// The best error improved if the error is below this factor times the best
/// error, by more than 1%
const IMPROVEMENT_FACTOR: f64 = 0.99;
/// A Newton step is small if its squared length is below this factor times
/// `x_tol²`
const SMALL_STEP_FACTOR: f64 = 1e4;

/// Allowed steps where the error grew strongly, over the whole iteration
const MAX_ERROR_GROWTHS: usize = 20;
/// Allowed limit cycles in a row, far from a solution
const MAX_CYCLES_IN_A_ROW: usize = 20;
/// Allowed steps without improving the best error, far from a solution
const MAX_STEPS_WITHOUT_IMPROVEMENT: usize = 400;
/// Allowed heavily damped steps in a row, far from a solution
const MAX_DAMPED_STEPS_IN_A_ROW: usize = 400;
/// Allowed small Newton steps, over the whole iteration
const MAX_SMALL_STEPS: usize = 20;
/// Steps the error must hover before the iteration can stop at the bottom of
/// a limit cycle
const MIN_STEPS_HOVERING_AT_CYCLE_BOTTOM: usize = 20;

/// Why the Newton iteration moves away from any solution.
#[derive(Debug, PartialEq)]
pub(crate) enum Divergence {
    /// The error grew strongly in more than [`MAX_ERROR_GROWTHS`] steps
    ErrorGrew,
    /// The error cycled in more than [`MAX_CYCLES_IN_A_ROW`] steps in a row
    Cycling,
    /// No improvement for more than [`MAX_STEPS_WITHOUT_IMPROVEMENT`] steps, or
    /// more than [`MAX_DAMPED_STEPS_IN_A_ROW`] heavily damped steps in a row
    NoProgress,
}

/// Measures of one Newton step, the input of [`Tracker::update`]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Step {
    /// Squared residual error `‖F(x1)‖²` at the trial point
    pub(crate) err: f64,
    /// Squared scaled residual error at the trial point
    pub(crate) err_scaled: f64,
    /// Squared length `‖dy‖²` of the full, undamped Newton step
    pub(crate) dy: f64,
    /// Squared scaled length of the full Newton step
    pub(crate) dy_scaled: f64,
    /// Step factor `λ` the line search took
    pub(crate) step_factor: f64,
}

/// What the Newton iteration does after a step, see [`Tracker::decide`].
#[derive(Debug)]
pub(crate) enum Decision {
    /// Accept the trial point `x1` as the new iterate and iterate again
    Continue,
    /// Stop with a solution: the trial point `x1` if `take_step`, otherwise
    /// the current iterate `x`
    Converged {
        status: NewtonStatus,
        take_step: bool,
    },
    /// Stop without a solution
    Failed(NewtonError),
}

/// Tracks the progress of a Newton iteration and decides when to stop it.
///
/// Updated with the [`Step`] of every iteration, see [`Tracker::update`].
pub(crate) struct Tracker {
    /// Steps where the error grew strongly, never reset
    num_error_growths: usize,
    /// Limit cycles in a row, far from a solution
    num_cycles_in_a_row: usize,
    /// Steps since the best error last improved by 1%
    num_steps_without_improvement: usize,
    /// Steps since the error started hovering around `hover_error`
    num_steps_hovering: usize,
    /// Heavily damped steps in a row
    num_damped_steps_in_a_row: usize,
    /// Small Newton steps, never reset
    num_small_steps: usize,
    /// `num_small_steps` when the error last stopped hovering
    num_small_steps_at_hover: usize,
    /// Iterations so far
    num_iterations: usize,

    /// The level the error hovers around
    hover_error: f64,
    /// Best error so far
    best_error: f64,

    /// The last errors, to detect limit cycles
    error_history: ErrorHistory,
    /// The latest step
    step: Step,
    /// Error of the iterate before the latest step
    previous_error: f64,
    /// Scaled error of the iterate before the latest step
    previous_error_scaled: f64,

    /// The latest plain and scaled error are outside the less accurate band
    is_far_from_solution: bool,
    /// Squared residual tolerance
    f_tol_squared: f64,
    /// Squared step tolerance
    x_tol_squared: f64,
    /// Maximum number of iterations
    maximum_iterations: usize,
}

impl Tracker {
    /// Creates a tracker for a new Newton iteration with the tolerances and
    /// the maximum number of iterations of `settings`, starting at an iterate
    /// with the squared error `err` and the squared scaled error `err_scaled`.
    pub(crate) fn new(settings: &Settings, err: f64, err_scaled: f64) -> Self {
        Tracker {
            num_error_growths: 0,
            num_cycles_in_a_row: 0,
            num_steps_without_improvement: 0,
            num_steps_hovering: 0,
            num_damped_steps_in_a_row: 0,
            num_small_steps: 0,
            num_small_steps_at_hover: 0,
            num_iterations: 0,

            hover_error: f64::INFINITY,
            best_error: f64::INFINITY,

            error_history: ErrorHistory::new(),
            step: Step {
                err,
                err_scaled,
                dy: f64::INFINITY,
                dy_scaled: f64::INFINITY,
                step_factor: 1.0,
            },
            previous_error: f64::INFINITY,
            previous_error_scaled: f64::INFINITY,

            is_far_from_solution: true,
            f_tol_squared: settings.f_tol * settings.f_tol,
            x_tol_squared: settings.x_tol * settings.x_tol,
            maximum_iterations: settings.maximum_iterations,
        }
    }

    /// Updates the counters with the latest Newton `step`.
    pub(crate) fn update(&mut self, step: Step) {
        self.num_iterations += 1;
        self.previous_error = self.step.err;
        self.previous_error_scaled = self.step.err_scaled;
        self.step = step;

        let band = self.f_tol_squared * LESS_ACCURATE_BAND;
        self.is_far_from_solution = step.err >= band && step.err_scaled >= band;

        self.count_error_growth();
        self.count_cycles();
        self.count_progress();
        self.count_damped_steps();
        self.count_small_steps();
    }

    /// Counts a strong growth of the error since the previous step.
    fn count_error_growth(&mut self) {
        if self.step.err > GROWTH_FACTOR * self.previous_error {
            self.num_error_growths += 1;
        }
    }

    /// Counts limit cycles in a row, far from a solution. A cycle inside the
    /// less accurate band is left to the convergence check.
    fn count_cycles(&mut self) {
        let has_cycle = self.error_history.push(self.step.err);
        if has_cycle && self.is_far_from_solution {
            self.num_cycles_in_a_row += 1;
        } else {
            self.num_cycles_in_a_row = 0;
        }
    }

    /// Counts the steps without improvement and the steps hovering.
    ///
    /// An improvement of the best error by 1% resets both, otherwise the error
    /// hovers within 10 times the hover level or jumps away from it.
    fn count_progress(&mut self) {
        let err = self.step.err;
        if err < IMPROVEMENT_FACTOR * self.best_error {
            self.best_error = err;
            self.num_steps_without_improvement = 0;
            self.num_steps_hovering = 0;
        } else if err < HOVER_WINDOW * self.hover_error {
            self.num_steps_without_improvement += 1;
            self.num_steps_hovering += 1;
        } else {
            self.num_steps_without_improvement += 1;
            self.num_steps_hovering = 0;
        }

        // A new hover level after an improvement or a jump, a lower one while
        // hovering
        if self.num_steps_hovering == 0 || err < self.hover_error {
            self.hover_error = err;
        }
    }

    /// Counts heavily damped steps in a row.
    fn count_damped_steps(&mut self) {
        if self.step.step_factor < DAMPED_STEP_FACTOR {
            self.num_damped_steps_in_a_row += 1;
        } else {
            self.num_damped_steps_in_a_row = 0;
        }
    }

    /// Counts small Newton steps, and remembers their number while the error
    /// doesn't hover.
    fn count_small_steps(&mut self) {
        let small = SMALL_STEP_FACTOR * self.x_tol_squared;
        if self.step.dy < small || self.step.dy_scaled < small {
            self.num_small_steps += 1;
        }
        if self.num_steps_hovering == 0 {
            self.num_small_steps_at_hover = self.num_small_steps;
        }
    }

    /// Why the iteration diverges, or `None` while it makes progress.
    fn divergence(&self) -> Option<Divergence> {
        if self.num_error_growths > MAX_ERROR_GROWTHS {
            return Some(Divergence::ErrorGrew);
        }
        if self.num_cycles_in_a_row > MAX_CYCLES_IN_A_ROW {
            return Some(Divergence::Cycling);
        }
        let no_progress = self.num_steps_without_improvement > MAX_STEPS_WITHOUT_IMPROVEMENT
            || self.num_damped_steps_in_a_row > MAX_DAMPED_STEPS_IN_A_ROW;
        if no_progress && self.is_far_from_solution {
            return Some(Divergence::NoProgress);
        }

        None
    }

    /// Decides after [`Tracker::update`] whether the iteration goes on.
    ///
    /// `determinant_zero` is set if the linear solver of this step reported a
    /// vanishing determinant for a casual tearing set. The checks and their
    /// order are those of the C runtime, the first that applies wins:
    ///
    /// 1. The iteration diverges.
    /// 2. The error and the Newton step are below their tolerances. The
    ///    previous iterate is kept if it was at least as good (issue #6419 of
    ///    OpenModelica).
    /// 3. A vanishing determinant without a solution fails as singular.
    /// 4. The error stopped decreasing while the previous error already met
    ///    `f_tol`: the previous iterate is the solution.
    /// 5. The maximum number of iterations is exceeded.
    /// 6. The Newton step is below `x_tol`, more than [`MAX_SMALL_STEPS`] steps
    ///    were small, or the error sits at the bottom of a limit cycle inside
    ///    the less accurate band. Inside that band this is a less accurate
    ///    solution, otherwise the iteration is stuck.
    pub(crate) fn decide(&self, determinant_zero: bool) -> Decision {
        if self.divergence().is_some() {
            return Decision::Failed(NewtonError::Diverging);
        }

        let previous_is_as_good = self.step.err >= self.previous_error;
        if self.error_below_tolerance() && self.step_below_tolerance() {
            return Decision::Converged {
                status: NewtonStatus::Solved,
                take_step: !previous_is_as_good,
            };
        }
        if determinant_zero {
            return Decision::Failed(NewtonError::Singular);
        }
        let previous_below_tolerance = self.previous_error < self.f_tol_squared
            || self.previous_error_scaled < self.f_tol_squared;
        if previous_is_as_good && previous_below_tolerance {
            return Decision::Converged {
                status: NewtonStatus::Solved,
                take_step: false,
            };
        }

        if self.num_iterations > self.maximum_iterations {
            return Decision::Failed(NewtonError::MaxIterations);
        }

        let less_accurate = !self.is_far_from_solution;
        let at_cycle_bottom = self.at_cycle_bottom();
        if self.step_below_tolerance()
            || self.num_small_steps > MAX_SMALL_STEPS
            || (less_accurate && at_cycle_bottom)
        {
            return if less_accurate {
                Decision::Converged {
                    status: NewtonStatus::SolvedLessAccurate,
                    take_step: at_cycle_bottom,
                }
            } else {
                Decision::Failed(NewtonError::Stuck)
            };
        }

        Decision::Continue
    }

    /// Whether the plain or the scaled error is below `f_tol²`.
    fn error_below_tolerance(&self) -> bool {
        self.step.err < self.f_tol_squared || self.step.err_scaled < self.f_tol_squared
    }

    /// Whether the plain or the scaled Newton step is below `x_tol²`.
    fn step_below_tolerance(&self) -> bool {
        self.step.dy < self.x_tol_squared || self.step.dy_scaled < self.x_tol_squared
    }

    /// Whether the error hovers in a stationary limit cycle and is at its
    /// lowest of the last four steps. Cycles with small steps are left to the
    /// small step exit.
    fn at_cycle_bottom(&self) -> bool {
        self.num_steps_hovering > MIN_STEPS_HOVERING_AT_CYCLE_BOTTOM
            && self.num_small_steps == self.num_small_steps_at_hover
            && self.error_history.newest_is_lowest()
    }
}

#[cfg(test)]
mod tracker_tests {
    use super::*;

    /// `newtonFTol` of the C runtime. The "less accurate" band is
    /// `e < F_TOL²·10⁶ = 1e-18`.
    const F_TOL: f64 = 1e-12;
    /// `newtonXTol` of the C runtime. Steps below `X_TOL²·10⁴ = 1e-20` are
    /// small.
    const X_TOL: f64 = 1e-12;

    /// A tracker with [`F_TOL`] and [`X_TOL`], at most 1000 iterations and no
    /// initial error, so the first step never grows the error
    fn tracker() -> Tracker {
        let settings = Settings::new(10).with_f_tol(F_TOL).with_x_tol(X_TOL);
        Tracker::new(&settings, f64::INFINITY, f64::INFINITY)
    }

    /// A full Newton step (step factor 1) of length 1 that leads to the error
    /// `err`, the same plain and scaled
    fn step(err: f64) -> Step {
        Step {
            err,
            err_scaled: err,
            dy: 1.0,
            dy_scaled: 1.0,
            step_factor: 1.0,
        }
    }

    /// An error far outside the less accurate band
    const FAR: f64 = 1.0;
    /// An error inside the less accurate band
    const NEAR: f64 = 1e-20;

    /// Run `update` for each `(err, step_factor)` in `steps`, with the same
    /// plain and scaled error. Returns `divergence()` after each step.
    fn run(steps: &[(f64, f64)]) -> Vec<Option<Divergence>> {
        let mut tracker = tracker();
        steps
            .iter()
            .map(|&(err, step_factor)| {
                tracker.update(Step {
                    step_factor,
                    ..step(err)
                });
                tracker.divergence()
            })
            .collect()
    }

    /// `n` full steps (step factor 1) with the errors `err(k)`, `k = 0..n`
    fn full_steps(n: usize, err: impl Fn(usize) -> f64) -> Vec<(f64, f64)> {
        (0..n).map(|k| (err(k), 1.0)).collect()
    }

    #[test]
    /// A steadily decreasing error never diverges
    fn decreasing_error_test() {
        let diverging = run(&full_steps(100, |k| 0.5_f64.powi(k as i32)));

        assert!(diverging.iter().all(Option::is_none));
    }

    #[test]
    /// More than 20 steps where the error grows by more than 10 times diverge
    fn error_growth_test() {
        // The first error has no predecessor, then 21 steps growing 11 times
        let diverging = run(&full_steps(22, |k| 1e-3 * 11_f64.powi(k as i32)));

        assert_eq!(diverging[20], None, "20 error growths are allowed");
        assert_eq!(
            diverging[21],
            Some(Divergence::ErrorGrew),
            "the 21st error growth diverges"
        );
    }

    #[test]
    /// Error growths are counted over all iterations, decreasing steps in
    /// between don't reset the counter
    fn error_growth_not_reset_test() {
        // Alternately grow 11 times and shrink 12 times, 21 times each. The
        // values never repeat within 1%, so this is no cycle.
        let mut steps = vec![(1.0, 1.0)];
        for _ in 0..21 {
            let err = steps.last().unwrap().0;
            steps.push((err * 11.0, 1.0));
            steps.push((err * 11.0 / 12.0, 1.0));
        }

        let diverging = run(&steps);

        assert_eq!(
            diverging[diverging.len() - 3],
            None,
            "20 error growths so far"
        );
        assert_eq!(
            diverging.last().unwrap(),
            &Some(Divergence::ErrorGrew),
            "21 error growths in total"
        );
    }

    #[test]
    /// More than 20 limit cycles in a row, far from a solution, diverge
    fn cycles_far_test() {
        // Period 2: 4 steps fill the history, then every step is a cycle
        let diverging = run(&full_steps(25, |k| {
            if k.is_multiple_of(2) { FAR } else { 2.0 * FAR }
        }));

        assert_eq!(diverging[23], None, "20 cycles in a row are allowed");
        assert_eq!(
            diverging[24],
            Some(Divergence::Cycling),
            "the 21st cycle in a row diverges"
        );
    }

    #[test]
    /// Limit cycles inside the less accurate band are left to the stopping
    /// decision, they don't count as diverging
    fn cycles_near_test() {
        let diverging = run(&full_steps(40, |k| {
            if k.is_multiple_of(2) {
                NEAR
            } else {
                2.0 * NEAR
            }
        }));

        assert!(diverging.iter().all(Option::is_none));
    }

    #[test]
    /// A cycle only counts if the scaled error is outside the less accurate
    /// band as well
    fn cycles_scaled_error_near_test() {
        let mut tracker = tracker();
        for k in 0..40_usize {
            let err = if k.is_multiple_of(2) { FAR } else { 2.0 * FAR };
            tracker.update(Step {
                err_scaled: NEAR,
                ..step(err)
            });

            assert_eq!(tracker.divergence(), None, "step {k}");
        }
    }

    #[test]
    /// A step that is no cycle resets the count of cycles in a row
    fn cycles_reset_test() {
        let alternate = |k: usize| if k.is_multiple_of(2) { FAR } else { 2.0 * FAR };
        // 4 steps fill the history, then 20 cycles
        let mut steps = full_steps(24, alternate);
        // 1.5 returns to no older error: no cycle, the count is reset
        steps.push((1.5 * FAR, 1.0));
        // 20 more cycles, starting with 1.0 after 1.5
        steps.extend(full_steps(20, alternate));

        let diverging = run(&steps);

        assert!(diverging.iter().all(Option::is_none));
    }

    #[test]
    /// More than 400 steps without improving the best error by 1%, far from a
    /// solution, diverge
    fn no_improvement_far_test() {
        // The first step sets the best error, then 401 steps don't improve it
        let diverging = run(&full_steps(402, |_| FAR));

        assert_eq!(
            diverging[400], None,
            "400 steps without improvement are allowed"
        );
        assert_eq!(
            diverging[401],
            Some(Divergence::NoProgress),
            "the 401st diverges"
        );
    }

    #[test]
    /// Steps without improvement inside the less accurate band don't diverge
    fn no_improvement_near_test() {
        let diverging = run(&full_steps(500, |_| NEAR));

        assert!(diverging.iter().all(Option::is_none));
    }

    #[test]
    /// Improving the best error by more than 1% resets the count
    fn no_improvement_reset_test() {
        // 400 steps without improvement, an improvement by 2%, then 400 more
        let mut steps = full_steps(401, |_| FAR);
        steps.extend(full_steps(401, |_| 0.98 * FAR));

        let diverging = run(&steps);

        assert!(diverging.iter().all(Option::is_none));
    }

    #[test]
    /// More than 400 heavily damped steps (step factor below `1e-3`) in a row,
    /// far from a solution, diverge
    fn damped_steps_far_test() {
        // The error improves by 2% per step, so every step is an improvement
        let steps: Vec<_> = (0..401).map(|k| (0.98_f64.powi(k), 1e-4)).collect();

        let diverging = run(&steps);

        assert_eq!(diverging[399], None, "400 heavily damped steps are allowed");
        assert_eq!(
            diverging[400],
            Some(Divergence::NoProgress),
            "the 401st heavily damped step diverges"
        );
    }

    #[test]
    /// A step that isn't heavily damped resets the count of damped steps
    fn damped_steps_reset_test() {
        let mut steps: Vec<_> = (0..400).map(|k| (0.98_f64.powi(k), 1e-4)).collect();
        steps.push((0.98_f64.powi(400), 1.0));
        steps.extend((401..801).map(|k| (0.98_f64.powi(k), 1e-4)));

        let diverging = run(&steps);

        assert!(diverging.iter().all(Option::is_none));
    }

    #[test]
    /// Heavily damped steps inside the less accurate band don't diverge
    fn damped_steps_near_test() {
        let steps: Vec<_> = (0..500).map(|k| (NEAR * 0.98_f64.powi(k), 1e-4)).collect();

        let diverging = run(&steps);

        assert!(diverging.iter().all(Option::is_none));
    }

    /// Run `update` for each error with full steps, return the tracker
    fn track(errors: &[f64]) -> Tracker {
        let mut tracker = tracker();
        for &err in errors {
            tracker.update(step(err));
        }
        tracker
    }

    #[test]
    /// An improvement of the best error by 1% resets the hover count, and the
    /// error becomes the new hover level
    fn hover_reset_by_improvement_test() {
        let tracker = track(&[1.0, 2.0, 3.0, 0.5]);

        assert_eq!(tracker.num_steps_hovering, 0);
        assert_eq!(tracker.hover_error, 0.5);
    }

    #[test]
    /// Errors within 10 times of the hover level count as hovering, the level
    /// stays while the error is above it
    fn hover_count_test() {
        let tracker = track(&[1.0, 2.0, 3.0, 9.0]);

        assert_eq!(tracker.num_steps_hovering, 3);
        assert_eq!(tracker.hover_error, 1.0);
    }

    #[test]
    /// While hovering, an error below the hover level lowers it. 0.995 is no
    /// improvement of the best error 1.0 by 1%.
    fn hover_level_lowered_test() {
        let tracker = track(&[1.0, 2.0, 0.995]);

        assert_eq!(tracker.num_steps_hovering, 2);
        assert_eq!(tracker.hover_error, 0.995);
    }

    #[test]
    /// A jump above 10 times the hover level resets the hover count and sets
    /// a new hover level
    fn hover_jump_test() {
        let tracker = track(&[1.0, 2.0, 20.0]);
        assert_eq!(tracker.num_steps_hovering, 0);
        assert_eq!(tracker.hover_error, 20.0);
        assert_eq!(
            tracker.num_steps_without_improvement, 2,
            "hovering and jumping are no improvement"
        );

        let tracker = track(&[1.0, 2.0, 20.0, 30.0]);
        assert_eq!(tracker.num_steps_hovering, 1);
        assert_eq!(tracker.hover_error, 20.0);
    }

    /// Update `tracker` with each step and return `decide(false)` after each
    fn decisions(mut tracker: Tracker, steps: &[Step]) -> Vec<Decision> {
        steps
            .iter()
            .map(|&step| {
                tracker.update(step);
                tracker.decide(false)
            })
            .collect()
    }

    /// The single decision after one `step` from the initial error `err`
    fn decide_one(err: f64, step: Step, determinant_zero: bool) -> Decision {
        let settings = Settings::new(10).with_f_tol(F_TOL).with_x_tol(X_TOL);
        let mut tracker = Tracker::new(&settings, err, err);
        tracker.update(step);
        tracker.decide(determinant_zero)
    }

    /// A step to the error `err` with the squared step length `dy`
    fn step_to(err: f64, dy: f64) -> Step {
        Step {
            dy,
            dy_scaled: dy,
            ..step(err)
        }
    }

    #[test]
    /// A decreasing error far from the solution goes on
    fn decide_continue_test() {
        let decision = decide_one(2.0, step(1.0), false);
        assert!(matches!(decision, Decision::Continue), "got {decision:?}");
    }

    #[test]
    /// A diverging iteration fails
    fn decide_diverging_test() {
        let steps: Vec<_> = (0..22).map(|k| step(1e-3 * 11_f64.powi(k))).collect();
        let decisions = decisions(tracker(), &steps);

        assert!(matches!(decisions[20], Decision::Continue));
        assert!(
            matches!(decisions[21], Decision::Failed(NewtonError::Diverging)),
            "got {:?}",
            decisions[21]
        );
    }

    #[test]
    /// Error and step below their tolerances solve the system with the new
    /// iterate
    fn decide_solved_test() {
        let decision = decide_one(1.0, step_to(1e-30, 1e-30), false);
        assert!(
            matches!(
                decision,
                Decision::Converged {
                    status: NewtonStatus::Solved,
                    take_step: true
                }
            ),
            "got {decision:?}"
        );
    }

    #[test]
    /// A solution that is not better than the previous iterate keeps the
    /// previous iterate
    fn decide_solved_keeps_previous_test() {
        let decision = decide_one(1e-30, step_to(2e-30, 1e-30), false);
        assert!(
            matches!(
                decision,
                Decision::Converged {
                    status: NewtonStatus::Solved,
                    take_step: false
                }
            ),
            "got {decision:?}"
        );
    }

    #[test]
    /// A vanishing determinant fails as singular, unless the step solved the
    /// system
    fn decide_singular_test() {
        let decision = decide_one(2.0, step(1.0), true);
        assert!(
            matches!(decision, Decision::Failed(NewtonError::Singular)),
            "got {decision:?}"
        );

        let decision = decide_one(1.0, step_to(1e-30, 1e-30), true);
        assert!(
            matches!(decision, Decision::Converged { .. }),
            "got {decision:?}"
        );
    }

    #[test]
    /// The error stopped decreasing with a large step, but the previous error
    /// was already below `f_tol²`: the previous iterate is the solution
    fn decide_previous_was_solution_test() {
        let decision = decide_one(1e-30, step(2e-30), false);
        assert!(
            matches!(
                decision,
                Decision::Converged {
                    status: NewtonStatus::Solved,
                    take_step: false
                }
            ),
            "got {decision:?}"
        );
    }

    #[test]
    /// More iterations than the maximum fail
    fn decide_max_iterations_test() {
        let settings = Settings::new(10)
            .with_f_tol(F_TOL)
            .with_x_tol(X_TOL)
            .with_maximum_iterations(3);
        let tracker = Tracker::new(&settings, f64::INFINITY, f64::INFINITY);
        let steps: Vec<_> = (0..4).map(|k| step(0.5_f64.powi(k))).collect();
        let decisions = decisions(tracker, &steps);

        assert!(matches!(decisions[2], Decision::Continue));
        assert!(
            matches!(decisions[3], Decision::Failed(NewtonError::MaxIterations)),
            "got {:?}",
            decisions[3]
        );
    }

    #[test]
    /// A step below `x_tol` far from the solution is stuck
    fn decide_stuck_test() {
        let decision = decide_one(2.0, step_to(1.0, 1e-30), false);
        assert!(
            matches!(decision, Decision::Failed(NewtonError::Stuck)),
            "got {decision:?}"
        );
    }

    #[test]
    /// A step below `x_tol` inside the less accurate band, but above `f_tol²`,
    /// is a less accurate solution at the previous iterate
    fn decide_less_accurate_test() {
        let decision = decide_one(f64::INFINITY, step_to(1e-20, 1e-30), false);
        assert!(
            matches!(
                decision,
                Decision::Converged {
                    status: NewtonStatus::SolvedLessAccurate,
                    take_step: false
                }
            ),
            "got {decision:?}"
        );
    }

    #[test]
    /// More than 20 small steps (below `x_tol²·10⁴`, above `x_tol²`) stop the
    /// iteration, here far from a solution
    fn decide_small_steps_test() {
        let steps: Vec<_> = (0..21).map(|k| step_to(0.5_f64.powi(k), 1e-22)).collect();
        let decisions = decisions(tracker(), &steps);

        assert!(matches!(decisions[19], Decision::Continue));
        assert!(
            matches!(decisions[20], Decision::Failed(NewtonError::Stuck)),
            "got {:?}",
            decisions[20]
        );
    }

    #[test]
    /// An error alternating inside the less accurate band hovers. Once it
    /// hovered more than 20 steps, the iteration stops at the next bottom of
    /// the cycle, with the new iterate.
    fn decide_cycle_bottom_test() {
        let steps: Vec<_> = (0..23)
            .map(|k| step(if k % 2 == 0 { 1e-20 } else { 2e-20 }))
            .collect();
        let decisions = decisions(tracker(), &steps);

        assert!(
            decisions[..22]
                .iter()
                .all(|decision| matches!(decision, Decision::Continue)),
            "got {decisions:?}"
        );
        assert!(
            matches!(
                decisions[22],
                Decision::Converged {
                    status: NewtonStatus::SolvedLessAccurate,
                    take_step: true
                }
            ),
            "got {:?}",
            decisions[22]
        );
    }

    #[test]
    /// The number of small steps is remembered while the error doesn't
    /// hover, not while it hovers
    fn small_steps_at_hover_test() {
        let mut tracker = tracker();
        // 1.0 sets the hover level, 2.0 and 3.0 hover around it
        for (err, dy) in [(1.0, 1e-22), (2.0, 1.0), (3.0, 1e-22)] {
            tracker.update(step_to(err, dy));
        }

        assert_eq!(tracker.num_small_steps, 2);
        assert_eq!(tracker.num_small_steps_at_hover, 1);
    }

    /// The single decision after one `step` from the initial errors `err` and
    /// `err_scaled`
    fn decide_one_scaled(err: f64, err_scaled: f64, step: Step) -> Decision {
        let settings = Settings::new(10).with_f_tol(F_TOL).with_x_tol(X_TOL);
        let mut tracker = Tracker::new(&settings, err, err_scaled);
        tracker.update(step);
        tracker.decide(false)
    }

    /// Whether `decision` is a solution with the new or the previous iterate
    fn is_solved(decision: &Decision, take_step: bool) -> bool {
        matches!(decision, Decision::Converged { status: NewtonStatus::Solved, take_step: t } if *t == take_step)
    }

    #[test]
    /// The plain or the scaled error below `f_tol²` is enough to solve, with
    /// a step below `x_tol²`
    fn decide_solved_one_error_test() {
        let plain = Step {
            err: 1e-30,
            err_scaled: 1.0,
            ..step_to(1e-30, 1e-30)
        };
        let scaled = Step {
            err: 1.0,
            err_scaled: 1e-30,
            ..step_to(1e-30, 1e-30)
        };

        let decision = decide_one(2.0, plain, false);
        assert!(is_solved(&decision, true), "plain: got {decision:?}");
        let decision = decide_one(2.0, scaled, false);
        assert!(is_solved(&decision, true), "scaled: got {decision:?}");
    }

    #[test]
    /// The plain or the scaled step below `x_tol²` is enough to solve, with an
    /// error below `f_tol²`
    fn decide_solved_one_step_test() {
        let plain = Step {
            dy_scaled: 1.0,
            ..step_to(1e-30, 1e-30)
        };
        let scaled = Step {
            dy: 1.0,
            ..step_to(1e-30, 1e-30)
        };

        let decision = decide_one(2.0, plain, false);
        assert!(is_solved(&decision, true), "plain: got {decision:?}");
        let decision = decide_one(2.0, scaled, false);
        assert!(is_solved(&decision, true), "scaled: got {decision:?}");
    }

    #[test]
    /// The error stopped decreasing and the plain or the scaled previous error
    /// was below `f_tol²`: the previous iterate is the solution
    fn decide_previous_one_error_was_solution_test() {
        // Previous plain error 1e-30 and scaled 1, the step doesn't decrease
        // the plain error and has a large scaled error and step
        let decision = decide_one_scaled(
            1e-30,
            1.0,
            Step {
                err_scaled: 2.0,
                ..step(2e-30)
            },
        );
        assert!(is_solved(&decision, false), "plain: got {decision:?}");

        // Previous plain error 1 and scaled 1e-30, the step doesn't decrease
        // the plain error
        let decision = decide_one_scaled(1.0, 1e-30, step(2.0));
        assert!(is_solved(&decision, false), "scaled: got {decision:?}");
    }

    #[test]
    /// A step is small if its plain or its scaled length is below
    /// `x_tol²·10⁴`, more than 20 of them stop the iteration
    fn decide_small_steps_one_length_test() {
        for (dy, dy_scaled) in [(1e-22, 1.0), (1.0, 1e-22)] {
            let steps: Vec<_> = (0..21)
                .map(|k| Step {
                    dy,
                    dy_scaled,
                    ..step(0.5_f64.powi(k))
                })
                .collect();
            let decisions = decisions(tracker(), &steps);

            assert!(
                matches!(decisions[20], Decision::Failed(NewtonError::Stuck)),
                "dy {dy}, scaled {dy_scaled}: got {:?}",
                decisions[20]
            );
        }
    }
}
