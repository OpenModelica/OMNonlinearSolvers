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

//! Detection of limit cycles in the residual error of the Newton iteration.

/// The last four residual errors of the Newton iteration, to detect a limit
/// cycle.
///
/// A limit cycle is an error that jumps back and forth between a few values
/// instead of decreasing, with a period of 2, 3 or 4 iterations. Port of the
/// error history `errorHist` and `newtonLimitCycleStep` of the C runtime.
pub(crate) struct ErrorHistory {
    /// The last errors, newest first. Only the first `len` entries are valid.
    values: [f64; 4],
    /// Number of valid entries in `values`, at most 4
    len: usize,
    /// Relative tolerance: two errors are equal if they differ by at most
    /// `tol` times the current error
    tol: f64,
}

impl ErrorHistory {
    /// Creates an empty history with a relative tolerance of 1%.
    pub(crate) fn new() -> Self {
        ErrorHistory {
            values: [0.0; 4],
            len: 0,
            tol: 1e-2,
        }
    }

    /// Adds the error `err` of the current iteration and returns whether it
    /// closes a limit cycle.
    ///
    /// The cycle is checked against the errors before `err` is added, as in
    /// `newtonLimitCycleStep` of the C runtime.
    pub(crate) fn push(&mut self, err: f64) -> bool {
        let cycle = self.is_cycle(err);
        self.push_front(err);
        cycle
    }

    /// Whether the history is full and its newest error is not above the three
    /// before it, the bottom of a limit cycle.
    pub(crate) fn newest_is_lowest(&self) -> bool {
        self.len == 4 && self.values[1..].iter().all(|&err| self.values[0] <= err)
    }

    /// Whether `err` closes a limit cycle, without changing the history.
    ///
    /// True if the history is full, `err` changed since the last error, and
    /// `err` returned to the error of 2, 3 or 4 iterations ago. Errors are
    /// compared relative to `err`, see [`ErrorHistory::tol`]. An error that
    /// doesn't change is no cycle, stagnation is detected elsewhere.
    fn is_cycle(&self, err: f64) -> bool {
        let history_full = self.len == 4;
        let error_changed = (err - self.values[0]).abs() > self.tol * err;
        let error_returned = self.values[1..]
            .iter()
            .any(|&old| (err - old).abs() <= self.tol * err);

        history_full && error_changed && error_returned
    }

    /// Adds `err` as the newest error. Once the history is full, the oldest
    /// error is dropped.
    fn push_front(&mut self, err: f64) {
        self.values.rotate_right(1);
        self.values[0] = err;
        self.len += if self.len < 4 { 1 } else { 0 };
    }
}

#[cfg(test)]
mod error_history_tests {
    use super::*;

    impl ErrorHistory {
        /// The valid errors, newest first
        fn values(&self) -> &[f64] {
            &self.values[0..self.len]
        }
    }

    #[test]
    /// `push_front` adds the newest error at the front and drops the oldest
    /// once the history is full
    fn error_history_push_test() {
        let mut history = ErrorHistory::new();

        assert_eq!(history.values().len(), 0);

        history.push_front(1.0);
        assert_eq!(history.values(), [1.0]);

        history.push_front(2.0);
        assert_eq!(history.values(), [2.0, 1.0]);

        history.push_front(3.0);
        assert_eq!(history.values(), [3.0, 2.0, 1.0]);

        history.push_front(4.0);
        assert_eq!(history.values(), [4.0, 3.0, 2.0, 1.0]);

        history.push_front(5.0);
        assert_eq!(history.values(), [5.0, 4.0, 3.0, 2.0]);
    }

    /// Push `errors` one after another, return the result of each `push`
    fn push_all(history: &mut ErrorHistory, errors: &[f64]) -> Vec<bool> {
        errors.iter().map(|&err| history.push(err)).collect()
    }

    #[test]
    /// `push` adds every error to the history, also when it detects a cycle
    fn push_adds_to_history_test() {
        let mut history = ErrorHistory::new();

        push_all(&mut history, &[1.0, 2.0, 1.0, 2.0]);
        assert_eq!(history.values(), [2.0, 1.0, 2.0, 1.0]);

        // The cycle is reported and 1.0 is added nevertheless
        assert!(history.push(1.0));
        assert_eq!(history.values(), [1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    /// The newest error is the lowest if it is not above any of the three
    /// before it, equal errors included. A history that isn't full has none.
    fn newest_is_lowest_test() {
        let mut history = ErrorHistory::new();
        push_all(&mut history, &[2.0, 1.0, 3.0]);
        assert!(!history.newest_is_lowest(), "history not full");

        history.push(1.0);
        assert!(history.newest_is_lowest(), "equal to the lowest");

        history.push(1.5);
        assert!(!history.newest_is_lowest());
    }

    #[test]
    /// No cycle while the history has less than 4 errors, even if the errors
    /// already alternate
    fn push_history_not_full_test() {
        let mut history = ErrorHistory::new();

        assert_eq!(
            push_all(&mut history, &[1.0, 2.0, 1.0, 2.0]),
            [false, false, false, false]
        );
    }

    #[test]
    /// Period 2: the error returns to the error of 2 iterations ago
    fn push_cycle_period_2_test() {
        let mut history = ErrorHistory::new();

        assert_eq!(
            push_all(&mut history, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]),
            [false, false, false, false, true, true]
        );
    }

    #[test]
    /// Period 3: the error returns to the error of 3 iterations ago
    fn push_cycle_period_3_test() {
        let mut history = ErrorHistory::new();

        assert_eq!(
            push_all(&mut history, &[1.0, 2.0, 3.0, 1.0, 2.0]),
            [false, false, false, false, true]
        );
    }

    #[test]
    /// Period 4: the error returns to the error of 4 iterations ago, the
    /// oldest one in the history
    fn push_cycle_period_4_test() {
        let mut history = ErrorHistory::new();

        assert_eq!(
            push_all(&mut history, &[1.0, 2.0, 3.0, 4.0, 1.0]),
            [false, false, false, false, true]
        );
    }

    #[test]
    /// A steadily decreasing error is no cycle
    fn push_decreasing_test() {
        let mut history = ErrorHistory::new();

        assert_eq!(
            push_all(&mut history, &[8.0, 4.0, 2.0, 1.0, 0.5, 0.25]),
            [false; 6]
        );
    }

    #[test]
    /// An error that doesn't change is no cycle, although it equals all
    /// older errors. Stagnation is detected elsewhere.
    fn push_constant_test() {
        let mut history = ErrorHistory::new();

        assert_eq!(push_all(&mut history, &[1.0; 6]), [false; 6]);
    }

    #[test]
    /// An error within 1% of the last error didn't change, so it is no cycle
    /// even if it also returned to an older error
    fn push_unchanged_within_tolerance_test() {
        let mut history = ErrorHistory::new();

        push_all(&mut history, &[2.0, 1.0, 2.0, 1.0]);
        assert!(!history.push(1.005));
    }

    #[test]
    /// An older error within 1% of the current error counts as returned to,
    /// one more than 1% away doesn't. The tolerance is relative to the
    /// current error.
    fn push_returned_tolerance_test() {
        let mut history = ErrorHistory::new();
        push_all(&mut history, &[2.0, 1.0, 2.0, 1.0]);
        assert!(history.push(2.0 * 1.005));

        let mut history = ErrorHistory::new();
        push_all(&mut history, &[2.0, 1.0, 2.0, 1.0]);
        assert!(!history.push(2.0 * 1.02));
    }

    #[test]
    /// The tolerance of an unchanged error is relative to the error too:
    /// `100.5` is within 1% of the last error `100`, so it is no cycle
    fn push_unchanged_relative_test() {
        let mut history = ErrorHistory::new();

        push_all(&mut history, &[200.0, 100.0, 200.0, 100.0]);
        assert!(!history.push(100.5));
    }
}
