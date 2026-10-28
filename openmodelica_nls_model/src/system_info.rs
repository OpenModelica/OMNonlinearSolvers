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

//! Facts about a nonlinear system, independent of how it is evaluated.

/// Sizes, nominal values, bounds and tearing kind of a nonlinear system.
///
/// `nominal`, `min` and `max` have one entry per unknown. [`SystemInfo::new`]
/// creates them with the defaults: nominal values of 1, no bounds and no
/// casual tearing set. The `with_*` methods set them and check their lengths.
///
/// # Example
///
/// ```
/// use openmodelica_nls_model::SystemInfo;
///
/// let info = SystemInfo::new(2, 2).with_min([0.0, f64::NEG_INFINITY]);
///
/// assert_eq!(*info.nominal, [1.0, 1.0]);
/// assert_eq!(*info.min, [0.0, f64::NEG_INFINITY]);
/// assert_eq!(*info.max, [f64::INFINITY, f64::INFINITY]);
/// ```
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Number of residuals, the length of `F(x)` and the number of rows of
    /// the Jacobian
    pub num_residuals: usize,
    /// Number of unknowns, the length of `x` and the number of columns of the
    /// Jacobian
    pub num_unknowns: usize,
    /// Nominal value of each unknown, used by the solvers to scale `x`
    pub nominal: Box<[f64]>,
    /// Lower bound of each unknown, negative infinity for no bound
    pub min: Box<[f64]>,
    /// Upper bound of each unknown, positive infinity for no bound
    pub max: Box<[f64]>,
    /// The system is the casual set of a dynamic tearing
    pub casual_tearing_set: bool,
}

impl SystemInfo {
    /// Creates the info of a system with `num_residuals` residuals and
    /// `num_unknowns` unknowns, with nominal values of 1, no bounds and no
    /// casual tearing set.
    pub fn new(num_residuals: usize, num_unknowns: usize) -> Self {
        SystemInfo {
            num_residuals,
            num_unknowns,
            nominal: vec![1.0; num_unknowns].into_boxed_slice(),
            min: vec![f64::NEG_INFINITY; num_unknowns].into_boxed_slice(),
            max: vec![f64::INFINITY; num_unknowns].into_boxed_slice(),
            casual_tearing_set: false,
        }
    }

    /// Marks the system as the casual set of a dynamic tearing.
    pub fn with_casual_tearing_set(mut self) -> Self {
        self.casual_tearing_set = true;
        self
    }

    /// Sets the nominal values of the unknowns.
    ///
    /// # Panics
    ///
    /// Panics if `nominal` doesn't have `num_unknowns` entries.
    pub fn with_nominal(mut self, nominal: impl Into<Box<[f64]>>) -> Self {
        self.nominal = self.checked(nominal.into());
        self
    }

    /// Sets the lower bounds of the unknowns.
    ///
    /// # Panics
    ///
    /// Panics if `min` doesn't have `num_unknowns` entries.
    pub fn with_min(mut self, min: impl Into<Box<[f64]>>) -> Self {
        self.min = self.checked(min.into());
        self
    }

    /// Sets the upper bounds of the unknowns.
    ///
    /// # Panics
    ///
    /// Panics if `max` doesn't have `num_unknowns` entries.
    pub fn with_max(mut self, max: impl Into<Box<[f64]>>) -> Self {
        self.max = self.checked(max.into());
        self
    }

    /// Returns `values` after checking that it has one entry per unknown.
    fn checked(&self, values: Box<[f64]>) -> Box<[f64]> {
        assert_eq!(
            values.len(),
            self.num_unknowns,
            "Expected one value per unknown"
        );
        values
    }
}

#[cfg(test)]
mod system_info_tests {
    use super::*;

    #[test]
    /// `SystemInfo::new` has the defaults: nominal values of 1, no bounds and no
    /// casual tearing set
    fn system_info_defaults() {
        let info = SystemInfo::new(2, 3);

        assert!(info.num_residuals == 2);
        assert!(info.num_unknowns == 3);
        assert_eq!(*info.nominal, [1.0; 3]);
        assert_eq!(*info.min, [f64::NEG_INFINITY; 3]);
        assert_eq!(*info.max, [f64::INFINITY; 3]);
        assert!(!info.casual_tearing_set);
    }

    #[test]
    /// The `with_*` methods of `SystemInfo` set the values and the tearing kind
    fn system_info_with() {
        let info = SystemInfo::new(1, 2)
            .with_nominal([1.0, 1e3])
            .with_min([0.0, -1.0])
            .with_max([1.0, 1.0])
            .with_casual_tearing_set();

        assert_eq!(*info.nominal, [1.0, 1e3]);
        assert_eq!(*info.min, [0.0, -1.0]);
        assert_eq!(*info.max, [1.0, 1.0]);
        assert!(info.casual_tearing_set);
    }
}
