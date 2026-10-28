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

//! The homotopy algorithm of `nonlinearSolverHomotopy.c`, built on the Newton
//! solver of the `openmodelica_newton` crate.
//!
//! [`HomotopySolver`] solves a system `F(x) = 0` like the C runtime: Newton's
//! method from the start point first, and if that fails Newton's method from
//! the end of the path of a homotopy. The system implements
//! [`openmodelica_nls_model::ModelFunctions`].
//!
//! # Example
//!
//! `x³ − 2x + 2 = 0` from `x = 0`, where Newton's method alone gets stuck:
//!
//! ```
//! use openmodelica_homotopy::settings::Settings;
//! use openmodelica_homotopy::{HomotopyRun, HomotopySolver, Solved};
//! use openmodelica_newton::settings::Settings as NewtonSettings;
//! use openmodelica_nls_model::ClosureModel;
//!
//! let mut model = ClosureModel::new(1, 1, |x: &[f64], res: &mut [f64]| {
//!     res[0] = x[0].powi(3) - 2.0 * x[0] + 2.0;
//!     Ok(())
//! });
//! let mut solver = HomotopySolver::new(1, NewtonSettings::new(1), Settings::new(1));
//!
//! let mut x = [0.0];
//! let how = solver.solve(&mut model, &mut x).unwrap();
//!
//! assert_eq!(how, Solved::Homotopy(HomotopyRun::NewtonNegative));
//! assert!((x[0] + 1.769_292_354_238_631).abs() < 1e-10);
//! ```
//!
//! `openmodelica_homotopy/examples/solve.rs` shows the same system as a type of its
//! own.

pub mod settings;

mod corrector;
mod homotopy_function;
mod path_follower;
mod predictor;
mod step_control;
mod strategy;
mod tangent;

pub use homotopy_function::{FixpointHomotopy, NewtonHomotopy};
pub use path_follower::{Direction, NoObserver, PathError, PathObserver, follow_path};
pub use strategy::{
    Fallback, HomotopyError, HomotopyRun, HomotopySolver, NoFallback, Solved, Statistics,
};
