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

//! Solve a nonlinear system with the homotopy solver.
//!
//! Run with `cargo run --example solve -p openmodelica_homotopy`.

use openmodelica_homotopy::HomotopySolver;
use openmodelica_homotopy::settings::Settings;
use openmodelica_newton::settings::Settings as NewtonSettings;
use openmodelica_nls_model::{EvalError, ModelFunctions, SystemInfo};

/// The system `F(x) = x³ − 2x + 2 = 0`, with its only real root at
/// `x ≈ −1.7693`. From the start point `x = 0`, Newton's method alone gets
/// stuck in the local minimum of `|F|` at `x ≈ 0.82`.
struct Cubic {
    info: SystemInfo,
}

impl ModelFunctions for Cubic {
    fn info(&self) -> &SystemInfo {
        &self.info
    }

    fn eval(&mut self, x: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
        res[0] = x[0].powi(3) - 2.0 * x[0] + 2.0;
        Ok(())
    }

    // The Jacobian defaults to forward differences. A system with an
    // analytical Jacobian overrides `derivative`.
}

fn main() {
    let size = 1;
    let mut model = Cubic {
        info: SystemInfo::new(size, size),
    };
    let mut solver = HomotopySolver::new(size, NewtonSettings::new(size), Settings::new(size));

    let mut x = [0.0];
    match solver.solve(&mut model, &mut x) {
        Ok(how) => println!("x = {} ({how:?})", x[0]),
        Err(error) => println!("no solution: {error:?}"),
    }
}
