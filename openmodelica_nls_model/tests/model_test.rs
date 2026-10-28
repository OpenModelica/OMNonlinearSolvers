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

//! How to implement `ModelFunctions` for a nonlinear system: square and
//! non-square systems, systems with user data, constraints and failing
//! evaluations.

use faer::Mat;
use openmodelica_nls_model::{EvalError, ModelFunctions, SystemInfo};

#[test]
/// Evaluate `F(x) = x² - 1` at `x = -1.0`
fn eval_nonlinear_system() {
    /// `F(x) = x² - 1`
    struct F {
        info: SystemInfo,
    }
    impl ModelFunctions for F {
        fn info(&self) -> &SystemInfo {
            &self.info
        }

        fn eval(&mut self, x: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
            res[0] = x[0] * x[0] - 1.0;
            Ok(())
        }
    }

    let mut model = F {
        info: SystemInfo::new(1, 1),
    };
    let mut res = [0.0];

    let x = [-1.0];
    model.eval(&x, &mut res).unwrap();

    assert_eq!(res, [0.0]);
}

#[test]
/// Evaluate `F(x) = x² - 1` at `x = -1.0`
fn test_with_user_data() {
    /// `F(x) = x² - 1`
    struct F {
        info: SystemInfo,
        counter_f_eval: usize,
    }
    impl ModelFunctions for F {
        fn info(&self) -> &SystemInfo {
            &self.info
        }

        fn eval(&mut self, x: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
            self.counter_f_eval += 1;
            res[0] = x[0] * x[0] - 1.0;
            Ok(())
        }
    }

    let mut model = F {
        info: SystemInfo::new(1, 1),
        counter_f_eval: 0,
    };
    let mut res = [0.0];

    let x = [-1.0];
    model.eval(&x, &mut res).unwrap();
    assert_eq!(res[0], 0.0);
    let x = [1.0];
    model.eval(&x, &mut res).unwrap();
    assert_eq!(res[0], 0.0);
    let x = [2.0];
    model.eval(&x, &mut res).unwrap();
    assert_eq!(res[0], 3.0);

    assert_eq!(model.counter_f_eval, 3);
}

#[test]
/// Evaluate a non-square system with 2 residuals and 3 unknowns, shaped
/// like a homotopy `H(x, λ)` with `λ` as additional unknown:
/// ```text
/// H(x, y, λ) = ( x² + y - λ   )
///              ( x·y·λ - 2    )
/// ```
/// at `(x, y, λ) = (1, 2, 1)` and `(2, -1, 0.5)`
fn eval_non_square() {
    /// `H: ℝ³ → ℝ²`
    struct H {
        info: SystemInfo,
    }
    impl ModelFunctions for H {
        fn info(&self) -> &SystemInfo {
            &self.info
        }

        fn eval(&mut self, x_vec: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
            let [x, y, lambda] = [x_vec[0], x_vec[1], x_vec[2]];

            res[0] = x * x + y - lambda;
            res[1] = x * y * lambda - 2.0;
            Ok(())
        }
    }

    let mut model = H {
        info: SystemInfo::new(2, 3),
    };
    let mut res = [0.0; 2];

    model.eval(&[1.0, 2.0, 1.0], &mut res).unwrap();
    assert_eq!(res, [2.0, 0.0]);

    model.eval(&[2.0, -1.0, 0.5], &mut res).unwrap();
    assert_eq!(res, [2.5, -3.0]);
}

#[test]
/// A casual tearing set with a constraint: `F(x) = x² - 4` with `x ≥ 0`.
///
/// Without the constraint both `x = 2` and `x = -2` are roots, the
/// constraint rules out the negative one. The model checks it in `eval`
/// and reports a violation as `EvalError::ConstraintViolated`, which the
/// default `derivative` passes on.
fn eval_constrained() {
    /// `F(x) = x² - 4`, valid for `x ≥ 0`
    struct F {
        info: SystemInfo,
    }
    impl ModelFunctions for F {
        fn info(&self) -> &SystemInfo {
            &self.info
        }

        fn eval(&mut self, x: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
            if x[0] < 0.0 {
                return Err(EvalError::ConstraintViolated);
            }
            res[0] = x[0] * x[0] - 4.0;
            Ok(())
        }
    }

    let mut model = F {
        info: SystemInfo::new(1, 1).with_casual_tearing_set(),
    };
    let mut res = [0.0];
    let mut jac = Mat::<f64>::zeros(1, 1);

    assert!(model.info().casual_tearing_set);

    // The root inside the constraint
    model.eval(&[2.0], &mut res).unwrap();
    assert!(res == [0.0]);

    // The root outside the constraint is not a valid point
    let result = model.eval(&[-2.0], &mut res);
    assert!(matches!(result, Err(EvalError::ConstraintViolated)));

    // On the boundary the forward difference stays inside the constraint
    model.eval(&[0.0], &mut res).unwrap();
    model
        .derivative(&[0.0], &res.clone(), &[1.0], jac.as_mut())
        .unwrap();

    // Outside the constraint the derivative fails as well
    let result = model.derivative(&[-1.0], &[-3.0], &[1.0], jac.as_mut());
    assert!(matches!(result, Err(EvalError::ConstraintViolated)));
}

#[test]
#[should_panic = "`Result::unwrap()` on an `Err` value: Assert"]
/// Test evaluation of function with division by zero
fn test_division_by_zero() {
    /// `F(x) = 1/x - 2`
    struct F {
        info: SystemInfo,
    }
    impl ModelFunctions for F {
        fn info(&self) -> &SystemInfo {
            &self.info
        }

        fn eval(&mut self, x: &[f64], res: &mut [f64]) -> Result<(), EvalError> {
            match x[0] {
                0.0 => Err(EvalError::Assert),
                _ => {
                    res[0] = 1.0 / x[0] - 2.0;
                    Ok(())
                }
            }
        }
    }

    let mut model = F {
        info: SystemInfo::new(1, 1),
    };
    let mut res = [0.0];

    let x = [0.0];
    model.eval(&x, &mut res).unwrap();
}
