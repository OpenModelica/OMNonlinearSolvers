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

//! The messages of the solver for a C host, see [`hom_set_logger`].

use std::ffi::{CString, c_char, c_int};
use std::sync::Mutex;

/// Messages with this level or a more severe one go to the logger: 1 error,
/// 2 warning, 3 info, 4 debug and 5 trace.
pub type HomLogLevel = c_int;

/// Receive a `message` of the solver with its `level` (see [`HomLogLevel`]),
/// and its `target`, for example `NLS` for Newton's method or `NLS_HOMOTOPY`
/// for the homotopy. Both strings are only valid during the call.
pub type HomLogger =
    Option<unsafe extern "C" fn(level: HomLogLevel, target: *const c_char, message: *const c_char)>;

static LOGGER: Mutex<HomLogger> = Mutex::new(None);

struct CLogger;

impl log::Log for CLogger {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        LOGGER.lock().is_ok_and(|logger| logger.is_some())
    }

    fn log(&self, record: &log::Record<'_>) {
        let Ok(logger) = LOGGER.lock() else {
            return;
        };
        let Some(callback) = *logger else {
            return;
        };
        let target = CString::new(record.target()).unwrap_or_default();
        let message = CString::new(record.args().to_string()).unwrap_or_default();
        // SAFETY: both strings stay valid during the call
        unsafe { callback(record.level() as c_int, target.as_ptr(), message.as_ptr()) };
    }

    fn flush(&self) {}
}

static C_LOGGER: CLogger = CLogger;

/// Send the messages of the solver with at most `max_level` (see
/// [`HomLogLevel`]) to `logger`, for all solvers of the process. `NULL` or a
/// level of 0 switches the messages off. Replaces an earlier logger.
///
/// The solver uses the Rust `log` crate. If the process already has another
/// Rust logger, the messages go there instead.
///
/// # Safety
///
/// `logger` must be `NULL` or a function that can be called from any thread
/// that solves.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hom_set_logger(logger: HomLogger, max_level: HomLogLevel) {
    let _ = log::set_logger(&C_LOGGER);
    if let Ok(mut current) = LOGGER.lock() {
        *current = logger;
    }
    let filter = match (logger, max_level) {
        (None, _) | (_, ..=0) => log::LevelFilter::Off,
        (_, 1) => log::LevelFilter::Error,
        (_, 2) => log::LevelFilter::Warn,
        (_, 3) => log::LevelFilter::Info,
        (_, 4) => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    log::set_max_level(filter);
}
