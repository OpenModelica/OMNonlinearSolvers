# C Program Layout

Structure of [`nonlinearSolverHomotopy.c`][c-source] from the OpenModelica C
runtime, the code this project ports to Rust.

Functions marked with ✔ are already ported.

## Class Diagram

`DATA_HOMOTOPY` is one struct holding everything: the work buffers of three
algorithms, configuration, statistics, and "methods" in the form of function
pointers. Its fields are grouped below by the part of the solver that uses
them.

```text
┌──────────────────────────── DATA_HOMOTOPY ─────────────────────────────┐
│ size            n, m (= n+1)                                           │
│ tolerances      xtol_sqrd, ftol_sqrd, maxNumberOfIterations            │
│ scaling         xScaling, resScaling, fvecScaled, hvecScaled, dxScaled │
│ bounds          minValue*, maxValue*          (borrowed from NLS data) │
│ Newton          x, x0, x1, xStart, finit, fx0, f1, f2, gradFx,         │
│                 fJac, fJacx0                                           │
│ homotopy        y0, y1, y2, yt, dy0, dy1, dy2, hvec, hJac, hJac2,      │
│                 hJacInit, ones, tau, startDirection, initHomotopy      │
│ linear solver   indRow, indCol                                         │
│ statistics      info, numberOfIterations, numberOfFunctionEvaluations  │
│ context         eqSystemNumber, timeValue, mixedSystem,                │
│                 casualTearingSet                                       │
│ debug           debug_fJac, debug_dx                                   │
├────────────────────────────────────────────────────────────────────────┤
│ "methods" (function pointers, chosen at run time)                      │
│   f        --> wrapper_fvec                  residual F(x)             │
│   f_con    --> wrapper_fvec_constraints      residual + constraints    │
│   fJac_f   --> wrapper_fvec_der              Jacobian (analyt./numer.) │
│   h_function -> wrapper_fvec  │ _homotopy_newton  │ _homotopy_fixpoint │
│   hJac_dh  --> wrapper_fvec_der │ _homotopy_newton_der │ _fixpoint_der │
└──────┬────────────────────────────────────────────────┬────────────────┘
       │ userData                                       │ dataHybrid
       ▼                                                ▼
┌──────────────── NLS_USERDATA ─────────────┐   ┌──── DATA_HYBRD ─────┐
│ data* ------> DATA (simulation state)     │   │ MINPACK fallback    │
│ threadData* (jump buffer for asserts)     │   │ (nonlinearSolver-   │
│ nlsData* ---> NONLINEAR_SYSTEM_DATA       │   │  Hybrd.c)           │
│ analyticJacobian* -> JACOBIAN             │   └─────────────────────┘
└───────────────────────────────────────────┘
```

All functions in the file work on that one struct, in five layers:

```text
 entry       solveHomotopy ─ allocateHomotopyData / freeHomotopyData / getHomotopyJacobian
 algorithms  newtonAlgorithm      homotopyAlgorithm         (+ solveHybrd fallback)
 model       wrapper_fvec*  getAnalyticalJacobianHomotopy  getNumericalJacobianHomotopy
 numerics    solveSystemWithTotalPivotSearch ✔   linearSolverWrapper   newtonLimitCycleStep
             orthogonalBacktraceMatrix   scaleMatrixRows   ~20 vec*/mat* helpers
 output      print*Step  debugMatrix*  debugVector*
```

## Flow Diagram

### Solve Homotopy

`solveHomotopy` is the entry point called by `nonlinearSystem.c`. It tries the
start point, then Newton, MINPACK's hybrd as fallback and finally up to three
homotopy runs.

```text
             ┌──────────────────────────────────────────────┐
             │ setup: start point x0 = nlsx or extrapolation│
             │        λ-entry, xScaling from nominal        │
             └──────────────────────┬───────────────────────┘
                                    │  (skipped for initHomotopy)
                                    ▼
      ┌──── "check start point", up to 3 tries ────────────────────────┐
      │  F(x0)  ── ‖F‖ tiny? ───────────────────────────> SOLVED       │
      │  J(x0), scale rows, first Newton step (total pivot)            │
      │  model assert? ─> perturb x0 (try 2, 3), casual set ─> give up │
      └───────────────────────────────┬────────────────────────────────┘
                                      ▼
  ┌──────────────── main loop: while !giveUp && !SOLVED ──────────────────┐
  │                                                                       │
  │   newtonAlgorithm(x)  ─ failed? ─> solveHybrd (MINPACK)               │
  │        │ (skipped after a failed homotopy run)                        │
  │        ▼                                                              │
  │   solved? ── mixed system: re-evaluate, relations changed? ─> retry   │
  │        │ yes ─────────────────────────────────────────────> SOLVED    │
  │        ▼ no                                                           │
  │   runHomotopy += 1, choose H(x, λ):                                   │
  │       init homotopy:   run 1: F, start +λ   run 2: F, start −λ        │
  │       otherwise:       run 1: Newton homotopy +   run 2: Newton −     │
  │                        run 3: fixpoint homotopy                       │
  │        ▼                                                              │
  │   homotopyAlgorithm(x)                                                │
  │        │ failed ─> skipNewton, next run (give up after 3)             │
  │        │ ok, init homotopy ─> SOLVED                                  │
  │        │ ok, otherwise ─> J(x), first Newton step, back to Newton     │
  └────────┴──────────────────────────────────────────────────────────────┘
                                      ▼
                write statistics, return SOLVED / FAILED
```

### Newton Algorithm

`newtonAlgorithm` is a damped Newton-Raphson method. Each iteration:

1. Solves `J·dy = -f` with the total pivot search ✔ (the first step is solved
   by the caller).
2. Shrinks the step if the model can't be evaluated at `x + dy` (asserts,
   violated constraints of a casual tearing set).
3. Damps the step with a line search (quadratic, then cubic backtracking).
4. Tracks stagnation: growing error, limit cycles (`newtonLimitCycleStep`),
   stalls, hovering and creeping steps.
5. Decides whether to stop: solved, solved with less accuracy, maximum
   iterations reached or stuck.
6. Evaluates and scales the Jacobian at the new `x`.

### Homotopy Algorithm

Path tracking from λ = 0 to λ = 1. ✔ marks where the total pivot search is
used.

```text
  y0 = (x, λ=0)
  while λ < 1:
     ┌ PREDICTOR ──────────────────────────────────────────────────┐
     │ H'(y0), tangent dy0 = total pivot with pos = None  ✔        │
     │ orient dy0 (startDirection), step τ, clip to λ ≤ 1          │
     │ y1 = y0 + τ·dy0,  H(y1) fails? ─> shrink τ, retry           │
     └──────────────────────────────────┬──────────────────────────┘
     ┌ CORRECTOR (Newton on H = 0) ─────▼──────────────────────────┐
     │ strategy 1: fix coordinate `pos` (from the predictor)  ✔   │
     │ strategy 2: orthogonal to the tangent (n+1)×(n+2) system  ✔│
     │ until ‖H(y1)‖ < hEps or maxiter                             │
     └──────────────────────────────────┬──────────────────────────┘
     ┌ STEP CONTROL ────────────────────▼──────────────────────────┐
     │ bend = ‖dy1‖/‖dy2‖                                          │
     │ too bent or not accepted ─> τ /= factor (fail at τ_min)     │
     │ accepted ─> y0 = y1, maybe τ *= factor                      │
     └─────────────────────────────────────────────────────────────┘
  x = y0[0..n]
```

## Porting Order

Bottom-up along the call graph, starting with what every algorithm needs:

1. **Model interface**: a trait for "evaluate the residual" and "evaluate the
   Jacobian", where both can fail (model assert). It replaces the function
   pointers `f`, `f_con` and `fJac_f` and the `OMC_TRY` blocks. Together with
   the numerical Jacobian (`getNumericalJacobianHomotopy`) as default
   implementation and the row and column scaling (`scaleMatrixRows`,
   `matVecMultAbsBB`, `xScaling`, `resScaling`).
2. **Newton** (`newtonAlgorithm`) in the `openmodelica_newton` crate, split
   into line search, stagnation tracking (including `newtonLimitCycleStep`),
   stopping decision and Jacobian refresh.
3. **Homotopy** (`homotopyAlgorithm`): predictor, corrector and step control as
   separate functions. The homotopy variants (Newton, fixpoint) become an enum
   implementing the residual trait for `H(x, λ)`.
4. **Strategy** (`solveHomotopy`): "start point, Newton, hybrd, homotopy runs
   1 to 3" as an explicit state machine instead of the flags `giveUp`,
   `runHomotopy`, `skipNewton` and `alreadyTested`.

Not ported one by one:

* the about 20 `vec*` and `mat*` helpers, they become faer operations or
  iterators where they are used,
* the LAPACK path of `linearSolverWrapper`, it becomes faer's LU decomposition,
* the `print*` and `debug*` functions, they become `trace!` with `Display`.

The handling of mixed systems (relations) and `discreteCall` is runtime state,
not numerics, and stays in OpenModelica's glue to `openmodelica_homotopy_capi`.

[c-source]: https://github.com/OpenModelica/OpenModelica/blob/18d4ff455d1beceb4709d770ca4faf53d65732da/OMCompiler/SimulationRuntime/c/simulation/solver/nonlinearSolverHomotopy.c
