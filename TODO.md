# TODO

* Offer hybrd of Minpack to pure Rust users. The OpenModelica glue passes
  `solveHybrd` of the C runtime as the fallback of the C interface, a Rust
  user could pass `SimulationRuntime/rust/minpack` of OpenModelica as a
  `openmodelica_homotopy::Fallback`. Also check whether hybrj, with the analytical
  Jacobian, is worth it.
* Test how our implementation compares to a newton algorithm from
 [newton_rootfinder][newton_rootfinder] or [nrfind][nrfind] (or similar)
* Move `ClosureModel` from `openmodelica_nls_model` into a test-support crate,
  e.g. `openmodelica_nls_test_problems`, used only as dev-dependency, so it is
  no longer part of the public API of `openmodelica_nls_model`. The crate can
  also collect standard test problems for nonlinear solvers (e.g. Rosenbrock's
  function, Powell's singular function, Broyden's tridiagonal function) and
  small systems from OpenModelica models, shared by the tests of
  `openmodelica_newton`, `openmodelica_homotopy` and regression tests against
  the C solver. Do this when the Newton tests need
  the same test systems in several crates.
* `SystemInfo` has public members, so a caller could input wrong values. Add
  getters.
* Port corrector strategy 2 of `homotopyAlgorithm` (`homBacktraceStrategy`):
  back to the path orthogonally to the tangent, with the extended system of
  `orthogonalBacktraceMatrix`. Only strategy 1, the default, fixing one
  coordinate, is ported.
* Fix the scaling of the fixpoint homotopy's Jacobian once there is a test
  oracle against the C runtime of OpenModelica. C scales the columns of `J`
  before adding the identity, so the path following solves with
  `λ·J·S + (1 − λ)·I` instead of `(λ·J + (1 − λ)·I)·S`. `FixpointHomotopy`
  reproduces that, see its doc comment.
* Call `ModelFunctions::start_point_checked` after `F(x₀)` instead of after
  the Jacobian at `x₀`, together with the C runtime. C evaluates the start
  Jacobian of a mixed system with switching relations, so near a switching
  point its first Newton step and its relations backup belong to another
  branch, and the system is solved twice. The port does the same to give the
  same results.
* Remove the CSV export of the homotopy path later on. The OpenModelica glue
  writes it with `LOG_INIT_HOMOTOPY` like the C runtime did
  (`<model>_nonlinsys<n>_adaptive_<global|local>_homotopy_<pos|neg>.csv`), but
  it looks like a debugging feature that isn't used any more. It is kept for
  now to stay comparable with the C runtime.

[newton_rootfinder]: https://github.com/Nateckert/newton_rootfinder
[nrfind]: https://github.com/NoraCodes/nrfind
