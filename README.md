# OMNonlinearSolvers

[![CI][ci-badge]][ci]

The nonlinear solvers of [OpenModelica][openmodelica], in Rust. The repository
is an experiment: can the nonlinear solvers of OpenModelica be split into
modules in Rust? It is meant to hold all nonlinear solvers developed for
OpenModelica, ported from its C run-time system one at a time, with the same
results as in C.

Solvers that only wrap an external library, like KINSOL of SUNDIALS or hybrd of
MINPACK, are not part of it for now. The linear solvers will get a repository
of their own.

| Solver   | Based on                                                   | Crate                   |
| -------- | ---------------------------------------------------------- | ----------------------- |
| Homotopy | [nonlinearSolverHomotopy.c][c-source] by Bernhard Bachmann | `openmodelica_homotopy` |

## Solve NLS

A nonlinear system `F(x) = 0` implements `ModelFunctions` of the `openmodelica_nls_model`
crate: its sizes and nominal values, and how to evaluate `F`. The Jacobian
defaults to forward differences. `HomotopySolver` of the `openmodelica_homotopy`
crate solves it like the C runtime: Newton's method from the start point, and
if that fails Newton's method from the end of a homotopy path.

```rust
use openmodelica_homotopy::HomotopySolver;
use openmodelica_homotopy::settings::Settings;
use openmodelica_newton::settings::Settings as NewtonSettings;

let mut solver = HomotopySolver::new(size, NewtonSettings::new(size), Settings::new(size));
let how = solver.solve(&mut model, &mut x)?;
```

[openmodelica_homotopy/examples/solve.rs](openmodelica_homotopy/examples/solve.rs)
is a complete example, run it with

```sh
cargo run --example solve -p openmodelica_homotopy
```

A system Newton's method solves on its own can also use `openmodelica_newton::solve`
directly.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to build, test and develop this package.
The design decisions behind the code are collected in [DESIGN.md](DESIGN.md).

## C Interface

The `openmodelica_homotopy_capi` crate builds a static library with a generic C
interface. Its header
`openmodelica_homotopy_capi/include/openmodelica_homotopy_capi.h` is generated
when the crate is built. A host describes its system as a `HomModel` of callbacks:

```c
#include <stdio.h>

#include "openmodelica_homotopy_capi.h"

/* F(x) = x³ − 2x + 2 */
static int residual(void *user, const double *x, double *f) {
  (void)user; /* the host's data, from model.user */
  f[0] = x[0] * x[0] * x[0] - 2.0 * x[0] + 2.0;
  return HOM_EVAL_OK; /* or HOM_EVAL_ASSERT, HOM_EVAL_CONSTRAINT_VIOLATED */
}

int main(void) {
  HomSettings settings = {0}; /* 0 keeps a default */
  HomSolver *solver = hom_solver_new(1, &settings);

  HomModel model = {0};
  model.num_residuals = model.num_unknowns = 1;
  model.residual = residual;

  double x[1] = {0.0}; /* the start point */
  HomSolvedBy solved_by;
  HomStatus status = hom_solve(solver, &model, x, &solved_by);
  if (status == HOM_STATUS_SOLVED) {
    printf("x = %g\n", x[0]);
  }

  hom_solver_free(solver);
  return status == HOM_STATUS_SOLVED ? 0 : 1;
}
```

The callbacks must return and must not leave with `longjmp`. Link the static
library together with the system libraries it needs, on Linux
`-lpthread -ldl -lm`. CI compiles and runs this example, see
[openmodelica_homotopy_capi/tests/c](openmodelica_homotopy_capi/tests/c), which
also has a test of the rest of the interface. The glue to the OpenModelica C
runtime lives in OpenModelica and uses this interface.

## License

This package is part of the OpenModelica Run-Time System and is distributed, at
your choice, under the terms of one of the following licenses:

* [BSD 3-Clause License (BSD New)][bsd-3-clause]
* [GNU Affero General Public License (AGPL) version 3][agpl-3]
* [OSMC Public License (OSMC-PL) version 1.8][osmc-pl]

This is the same choice as for the run-time system of [OpenModelica][openmodelica],
see the [OpenModelica Run-time System Public License][osmc-runtime].

[agpl-3]: AGPL-License.txt
[bsd-3-clause]: BSD-License.txt
[c-source]: https://github.com/OpenModelica/OpenModelica/blob/18d4ff455d1beceb4709d770ca4faf53d65732da/OMCompiler/SimulationRuntime/c/simulation/solver/nonlinearSolverHomotopy.c
[ci-badge]: https://github.com/OpenModelica/OMNonlinearSolvers/actions/workflows/ci.yml/badge.svg
[ci]: https://github.com/OpenModelica/OMNonlinearSolvers/actions/workflows/ci.yml
[openmodelica]: https://openmodelica.org/
[osmc-pl]: OSMC-License.txt
[osmc-runtime]: OSMC-Runtime-License.txt
