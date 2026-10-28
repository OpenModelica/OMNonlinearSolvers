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

/* Smoke test of the C interface: one function per way a C host uses it. */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "openmodelica_homotopy_capi.h"

static int failures = 0;

#define CHECK(condition)                                                    \
  do {                                                                      \
    if (!(condition)) {                                                     \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,      \
              #condition);                                                  \
      failures++;                                                           \
    }                                                                       \
  } while (0)

/* F(x) = x³ − 2x + 2, Newton's method alone gets stuck from x = 0 */
static int cubic(void *user, const double *x, double *f) {
  (void)user;
  f[0] = x[0] * x[0] * x[0] - 2.0 * x[0] + 2.0;
  return HOM_EVAL_OK;
}

/* J(x) = 3x² − 2, counting its calls in user */
static int cubic_jacobian(void *user, const double *x, const double *f,
                          double *jacobian, size_t leading_dimension) {
  (void)leading_dimension;
  (void)f;
  (*(int *)user)++;
  jacobian[0] = 3.0 * x[0] * x[0] - 2.0;
  return HOM_EVAL_OK;
}

/* A model that can't be evaluated anywhere */
static int asserts(void *user, const double *x, double *f) {
  (void)user;
  (void)x;
  (void)f;
  return HOM_EVAL_ASSERT;
}

/* The model's own homotopy H(x, λ) = x − 2λ, with λ as last unknown */
static int own_homotopy(void *user, const double *y, double *h) {
  (void)user;
  h[0] = y[0] - 2.0 * y[1];
  return HOM_EVAL_OK;
}

/* The root of the cubic */
static const double ROOT = -1.769292354238631;

static void solve_with_homotopy(void) {
  HomSolver *solver = hom_solver_new(1, NULL);
  HomModel model = {0};
  model.num_residuals = model.num_unknowns = 1;
  model.residual = cubic;
  double x[1] = {0.0};
  HomSolvedBy solved_by;

  CHECK(hom_solve(solver, &model, x, &solved_by) == HOM_STATUS_SOLVED);
  CHECK(solved_by == HOM_SOLVED_BY_HOMOTOPY_NEWTON_NEGATIVE);
  CHECK(fabs(x[0] - ROOT) < 1e-10);
  hom_solver_free(solver);
}

static void solve_with_jacobian(void) {
  HomSolver *solver = hom_solver_new(1, NULL);
  int calls = 0;
  HomModel model = {0};
  model.user = &calls;
  model.num_residuals = model.num_unknowns = 1;
  model.residual = cubic;
  model.jacobian = cubic_jacobian;
  double x[1] = {0.0};

  CHECK(hom_solve(solver, &model, x, NULL) == HOM_STATUS_SOLVED);
  CHECK(fabs(x[0] - ROOT) < 1e-10);
  CHECK(calls > 0);
  hom_solver_free(solver);
}

static void model_error(void) {
  HomSolver *solver = hom_solver_new(1, NULL);
  HomModel model = {0};
  model.num_residuals = model.num_unknowns = 1;
  model.residual = asserts;
  double x[1] = {1.0};

  CHECK(hom_solve(solver, &model, x, NULL) == HOM_STATUS_MODEL_ERROR);
  CHECK(x[0] == 1.0);
  hom_solver_free(solver);
}

static void solve_own_homotopy(void) {
  HomSolver *solver = hom_solver_new(1, NULL);
  HomModel model = {0};
  model.num_residuals = 1;
  model.num_unknowns = 2;
  model.residual = own_homotopy;
  double x[1] = {0.0};
  HomSolvedBy solved_by;

  CHECK(hom_solve_own_homotopy(solver, &model, x, false, &solved_by) ==
        HOM_STATUS_SOLVED);
  CHECK(solved_by == HOM_SOLVED_BY_OWN_HOMOTOPY_POSITIVE);
  CHECK(fabs(x[0] - 2.0) < 1e-5);
  hom_solver_free(solver);
}

/* The events of the paths, see observe_paths */
typedef struct Events {
  int paths;
  int accepted;
  size_t finishedSteps;
  HomPathFailure failure;
  double failureValue;
} Events;

static void path_started(void *user, int direction, const double *y, size_t size) {
  (void)direction;
  (void)y;
  (void)size;
  ((Events *)user)->paths++;
}

static void step_accepted(void *user, const double *y, size_t size) {
  (void)y;
  (void)size;
  ((Events *)user)->accepted++;
}

static void path_finished(void *user, const double *y, size_t size, size_t steps) {
  (void)y;
  (void)size;
  ((Events *)user)->finishedSteps = steps;
}

static void path_failed(void *user, HomPathFailure failure, double value) {
  ((Events *)user)->failure = failure;
  ((Events *)user)->failureValue = value;
}

static void observe_paths(void) {
  HomSolver *solver = hom_solver_new(1, NULL);
  Events events = {0};
  HomObserver observer = {0};
  observer.user = &events;
  observer.path_started = path_started;
  observer.step_accepted = step_accepted; /* step_started stays NULL */
  observer.path_finished = path_finished;
  observer.path_failed = path_failed;
  HomModel model = {0};
  model.num_residuals = model.num_unknowns = 1;
  model.residual = cubic;
  model.observer = &observer;
  double x[1] = {0.0};

  CHECK(hom_solve(solver, &model, x, NULL) == HOM_STATUS_SOLVED);
  CHECK(events.paths == 2); /* the first falls below λ = −1 */
  CHECK(events.failure == HOM_PATH_FAILURE_LAMBDA_BELOW_MINUS_ONE);
  CHECK(events.failureValue < -1.0);
  CHECK(events.finishedSteps > 0 && events.accepted >= (int)events.finishedSteps);

  HomStatistics statistics = hom_solver_statistics(solver);
  CHECK(statistics.path_steps == events.finishedSteps);
  CHECK(statistics.newton_iterations > 0);
  hom_solver_free(solver);
}

static int newtonMessages = 0;

/* Counts the messages of Newton's method */
static void logger(HomLogLevel level, const char *target, const char *message) {
  (void)level;
  if (strcmp(target, "NLS") == 0 && strncmp(message, "Newton iteration", 16) == 0) {
    newtonMessages++;
  }
}

static void log_messages(void) {
  HomSolver *solver = hom_solver_new(1, NULL);
  HomModel model = {0};
  model.num_residuals = model.num_unknowns = 1;
  model.residual = cubic;
  double x[1] = {1.0};

  hom_set_logger(logger, 5); /* up to trace */
  CHECK(hom_solve(solver, &model, x, NULL) == HOM_STATUS_SOLVED);
  hom_set_logger(NULL, 0);
  CHECK(newtonMessages > 0);
  hom_solver_free(solver);
}

/* Solves the cubic in place of hybrd, counting its calls in user */
static int fallback(void *user, double *x) {
  (*(int *)user)++;
  x[0] = ROOT;
  return 1;
}

static void solve_with_fallback(void) {
  HomSolver *solver = hom_solver_new(1, NULL);
  int calls = 0;
  HomModel model = {0};
  model.user = &calls;
  model.num_residuals = model.num_unknowns = 1;
  model.residual = cubic;
  model.fallback = fallback;
  double x[1] = {0.0};
  HomSolvedBy solved_by;

  CHECK(hom_solve(solver, &model, x, &solved_by) == HOM_STATUS_SOLVED);
  CHECK(solved_by == HOM_SOLVED_BY_FALLBACK);
  CHECK(calls == 1);
  CHECK(x[0] == ROOT);
  hom_solver_free(solver);
}

/* Solves the 1×1 linear system, counting its calls in user */
static int linear_solver(void *user, size_t n, double *matrix, size_t leading_dimension,
                         double *rhs, double *determinant) {
  (void)n;
  (void)leading_dimension;
  (*(int *)user)++;
  rhs[0] /= matrix[0];
  *determinant = matrix[0];
  return 0;
}

static void solve_with_linear_solver(void) {
  HomSolver *solver = hom_solver_new(1, NULL);
  int calls = 0;
  HomModel model = {0};
  model.user = &calls;
  model.num_residuals = model.num_unknowns = 1;
  model.residual = cubic;
  model.linear_solver = linear_solver;
  double x[1] = {-1.5};

  CHECK(hom_solve(solver, &model, x, NULL) == HOM_STATUS_SOLVED);
  CHECK(calls > 0);
  CHECK(fabs(x[0] - ROOT) < 1e-10);
  hom_solver_free(solver);
}

static void settings(void) {
  HomSettings settings = {0};
  settings.tau_start = 0.1;
  settings.max_tries = 5;
  HomSolver *solver = hom_solver_new(1, &settings);
  CHECK(solver != NULL);
  hom_solver_free(solver);

  settings.tau_decreasing_factor = 0.5; /* must be greater than 1 */
  CHECK(hom_solver_new(1, &settings) == NULL);
  CHECK(hom_solver_new(0, NULL) == NULL);
}

static void invalid_arguments(void) {
  HomSolver *solver = hom_solver_new(1, NULL);
  HomModel model = {0};
  model.num_residuals = model.num_unknowns = 1;
  double x[1] = {0.0};

  CHECK(hom_solve(solver, &model, x, NULL) == HOM_STATUS_INVALID_ARGUMENT);
  model.residual = cubic;
  CHECK(hom_solve(NULL, &model, x, NULL) == HOM_STATUS_INVALID_ARGUMENT);
  CHECK(hom_solve(solver, NULL, x, NULL) == HOM_STATUS_INVALID_ARGUMENT);
  CHECK(hom_solve(solver, &model, NULL, NULL) == HOM_STATUS_INVALID_ARGUMENT);
  hom_solver_free(solver);
  hom_solver_free(NULL);
}

int main(void) {
  solve_with_homotopy();
  solve_with_jacobian();
  model_error();
  solve_own_homotopy();
  solve_with_fallback();
  solve_with_linear_solver();
  observe_paths();
  log_messages();
  settings();
  invalid_arguments();

  if (failures > 0) {
    fprintf(stderr, "%d checks failed\n", failures);
    return 1;
  }
  printf("All checks passed\n");
  return 0;
}
