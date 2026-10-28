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

use super::*;
use faer::mat;

/// Permutations for the shape of `m`
fn permutations_of<T>(m: &Mat<T>) -> Permutations {
    Permutations::new(m.nrows(), m.ncols())
}

/// The values of `m`, column after column
fn values_of(m: &Mat<f64>) -> Vec<f64> {
    m.col_iter()
        .flat_map(|column| column.iter().copied().collect::<Vec<_>>())
        .collect()
}

/// The `nrows`-row matrix of `values`, column after column
fn mat_of(values: &[f64], nrows: usize) -> Mat<f64> {
    Mat::from_fn(nrows, values.len() / nrows, |i, j| values[i + j * nrows])
}

#[test]
fn identity_permutation_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    let m = m_permutation.to_mat();
    assert_eq!(m, mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
}

#[test]
fn swap_rows_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_rows(0, 1);
    let m = m_permutation.to_mat();
    assert_eq!(m, mat![[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]);
}

#[test]
fn swap_columns_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_columns(0, 1);
    let m = m_permutation.to_mat();
    assert_eq!(
        m,
        mat![[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]],
        "Permutation 0 <-> 1 failed!"
    );

    m_permutation.swap_columns(2, 1);
    let m = m_permutation.to_mat();
    assert_eq!(
        m,
        mat![[2.0, 3.0, 1.0], [5.0, 6.0, 4.0]],
        "Permutation 2 <-> 1 failed!"
    );
}

#[test]
/// Permutations reused for a new view start as the identity again
fn reused_permutations_test() {
    let m = mat![[1.0, 2.0], [3.0, 4.0]];
    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);
    m_permutation.swap_rows(0, 1);
    m_permutation.swap_columns(0, 1);
    let m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    assert_eq!(m_permutation.row_permutation(), &[0, 1]);
    assert_eq!(m_permutation.column_permutation(), &[0, 1]);
}

#[test]
/// `row_from` iterates over a range of logical columns of a logical row
fn row_from_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_rows(0, 1);
    m_permutation.swap_columns(0, 2);

    assert_eq!(
        m_permutation.row_from(0, 1..3).copied().collect::<Vec<_>>(),
        [5.0, 4.0]
    );
}

#[test]
/// `add_scaled_row` adds a multiple of a logical row to another one in a
/// range of logical columns, the others stay unchanged
fn add_scaled_row_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_columns(0, 2);
    m_permutation.add_scaled_row(1, 0, -2.0, 1..3);

    assert_eq!(
        m_permutation.to_mat(),
        mat![[3.0, 2.0, 1.0], [6.0, 1.0, 2.0]]
    );
}

#[test]
#[should_panic = "Permutations have the wrong shape"]
fn wrong_shape_permutations_test() {
    let mut values = values_of(&mat![[1.0, 2.0], [3.0, 4.0]]);
    let mut permutations = Permutations::new(2, 3);

    let _ = PermutedMatrix::new(&mut values, &mut permutations);
}

#[test]
fn mat_outliving_perm_test() {
    let m = mat![[1.0, 2.0], [3.0, 4.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let _ = PermutedMatrix::new(&mut values, &mut permutations);

    assert_eq!(mat_of(&values, 2), m)
}

#[test]
fn new_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    assert_eq!(m_permutation.nrows(), 2);
    assert_eq!(m_permutation.ncols(), 3);
    assert_eq!(m_permutation.row_permutation(), &[0, 1]);
    assert_eq!(m_permutation.column_permutation(), &[0, 1, 2]);
}

#[test]
fn swap_rows_repeated_test() {
    let m = mat![[1.0], [2.0], [3.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    // Second swap moves a row that was already moved by the first one
    m_permutation.swap_rows(0, 1);
    m_permutation.swap_rows(2, 1);
    assert_eq!(m_permutation.row_permutation(), &[1, 2, 0]);
    assert_eq!(m_permutation.to_mat(), mat![[2.0], [3.0], [1.0]]);
}

#[test]
fn swap_same_index_test() {
    let m = mat![[1.0, 2.0], [3.0, 4.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_rows(1, 1);
    m_permutation.swap_columns(0, 0);
    assert_eq!(m_permutation.row_permutation(), &[0, 1]);
    assert_eq!(m_permutation.column_permutation(), &[0, 1]);
}

#[test]
fn swap_rows_and_columns_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_rows(0, 1);
    m_permutation.swap_columns(0, 2);
    assert_eq!(
        m_permutation.to_mat(),
        mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]
    );
}

#[test]
fn index_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_rows(0, 1);
    m_permutation.swap_columns(1, 2);

    // Logical (0, 0) is physical (1, 0), logical (1, 1) is physical (0, 2)
    assert_eq!(m_permutation[(0, 0)], 4.0);
    assert_eq!(m_permutation[(0, 1)], 6.0);
    assert_eq!(m_permutation[(1, 1)], 3.0);
    assert_eq!(m_permutation[(1, 2)], 2.0);
}

#[test]
fn index_mut_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_rows(0, 1);
    m_permutation.swap_columns(0, 2);
    m_permutation[(0, 0)] = 10.0;
    assert_eq!(m_permutation[(0, 0)], 10.0);

    // Writes go to the physical position, the memory layout is unchanged
    assert_eq!(mat_of(&values, 2), mat![[1.0, 2.0, 3.0], [4.0, 5.0, 10.0]]);
}

#[test]
fn diagonal_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);
    m_permutation.swap_columns(0, 2);

    let diag: Vec<f64> = m_permutation.diagonal().copied().collect();
    assert_eq!(diag, [3.0, 5.0]);
}

#[test]
fn row_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);
    m_permutation.swap_columns(0, 2);
    m_permutation.swap_rows(0, 1);

    let diag: Vec<f64> = m_permutation.row(1).copied().collect();
    assert_eq!(diag, [3.0, 2.0, 1.0]);
}

#[test]
fn fmt_display_test() {
    let m = mat![[1.5, 2.0], [1. / 3., f64::NAN], [-f64::INFINITY, 6.0]];
    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);
    m_permutation.swap_columns(1, 0);
    m_permutation.swap_rows(1, 2);

    let format = format!("{m_permutation:4.2}");

    assert_eq!(
        "[2.00, 1.50]\n\
         [6.00, -inf]\n\
         [ NaN, 0.33]",
        format
    );
}

#[test]
fn fmt_debug_display_test() {
    let m = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    let format = format!("{:?}", m_permutation);
    assert_eq!(
        "PermutedMatrix { values: [1.0, 3.0, 5.0, 2.0, 4.0, 6.0], \
        permutations: Permutations { rows: [0, 1, 2], columns: [0, 1] } }",
        format
    )
}

#[test]
fn to_mat_keeps_original_test() {
    let m = mat![[1.0, 2.0], [3.0, 4.0]];
    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_rows(0, 1);
    let mut copy = m_permutation.to_mat();
    copy[(0, 0)] = 10.0;
    assert_eq!(m_permutation[(0, 0)], 3.0);

    assert_eq!(mat_of(&values, 2), m);
}

#[test]
#[should_panic]
fn index_row_out_of_bounds_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    let _ = m_permutation[(2, 0)];
}

#[test]
#[should_panic]
fn index_column_out_of_bounds_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    let _ = m_permutation[(0, 3)];
}

#[test]
#[should_panic]
fn swap_rows_out_of_bounds_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_rows(0, 2);
}

#[test]
#[should_panic]
fn swap_columns_out_of_bounds_test() {
    let m = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut permutations = permutations_of(&m);
    let mut values = values_of(&m);
    let mut m_permutation = PermutedMatrix::new(&mut values, &mut permutations);

    m_permutation.swap_columns(0, 3);
}
