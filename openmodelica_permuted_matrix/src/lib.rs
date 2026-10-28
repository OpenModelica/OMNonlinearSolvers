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

//! A view of a column-major matrix with permuted rows and columns.
//!
//! [`PermutedMatrix`] swaps rows and columns by swapping entries of two index
//! permutations, the data of the matrix never moves. The matrix is a slice of
//! its values, column after column, and the permutations are a
//! [`Permutations`] buffer of the caller, so a view doesn't allocate. Elements
//! are accessed by their logical (permuted) position with `m[(i, j)]`, which
//! reads or writes the physical position
//! `(row_permutation[i], column_permutation[j])`.
//! [`PermutedMatrix::to_mat`] creates a new matrix with the permutations
//! applied, [`PermutedMatrix::row`], [`PermutedMatrix::row_iter`] and
//! [`PermutedMatrix::diagonal`] iterate over elements in logical order, and
//! the `Display` implementation prints the logical matrix.
//!
//! This is useful for algorithms like Gaussian elimination with complete
//! pivoting, which swap rows and columns in every step.
//!
//! # Example
//!
//! ```
//! use faer::mat;
//! use openmodelica_permuted_matrix::{PermutedMatrix, Permutations};
//!
//! // [[1, 2, 3], [4, 5, 6]], column after column
//! let mut m = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
//! let mut permutations = Permutations::new(2, 3);
//! let mut p = PermutedMatrix::new(&mut m, &mut permutations);
//!
//! p.swap_rows(0, 1);
//! p.swap_columns(0, 2);
//! assert_eq!(p[(0, 0)], 6.0);
//! assert_eq!(p.to_mat(), mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
//!
//! // Writes go to the physical position
//! p[(0, 0)] = 10.0;
//! assert_eq!(m, [1.0, 4.0, 2.0, 5.0, 3.0, 10.0]);
//! ```

use std::fmt::{self};
use std::iter::Iterator;

use faer::Mat;
use std::ops::{Add, Index, IndexMut, Mul, Range};

/// The row and column permutations of a [`PermutedMatrix`], allocated once
/// for matrices of one shape.
#[derive(Debug)]
pub struct Permutations {
    /// Physical row of each logical row
    rows: Box<[usize]>,
    /// Physical column of each logical column
    columns: Box<[usize]>,
}

impl Permutations {
    /// Allocates the permutations of a matrix with `nrows` rows and `ncols`
    /// columns.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Permutations {
            rows: (0..nrows).collect(),
            columns: (0..ncols).collect(),
        }
    }

    /// Sets both permutations to the identity.
    fn reset(&mut self) {
        for (i, row) in self.rows.iter_mut().enumerate() {
            *row = i;
        }
        for (j, column) in self.columns.iter_mut().enumerate() {
            *column = j;
        }
    }
}

/// View of a borrowed column-major matrix with permuted rows and columns.
///
/// All positions in the methods of this type are logical positions, see the
/// [crate documentation](crate). The `Debug` output shows the internal state,
/// the physical values and both permutations, while `Display` shows the
/// logical matrix.
#[derive(Debug)]
pub struct PermutedMatrix<'a, T> {
    /// Borrowed values, column after column, never moved by the view
    values: &'a mut [T],
    /// Borrowed permutations, their lengths are the shape of the matrix
    permutations: &'a mut Permutations,
}

impl<'a, T> PermutedMatrix<'a, T> {
    /// Creates a view of the matrix `values`, column after column, with the
    /// shape of `permutations`, without any permutation: logical and physical
    /// positions are the same. `permutations` is reset and holds the
    /// permutations of the view.
    ///
    /// The values stay borrowed while the view exists. Writes through the
    /// view change `values`, the data is never copied.
    ///
    /// # Panics
    ///
    /// Panics if `values` doesn't have one entry per element of the shape of
    /// `permutations`.
    pub fn new(values: &'a mut [T], permutations: &'a mut Permutations) -> Self {
        assert_eq!(
            values.len(),
            permutations.rows.len() * permutations.columns.len(),
            "Permutations have the wrong shape"
        );
        permutations.reset();
        PermutedMatrix {
            values,
            permutations,
        }
    }

    /// Returns the number of rows of the matrix.
    pub fn nrows(&self) -> usize {
        self.permutations.rows.len()
    }

    /// Returns the number of columns of the matrix.
    pub fn ncols(&self) -> usize {
        self.permutations.columns.len()
    }

    /// Returns the row permutation.
    ///
    /// Entry `i` is the physical row of logical row `i`.
    pub fn row_permutation(&self) -> &[usize] {
        &self.permutations.rows
    }

    /// Returns the column permutation.
    ///
    /// Entry `j` is the physical column of logical column `j`. Use it to map
    /// results computed in logical order back to the original columns, for
    /// example the entries of a solution vector.
    pub fn column_permutation(&self) -> &[usize] {
        &self.permutations.columns
    }

    /// Returns an iterator over the logical diagonal, the elements at
    /// `(k, k)` for `k` in `0..min(nrows, ncols)`.
    ///
    /// This is not the physical diagonal of the borrowed matrix, unless no
    /// rows or columns were swapped.
    pub fn diagonal(&self) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator<Item = &T> {
        let ndiag = self
            .permutations
            .columns
            .len()
            .min(self.permutations.rows.len());
        (0..ndiag).map(move |i| &self[(i, i)])
    }

    /// Returns an iterator over the elements of logical row `i`, from the
    /// first to the last logical column.
    ///
    /// # Panics
    ///
    /// Panics while iterating if `i` is not smaller than [`Self::nrows`].
    pub fn row(
        &self,
        i: usize,
    ) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator<Item = &T> {
        self.row_from(i, 0..self.ncols())
    }

    /// Returns an iterator over all logical rows, each one an iterator over
    /// its elements as returned by [`Self::row`].
    pub fn row_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = impl DoubleEndedIterator<Item = &T>>
    + ExactSizeIterator<Item = impl ExactSizeIterator<Item = &T>> {
        (0..self.permutations.rows.len()).map(move |i| self.row(i))
    }

    /// Returns a new matrix with the permutations applied, its element
    /// `(i, j)` is the logical element `(i, j)` of the view.
    ///
    /// The borrowed matrix is not changed.
    pub fn to_mat(&self) -> Mat<T>
    where
        T: Clone,
    {
        Mat::from_fn(
            self.permutations.rows.len(),
            self.permutations.columns.len(),
            |i, j| self[(i, j)].clone(),
        )
    }

    /// Returns an iterator over the elements of logical row `i` in the
    /// logical `columns`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not smaller than [`Self::nrows`] or `columns` isn't
    /// a range of columns.
    pub fn row_from(
        &self,
        i: usize,
        columns: Range<usize>,
    ) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator<Item = &T> {
        let (row, nrows) = (self.permutations.rows[i], self.nrows());
        self.permutations.columns[columns]
            .iter()
            .map(move |column| &self.values[row + column * nrows])
    }

    /// Adds `factor` times logical row `source` to logical row `target`, in
    /// the logical `columns`: `m[(target, j)] = m[(target, j)] + factor·m[(source, j)]`.
    ///
    /// # Panics
    ///
    /// Panics if `target` or `source` is not smaller than [`Self::nrows`] or
    /// `columns` isn't a range of columns.
    pub fn add_scaled_row(&mut self, target: usize, source: usize, factor: T, columns: Range<usize>)
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        let nrows = self.nrows();
        let (target, source) = (
            self.permutations.rows[target],
            self.permutations.rows[source],
        );
        for column in &self.permutations.columns[columns] {
            let offset = column * nrows;
            self.values[target + offset] =
                self.values[target + offset] + factor * self.values[source + offset];
        }
    }

    /// Swaps logical rows `a` and `b`.
    ///
    /// Only the row permutation changes, the data of the matrix doesn't move.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` is not smaller than [`Self::nrows`].
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        self.permutations.rows.swap(a, b);
    }

    /// Swaps logical columns `a` and `b`.
    ///
    /// Only the column permutation changes, the data of the matrix doesn't
    /// move.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` is not smaller than [`Self::ncols`].
    pub fn swap_columns(&mut self, a: usize, b: usize) {
        self.permutations.columns.swap(a, b);
    }

    /// Write the logical (permuted) matrix to `f`, formatting each element
    /// with `fmt_elem`.
    ///
    /// Each row is written on its own line as `[a, b, c]`, without a trailing
    /// newline after the last row. Used by the `Display` implementation, which
    /// passes `fmt::Display::fmt` as `fmt_elem`.
    ///
    /// `fmt_elem` receives the same formatter `f`, so formatting options of
    /// the caller, like width and precision in `{:8.3}`, apply to every
    /// element.
    fn fmt_with(
        &self,
        f: &mut fmt::Formatter<'_>,
        fmt_elem: impl Fn(&T, &mut fmt::Formatter<'_>) -> fmt::Result,
    ) -> fmt::Result {
        for (i, row) in self.row_iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for (j, val) in row.enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                fmt_elem(val, f)?;
            }
            write!(f, "]")?;
        }

        Ok(())
    }
}

/// Read access to the logical element `(i, j)`.
///
/// # Panics
///
/// Panics if `i` or `j` is out of bounds.
impl<'a, T> Index<(usize, usize)> for PermutedMatrix<'a, T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.values[self.permutations.rows[i] + self.permutations.columns[j] * self.nrows()]
    }
}

/// Write access to the logical element `(i, j)`, the write goes to the
/// physical element `(row_permutation[i], column_permutation[j])`.
///
/// # Panics
///
/// Panics if `i` or `j` is out of bounds.
impl<'a, T> IndexMut<(usize, usize)> for PermutedMatrix<'a, T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        let nrows = self.nrows();
        &mut self.values[self.permutations.rows[i] + self.permutations.columns[j] * nrows]
    }
}

/// Prints the logical matrix, one row per line, like
///
/// ```text
/// [6, 5, 4]
/// [3, 2, 1]
/// ```
///
/// Formatting options apply to every element, `{:.2}` prints all elements
/// with two decimals.
impl<'a, T: fmt::Display> fmt::Display for PermutedMatrix<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with(f, fmt::Display::fmt)
    }
}

#[cfg(test)]
mod unit_tests;
