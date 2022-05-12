/**
 * Preliminary sketch of model (equations 2.2, 2.3) from:
 *
 * David A. Barmherzig and Ju Sun. 2022. Towards practical holographic
 * coherent diffraction imaging via maximum likelihood estimation.
 * arXiv 2105.11512v2.
 */
functions {
  // this won't be necessary once in the stan math library
  matrix abs(complex_matrix z) {
    matrix[rows(z), cols(z)] x;
    for (i in 1 : rows(z)) {
      for (j in 1 : cols(z)) {
        x[i][j] = abs(z[i][j]);
      }
    }
    return x;
  }

  /**
   * Return M1 x M2 matrix of 1 values with blocks in corners set to
   * 0, where the upper left is (r x r), the upper right is (r x r-1),
   * the lower left is (r-1 x r), and the lower right is (r-1 x r-1).
   * @param M1 number of rows
   * @param M2 number of cols
   * @param r block dimension
   * @return matrix of 1 values with 0-padded corners
   */
  matrix pad_corners(int M1, int M2, int r) {
    matrix[M1, M2] B_cal = rep_matrix(1, M1, M2);
    // upper left
    B_cal[1 : r, 1 : r] = rep_matrix(0, r, r);
    // upper right
    B_cal[1 : r, M2 - r + 2 : M2] = rep_matrix(0, r, r - 1);
    // lower left
    B_cal[M1 - r + 2 : M1, 1 : r] = rep_matrix(0, r - 1, r);
    // lower right
    B_cal[M1 - r + 2 : M1, M2 - r + 2 : M2] = rep_matrix(0, r - 1, r - 1);
    return B_cal;
  }
  /**
   * Return result of separating X and R with a matrix of 0s and then
   * 0 padding to right and below.  That is, assuming X and R are the
   * same shape and 0 is a matrix of zeros of the same shape, the
   * result is
   *
   *  [X, 0, R]  | 0
   *  --------------
   *      0      | 0
   *
   * @param X X matrix
   * @param R R matrix
   * @return 0-padded [X, 0, R] matrix
   */
  matrix pad(matrix X, matrix R, int M1, int M2) {
    matrix[M1, M2] y = rep_matrix(0, M1, M2);
    int N = rows(X);
    y[1 : N, 1 : N] = X;
    y[1 : N, 2 * N + 1 : 3 * N] = R;
    return y;
  }
}
data {
  int<lower=0> N; // image dimension
  matrix[N, N] R; // registration image
  int<lower=N> M1; // rows of padded matrices
  int<lower=3 * N> M2; // cols of padded matrices
  int<lower=1, upper=M1> r; // replaces omega1, omega2 in paper

  real<lower=0> N_p; // avg number of photons per pixel
  array[N, N] int<lower=0> Y_tilde; // observed number of photons
}
transformed data {
  matrix[M1, M2] B_cal = pad_corners(M1, M2, r);
}
parameters {
  matrix<lower=0>[N, N] X;
}
model {
  matrix[M1, M2] X0R_pad = pad(X, R, M1, M2);
  matrix[M1, M2] Y = B_cal .* abs(fft2(X0R_pad)) .^ 2;
  real Y_bar = mean(Y);

  // prior (look at Tikhonov or total variation regularization)
  // X ~ ???

  // likelihood
  real N_p_over_Y_bar = N_p / Y_bar;
  matrix[M1, M2] lambda = N_p_over_Y_bar * Y;
  for (m1 in 1 : M1) {
    for (m2 in 1 : M2) {
      // BMW: Y_tilde and lambda do not have the same dimensions.
      //      Is Y_tilde meant to be M1xM2?
      Y_tilde[m1, m2] ~ poisson(lambda[m1, m2]);
    }
  }
  // to_array_1d(Y_tilde) ~ poisson(to_vector(lambda));
}
