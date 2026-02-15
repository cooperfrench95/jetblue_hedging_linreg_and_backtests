use crate::linear_algebra::*;

#[cfg(test)]
use approx::assert_relative_eq;

fn assert_vecf64_eq(vecOne: &Vec<f64>, vecTwo: &Vec<f64>) {
    assert_relative_eq!(vecOne.as_slice(), vecTwo.as_slice(), epsilon = f64::EPSILON)
}

fn assert_matrix_f64_eq(matrixOne: &Vec<Vec<f64>>, matrixTwo: &Vec<Vec<f64>>) {
    for (index, entry) in matrixOne.iter().enumerate() {
        let correspondingEntry = matrixTwo.get(index);
        assert_vecf64_eq(&entry, &correspondingEntry.unwrap());
    }
}

#[test]
fn multiply_matrices_by_matrices() {
    // 2x2
    let matrixA: Vec<Vec<f64>> = vec![vec![1f64, 2f64], vec![-1f64, 3f64]];
    let matrixB: Vec<Vec<f64>> = vec![vec![2f64, 1f64], vec![2f64, -2f64]];
    let expectedOutputMatrix = vec![vec![6f64, -3f64], vec![4f64, -7f64]];
    let result = multiply_matrix_by_matrix(&matrixA, &matrixB);
    assert_matrix_f64_eq(&expectedOutputMatrix, &result);

    // 3x3
    let matrixA: Vec<Vec<f64>> = vec![
        vec![1f64, 2f64, -1f64],
        vec![-1f64, 3f64, 0f64],
        vec![2f64, 0f64, 2f64],
    ];
    let matrixB: Vec<Vec<f64>> = vec![
        vec![2f64, 1f64, 4f64],
        vec![2f64, -2f64, 0f64],
        vec![0f64, 1f64, 0f64],
    ];
    let expectedOutputMatrix = vec![
        vec![6f64, -4f64, 4f64],
        vec![4f64, -7f64, -4f64],
        vec![4f64, 4f64, 8f64],
    ];
    let result = multiply_matrix_by_matrix(&matrixA, &matrixB);
    assert_matrix_f64_eq(&expectedOutputMatrix, &result);

    // 3x2
    let matrixA: Vec<Vec<f64>> = vec![vec![1f64, 2f64, -1f64], vec![-1f64, 3f64, 0f64]];
    let matrixB: Vec<Vec<f64>> = vec![vec![2f64, 1f64], vec![2f64, -2f64], vec![0f64, 1f64]];
    let expectedOutputMatrix = vec![vec![6f64, -4f64], vec![4f64, -7f64]];
    let result = multiply_matrix_by_matrix(&matrixA, &matrixB);
    assert_matrix_f64_eq(&expectedOutputMatrix, &result);
}

#[test]
fn gram_matrix() {
    let matrix: Vec<[f64; 2]> = vec![[1f64, 2f64], [-1f64, 3f64]];
    let expectedOutput = [[2f64, -1f64], [-1f64, 13f64]];

    let output = get_gram_matrix(&matrix);

    assert_eq!(expectedOutput, output);
}

#[test]
fn transpositon() {
    let matrix: Vec<[f64; 2]> = vec![[1f64, 2f64], [-1f64, 3f64]];
    let expectedOutput = vec![vec![1f64, -1f64], vec![2f64, 3f64]];

    let output = transpose(&matrix);

    assert_matrix_f64_eq(&expectedOutput, &output);
}

#[test]
fn determinant() {
    let matrix = [[1f64, 2f64], [-1f64, 3f64]];
    let expectedOutput = 5f64;

    let output = getDeterminant(&matrix);

    assert_relative_eq!(expectedOutput, output);
}

#[test]
fn inversion() {
    let matrix = [[1f64, 2f64], [-1f64, 3f64]]; // Has a determinant of 5
    let expectedOutput = [[0.6f64, -0.4f64], [0.2f64, 0.2f64]];

    let output = invert(&matrix);

    assert_relative_eq!(expectedOutput[0][0], output[0][0]);
    assert_relative_eq!(expectedOutput[0][1], output[0][1]);
    assert_relative_eq!(expectedOutput[1][0], output[1][0]);
    assert_relative_eq!(expectedOutput[1][1], output[1][1]);
}

#[test]
fn multiply_matrices_by_scalar() {
    let matrix = [[1f64, 2f64], [-1f64, 3f64]];
    let expectedOutput = [[2f64, 4f64], [-2f64, 6f64]];

    let output = multiply_matrix_by_scalar(&matrix, 2f64);

    assert_relative_eq!(expectedOutput[0][0], output[0][0]);
    assert_relative_eq!(expectedOutput[0][1], output[0][1]);
    assert_relative_eq!(expectedOutput[1][0], output[1][0]);
    assert_relative_eq!(expectedOutput[1][1], output[1][1]);
}

#[test]
fn multiply_matrices_by_vector() {
    let matrix = vec![vec![1f64, 2f64], vec![-1f64, 3f64]];
    let vec: Vec<f64> = vec![2f64, 2f64];
    let expectedOutput = vec![6f64, 4f64];

    let output = multiply_matrix_by_vector(&matrix, &vec);

    assert_vecf64_eq(&expectedOutput, &output);
}

#[test]
fn test_dotproduct() {
    let row = vec![1f64, 2f64, 3f64];
    let col = vec![-2f64, 7f64, 4f64];

    let output = dot_product(&row, &col);
    assert_relative_eq!(24f64, output);
}
