use std;

type SimpleVector = [f32; 2];

type TwoByTwoMatrix = [SimpleVector; 2];

fn getDeterminant(matrix: &TwoByTwoMatrix) -> f32 {
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d = matrix[1][1];

    (a * d) - (b * c)
}

fn invert(matrix: &TwoByTwoMatrix) -> TwoByTwoMatrix {
    let determinant: f32 = getDeterminant(&matrix);

    if determinant == 0f32 {
        panic!("Cannot invert matrix with determinant of 0")
    }

    let negOne = -1f32;

    let negatedForm: TwoByTwoMatrix = [
        [matrix[1][1], negOne * matrix[0][1]],
        [negOne * matrix[1][0], matrix[0][0]],
    ];

    let inverted = multiply_matrix_by_scalar(&negatedForm, 1f32 / determinant);
    return inverted;
}

fn multiply_matrix_by_vector(matrix: &TwoByTwoMatrix, vec: &SimpleVector) -> SimpleVector {
    let mut outputVec: Vec<f32> = Vec::with_capacity(2);

    outputVec.push((matrix[0][0] * vec[0]) + (matrix[0][1] * vec[1]));
    outputVec.push((matrix[1][0] * vec[0]) + (matrix[1][1] * vec[1]));

    return [outputVec[0], outputVec[1]];
}

fn multiply_matrix_by_scalar(matrix: &TwoByTwoMatrix, scalar: f32) -> TwoByTwoMatrix {
    let mut outputVecOne: Vec<f32> = Vec::with_capacity(2);
    let mut outputVecTwo: Vec<f32> = Vec::with_capacity(2);

    outputVecOne.push(matrix[0][0] * scalar);
    outputVecOne.push(matrix[0][1] * scalar);
    outputVecTwo.push(matrix[1][0] * scalar);
    outputVecTwo.push(matrix[1][1] * scalar);

    return [
        [outputVecOne[0], outputVecOne[1]],
        [outputVecTwo[0], outputVecTwo[1]],
    ];
}

fn multiply_matrix_by_matrix(
    matrixOne: &TwoByTwoMatrix,
    matrixTwo: &TwoByTwoMatrix,
) -> TwoByTwoMatrix {
    let a = (matrixOne[0][0] * matrixTwo[0][0]) + (matrixOne[0][1] * matrixTwo[1][0]);
    let b = (matrixOne[0][0] * matrixTwo[0][1]) + (matrixOne[0][1] * matrixTwo[1][1]);
    let c = (matrixOne[1][0] * matrixTwo[0][0]) + (matrixOne[1][1] * matrixTwo[1][0]);
    let d = (matrixOne[1][0] * matrixTwo[0][1]) + (matrixOne[1][1] * matrixTwo[1][1]);

    return [[a, b], [c, d]];
}

fn main() {
    println!("Hello, world!");
}
