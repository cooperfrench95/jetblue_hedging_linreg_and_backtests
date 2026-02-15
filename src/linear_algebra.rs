use super::*;

pub fn getDeterminant(matrix: &TwoByTwoMatrix) -> f64 {
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d: f64 = matrix[1][1];

    (a * d) - (b * c)
}

pub fn dot_product(row: &Vec<f64>, col: &Vec<f64>) -> f64 {
    if row.len() != col.len() {
        panic!("Dot product can only be found with equal length vectors")
    }
    // "zip" each element together, times each one by its pair, then sum the results
    return row.iter().zip(col).map(|(a, b)| a * b).sum();
}

pub fn transpose(matrix: &PriceDataMatrix) -> ArbitrarySizeMatrix {
    let mut internalVecOne: Vec<f64> = Vec::with_capacity(matrix.len());
    let mut internalVecTwo: Vec<f64> = Vec::with_capacity(matrix.len());
    let mut transposed = Vec::with_capacity(2);

    for vector in matrix.iter() {
        internalVecOne.push(vector[0]);
        internalVecTwo.push(vector[1]);
    }

    transposed.push(internalVecOne);
    transposed.push(internalVecTwo);

    return transposed;
}

pub fn get_gram_matrix(matrix: &PriceDataMatrix) -> TwoByTwoMatrix {
    let transposed = transpose(matrix);
    let mut originalMatrixAsVec = Vec::with_capacity(matrix.len());

    for vector in matrix.iter() {
        let mut reconstructedVec = Vec::new();
        reconstructedVec.push(vector[0]);
        reconstructedVec.push(vector[1]);
        originalMatrixAsVec.push(reconstructedVec);
    }

    let asVectorMatrix = multiply_matrix_by_matrix(&transposed, &originalMatrixAsVec);

    return to2By2Matrix(&asVectorMatrix);
}

pub fn invert(matrix: &TwoByTwoMatrix) -> TwoByTwoMatrix {
    let determinant: f64 = getDeterminant(&matrix);

    if determinant == 0f64 {
        panic!("Cannot invert matrix with determinant of 0")
    }

    let negOne = -1f64;

    let negatedForm: TwoByTwoMatrix = [
        [matrix[1][1], negOne * matrix[0][1]],
        [negOne * matrix[1][0], matrix[0][0]],
    ];

    let inverted = multiply_matrix_by_scalar(&negatedForm, 1f64 / determinant);
    return inverted;
}

pub fn multiply_matrix_by_vector(matrix: &ArbitrarySizeMatrix, vec: &Vec<f64>) -> Vec<f64> {
    let mut outputVec: Vec<f64> = Vec::with_capacity(matrix.len());

    for rowNum in 0..matrix.len() {
        outputVec.push(dot_product(&matrix[rowNum], vec))
    }

    return outputVec;
}

pub fn multiply_matrix_by_scalar(matrix: &TwoByTwoMatrix, scalar: f64) -> TwoByTwoMatrix {
    return [
        [matrix[0][0] * scalar, matrix[0][1] * scalar],
        [matrix[1][0] * scalar, matrix[1][1] * scalar],
    ];
}

pub fn multiply_matrix_by_matrix(
    matrixOne: &Vec<Vec<f64>>,
    matrixTwo: &Vec<Vec<f64>>,
) -> ArbitrarySizeMatrix {
    let outputLength = matrixOne.len();
    let columnLength = matrixTwo[0].len();
    if columnLength != outputLength {
        panic!("Cannot multiply matrices with incompatible sizes")
    }

    // Construct output vec data structure first
    let mut outputMatrix: Vec<Vec<f64>> = Vec::with_capacity(outputLength);
    for _rowNum in 0..outputLength {
        let rowVector: Vec<f64> = Vec::with_capacity(columnLength);
        outputMatrix.push(rowVector);
    }

    for rowNum in 0..outputLength {
        for colNum in 0..columnLength {
            // Construct the column vector so we can calculate the dot product for the row and col
            let mut columnVec: Vec<f64> = Vec::with_capacity(columnLength);
            for entry in matrixTwo.iter() {
                columnVec.push(entry[colNum])
            }
            let dp = dot_product(&matrixOne[rowNum], &columnVec);

            outputMatrix
                .get_mut(rowNum)
                .expect("Unexpected output vector length")
                .push(dp);
        }
    }

    return outputMatrix;
}
