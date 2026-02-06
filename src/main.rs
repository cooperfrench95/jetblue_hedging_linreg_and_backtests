use csv;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct CSVEntry {
    price: f32,
}

type SimpleVector = [f32; 2];

type TwoByTwoMatrix = [SimpleVector; 2];

type ArbitrarySizeMatrix = Vec<Vec<f32>>;

type PriceDataMatrix = Vec<SimpleVector>;

fn getDeterminant(matrix: &TwoByTwoMatrix) -> f32 {
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d = matrix[1][1];

    (a * d) - (b * c)
}

fn dot_product(row: &Vec<f32>, col: &Vec<f32>) -> f32 {
    if row.iter().len() != col.iter().len() {
        panic!("Row and cols must be same length to calculate dot product")
    }

    let mut sum: f32 = 0f32;
    for (rowIdx, rowValue) in row.iter().enumerate() {
        for (colIdx, colValue) in col.iter().enumerate() {
            if colIdx == rowIdx {
                sum += rowValue * colValue;
                break;
            }
        }
    }

    return sum;
}

// Price data vector will be like
// [
//  [1, 12],
//  [1, 13],
//  [1, 14]
// ]

// Transposed
// [
//   [1, 1, 1],
//   [12, 13, 14]
// ]

fn safe_get_float(firstIndex: usize, secondIndex: usize, vm: &ArbitrarySizeMatrix) -> f32 {
    let val = vm
        .get(firstIndex)
        .expect("Missing A position in gram matrix")
        .get(secondIndex)
        .expect("Missing A position in gram matrix");
    return *val;
}

fn to2By2Matrix(input: &ArbitrarySizeMatrix) -> TwoByTwoMatrix {
    return [
        [safe_get_float(0, 0, &input), safe_get_float(0, 1, &input)],
        [safe_get_float(1, 0, &input), safe_get_float(1, 1, &input)],
    ];
}

fn from2By2Matrix(input: &TwoByTwoMatrix) -> ArbitrarySizeMatrix {
    let mut internalVecOne: Vec<f32> = Vec::with_capacity(2);
    let mut internalVecTwo: Vec<f32> = Vec::with_capacity(2);

    internalVecOne.push(input[0][0]);
    internalVecOne.push(input[0][1]);
    internalVecOne.push(input[1][0]);
    internalVecOne.push(input[1][1]);

    let mut outputVec: Vec<Vec<f32>> = Vec::with_capacity(2);
    outputVec.push(internalVecOne);
    outputVec.push(internalVecTwo);

    return outputVec;
}

fn transpose(matrix: &PriceDataMatrix) -> ArbitrarySizeMatrix {
    let mut internalVecOne: Vec<f32> = Vec::with_capacity(matrix.len());
    let mut internalVecTwo: Vec<f32> = Vec::with_capacity(matrix.len());
    let mut transposed = Vec::with_capacity(2);

    for vector in matrix.iter() {
        internalVecOne.push(vector[0]);
        internalVecTwo.push(vector[1]);
    }

    transposed.push(internalVecOne);
    transposed.push(internalVecTwo);

    return transposed;
}

fn get_gram_matrix(matrix: &PriceDataMatrix) -> TwoByTwoMatrix {
    let mut transposed = transpose(matrix);
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

fn multiply_matrix_by_vector(matrix: &ArbitrarySizeMatrix, vec: &Vec<f32>) -> Vec<f32> {
    let mut outputVec: Vec<f32> = Vec::with_capacity(matrix.len());

    for rowNum in 0..matrix.len() {
        outputVec.push(dot_product(&matrix[rowNum], vec))
    }

    return outputVec;
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
    matrixOne: &Vec<Vec<f32>>,
    matrixTwo: &Vec<Vec<f32>>,
) -> ArbitrarySizeMatrix {
    let outputLength = matrixOne.len();
    let rowLength = matrixTwo[0].len();
    let columnLength = matrixTwo.len();
    if rowLength != outputLength {
        panic!("Cannot multiply matrices with incompatible sizes")
    }

    // Construct output vec data structure first
    let mut outputMatrix: Vec<Vec<f32>> = Vec::with_capacity(outputLength);
    for rowNum in 0..outputLength {
        let rowVector: Vec<f32> = Vec::with_capacity(columnLength);
        outputMatrix.push(rowVector);
    }

    for rowNum in 0..outputLength {
        for colNum in 0..columnLength {
            // Construct the column vector so we can calculate the dot product for the row and col
            let mut columnVec: Vec<f32> = Vec::with_capacity(columnLength);
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

fn get_price_data_matrix(fileName: &str) -> PriceDataMatrix {
    let mut reader = match csv::Reader::from_path(fileName) {
        Ok(f) => f,
        Err(e) => panic!("File could not be read"),
    };

    let mut outputMatrix: PriceDataMatrix = Vec::new();

    for result in reader.deserialize() {
        let record: CSVEntry = result.expect("Malformed CSV data");
        outputMatrix.push([1f32, record.price]);
    }

    return outputMatrix;
}

fn get_price_data_vector(fileName: &str) -> Vec<f32> {
    let mut reader = match csv::Reader::from_path(fileName) {
        Ok(f) => f,
        Err(e) => panic!("File could not be read"),
    };

    let mut outputVec: Vec<f32> = Vec::new();

    for result in reader.deserialize() {
        let record: CSVEntry = result.expect("Malformed CSV data");
        outputVec.push(record.price);
    }

    return outputVec;
}

fn main() {
    let priceData = get_price_data_matrix("brent_prices.csv");
    let baselineData = get_price_data_vector("jet_fuel_prices.csv");

    let gramMatrix = get_gram_matrix(&priceData);
    let priceDataTranspose = transpose(&priceData);
    let baselineToPriceCovariance = multiply_matrix_by_vector(&priceDataTranspose, &baselineData);
    let invertedGramMatrix = invert(&gramMatrix);
    let coefficients = multiply_matrix_by_vector(
        &from2By2Matrix(&invertedGramMatrix),
        &baselineToPriceCovariance,
    );

    let alpha = match coefficients.get(0) {
        Some(f) => f,
        None => panic!("Missing alpha coefficient"),
    };
    let beta = match coefficients.get(0) {
        Some(f) => f,
        None => panic!("Missing beta coefficient"),
    };
}
