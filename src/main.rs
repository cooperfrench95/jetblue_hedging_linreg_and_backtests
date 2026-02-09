#![allow(non_snake_case)]

use csv;
use plotters::{
    self,
    chart::{ChartBuilder, SeriesLabelPosition},
    prelude::{BitMapBackend, Circle, IntoDrawingArea, PathElement},
    series::LineSeries,
    style::{self, Color},
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct CSVEntry {
    price: f64,
}

type SimpleVector = [f64; 2];

type TwoByTwoMatrix = [SimpleVector; 2];

type ArbitrarySizeMatrix = Vec<Vec<f64>>;

type PriceDataMatrix = Vec<SimpleVector>;

fn getDeterminant(matrix: &TwoByTwoMatrix) -> f64 {
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d: f64 = matrix[1][1];

    (a * d) - (b * c)
}

fn dot_product(row: &Vec<f64>, col: &Vec<f64>) -> f64 {
    // "zip" each element together, times each one by its pair, then sum the results
    return row.iter().zip(col).map(|(a, b)| a * b).sum();
}

fn safe_get_float(firstIndex: usize, secondIndex: usize, vm: &ArbitrarySizeMatrix) -> f64 {
    let val = vm
        .get(firstIndex)
        .expect("Missing A position in gram matrix")
        .get(secondIndex)
        .expect("Missing B position in gram matrix");
    return *val;
}

// Implication is it's already 2x2 and we are just shifting the type here
fn to2By2Matrix(input: &ArbitrarySizeMatrix) -> TwoByTwoMatrix {
    return [
        [safe_get_float(0, 0, &input), safe_get_float(0, 1, &input)],
        [safe_get_float(1, 0, &input), safe_get_float(1, 1, &input)],
    ];
}

fn from2By2Matrix(input: &TwoByTwoMatrix) -> ArbitrarySizeMatrix {
    let mut internalVecOne: Vec<f64> = Vec::with_capacity(2);
    let mut internalVecTwo: Vec<f64> = Vec::with_capacity(2);

    internalVecOne.push(input[0][0]);
    internalVecOne.push(input[0][1]);
    internalVecTwo.push(input[1][0]);
    internalVecTwo.push(input[1][1]);

    let mut outputVec: Vec<Vec<f64>> = Vec::with_capacity(2);
    outputVec.push(internalVecOne);
    outputVec.push(internalVecTwo);

    return outputVec;
}

fn transpose(matrix: &PriceDataMatrix) -> ArbitrarySizeMatrix {
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

fn get_gram_matrix(matrix: &PriceDataMatrix) -> TwoByTwoMatrix {
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

fn invert(matrix: &TwoByTwoMatrix) -> TwoByTwoMatrix {
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

fn multiply_matrix_by_vector(matrix: &ArbitrarySizeMatrix, vec: &Vec<f64>) -> Vec<f64> {
    let mut outputVec: Vec<f64> = Vec::with_capacity(matrix.len());

    for rowNum in 0..matrix.len() {
        outputVec.push(dot_product(&matrix[rowNum], vec))
    }

    return outputVec;
}

fn multiply_matrix_by_scalar(matrix: &TwoByTwoMatrix, scalar: f64) -> TwoByTwoMatrix {
    return [
        [matrix[0][0] * scalar, matrix[0][1] * scalar],
        [matrix[1][0] * scalar, matrix[1][1] * scalar],
    ];
}

fn multiply_matrix_by_matrix(
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

fn get_price_data_matrix(file: &str) -> PriceDataMatrix {
    let mut reader = match csv::Reader::from_path(format!("./data/{}.csv", file)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{e}");
            panic!("File could not be read")
        }
    };

    let mut outputMatrix: PriceDataMatrix = Vec::new();

    for result in reader.deserialize() {
        let record: CSVEntry = result.expect("Malformed CSV data");
        outputMatrix.push([1f64, record.price]);
    }

    return outputMatrix;
}

fn get_baseline_price_data_vector() -> Vec<f64> {
    let mut reader = match csv::Reader::from_path("./data/jet_fuel.csv") {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{e}");
            panic!("File could not be read")
        }
    };

    let mut outputVec: Vec<f64> = Vec::new();

    for result in reader.deserialize() {
        let record: CSVEntry = result.expect("Malformed CSV data");
        outputVec.push(record.price);
    }

    return outputVec;
}

fn plotResult(
    rSquared: f64,
    intercept: f64,
    mvhr: f64,
    priceData: PriceDataMatrix,
    baselineData: Vec<f64>,
    file: &str,
) {
    let graphTitle = if file == "wti" {
        "West Texas Intermediate"
    } else if file == "heating_oil" {
        "Heating Oil"
    } else if file == "brent" {
        "Brent"
    } else {
        "Unknown"
    };

    let fileName = format!("./output/{}.png", file);

    let root = BitMapBackend::new(&fileName, (800, 600)).into_drawing_area();
    root.fill(&style::WHITE).unwrap();

    let xMin = priceData.iter().fold(f64::INFINITY, |a, &b| a.min(b[1]));
    let xMax = priceData
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b[1]));
    let yMin = (xMin * mvhr) + intercept;
    let yMax = (xMax * mvhr) + intercept;

    let regLineColour = &style::BLACK;
    let priceDataColour = &style::RED;
    let residualColour = &style::BLUE;
    let backgroundColour = &style::WHITE.mix(0.5); // Faded a bit
    let axisColour = style::BLACK.stroke_width(2);

    let cleanZippedPriceData: Vec<(f64, f64)> = priceData
        .iter()
        .map(|[_placeholder, price]| *price)
        .zip(baselineData.iter().map(|price| *price))
        .collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(graphTitle, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(xMin..xMax, yMin..yMax)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // Scatter plot of the price data
    chart
        .draw_series(
            // x = independent (feature var), y = dependent (underlying asset prices we're hedging against)
            cleanZippedPriceData
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 5, priceDataColour.filled())),
        )
        .unwrap()
        .label(format!("R²: {rSquared}"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 1, y)], &style::BLACK));

    // The line itself
    chart
        .draw_series(LineSeries::new(
            vec![(xMin, yMin), (xMax, yMax)],
            &regLineColour,
        ))
        .unwrap()
        .label(format!("Slope (MVHR): {mvhr}"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 1, y)], &style::BLACK));

    // Residual tails
    let mut sortedResidualsData: Vec<Vec<(f64, f64)>> = cleanZippedPriceData
        .iter()
        .map(|(x, y)| {
            let predictedYVal = (*x * mvhr) + intercept;
            return vec![(*x, *y), (*x, predictedYVal)];
        })
        .collect();
    sortedResidualsData.sort_by(|a, b| a[0].0.total_cmp(&b[0].0));
    chart
        .draw_series(
            sortedResidualsData
                .into_iter()
                .map(|x| PathElement::new(x, residualColour)),
        )
        .unwrap()
        .label(format!("Intercept (Basis Drift): {intercept}"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 1, y)], &style::BLACK));

    // These are the x and y axes (crossing through 0,0). Wanted to make sure these are visible
    chart
        .draw_series(LineSeries::new(vec![(xMin, 0.0), (xMax, 0.0)], axisColour))
        .unwrap();
    chart
        .draw_series(LineSeries::new(vec![(0.0, yMin), (0.0, yMax)], axisColour))
        .unwrap();

    chart
        .configure_series_labels()
        .background_style(backgroundColour)
        .position(SeriesLabelPosition::LowerRight)
        .draw()
        .unwrap();
}

fn printResultStatistics(
    rSquared: &f64,
    residSumSquares: &f64,
    n: &f64,
    alpha: &f64,
    beta: &f64,
    invertedGramMatrix: &TwoByTwoMatrix,
) {
    let degFreedom = n - 2.0; // We have 1 variable so df = n - k - 1 = n - 2 
    let residualVariance = residSumSquares / degFreedom;
    let alphaStandErr = (residualVariance * invertedGramMatrix[0][0]).sqrt();
    let betaStandErr = (residualVariance * invertedGramMatrix[1][1]).sqrt();
    let alphaTStat = (alpha / alphaStandErr).abs();
    let betaTStat = (beta / betaStandErr).abs();

    println!("R^2 = {rSquared}, Intercept = {alpha}, MVHR = {beta}");
    if alphaTStat >= 2.0 {
        println!("Intercept (basis drift) statistically significant (T-stat: {alphaTStat})")
    } else {
        println!("Intercept (basis drift) statistically insignificant (T-stat: {alphaTStat})")
    }

    if betaTStat >= 2.0 {
        println!("Slope (MVHR) statistically significant (T-stat: {betaTStat})")
    } else {
        println!("Slope (MVHR) statistically insignificant (T-stat: {betaTStat})")
    }
}

fn runRegression(fileName: &str) {
    println!("Running regression for {}", fileName);
    let baselineData = get_baseline_price_data_vector();
    let priceData = get_price_data_matrix(fileName);

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
    let beta = match coefficients.get(1) {
        Some(f) => f,
        None => panic!("Missing beta coefficient"),
    };

    let mut sum: f64 = 0.0;
    for price in baselineData.iter() {
        sum += price;
    }
    let averageBaselinePrice = sum / baselineData.len() as f64;

    // Reuse sum for next loop
    sum = 0.0;
    for price in baselineData.iter() {
        sum += (price - averageBaselinePrice).powi(2);
    }
    let totalSumSquares = sum;

    // One final time
    sum = 0.0;
    for (index, [_placeholder, price]) in priceData.iter().enumerate() {
        let predictedPrice = (price * beta) + alpha;
        let actualPrice = baselineData.get(index).expect("Missing price data");
        sum += (actualPrice - predictedPrice).powi(2);
    }
    let residSumSquares = sum;
    let rSquared = 1.0 - (residSumSquares / totalSumSquares);

    printResultStatistics(
        &rSquared,
        &residSumSquares,
        &(baselineData.len() as f64),
        alpha,
        beta,
        &invertedGramMatrix,
    );
    plotResult(rSquared, *alpha, *beta, priceData, baselineData, fileName);
}

fn main() {
    runRegression("brent");
    runRegression("wti");
    runRegression("heating_oil");
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn multiply_matrices() {
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
}
