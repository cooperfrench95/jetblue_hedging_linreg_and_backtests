#![allow(non_snake_case)]

pub mod linear_algebra;
#[cfg(test)]
mod tests;

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

struct BacktestResult {
    cumulativeMonthlyPnL: f64,
    cumPnLPeak: f64,
    maxDrawdown: f64,
    monthlyPnL: Vec<f64>,
    stdDev: f64,
    mvhr: f64,
    numContracts: f64,
}

struct BacktestResults {
    noHedge: BacktestResult,
    withWTIHedge: BacktestResult,
    withBrentHedge: BacktestResult,
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

fn printBacktestStats(backtestResults: &BacktestResults) {
    println!("\n-------------------BACKTESTS-------------------\n");
    println!("-------No hedge-------");
    println!(
        "Max. drawdown: -${:.2} million",
        backtestResults.noHedge.maxDrawdown
    );
    println!(
        "Max. monthly loss: -${:.2?} million",
        backtestResults
            .noHedge
            .monthlyPnL
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .abs()
    );
    println!(
        "Std. dev (volatility): ${:.2} million",
        backtestResults.noHedge.stdDev
    );
    println!("MVHR: {:.2} (N/A)", backtestResults.noHedge.mvhr);
    println!(
        "Num. contracts: {:?}",
        backtestResults.noHedge.numContracts as usize
    );
    println!("---------WTI----------");
    println!(
        "Max. drawdown: -${:.2} million",
        backtestResults.withWTIHedge.maxDrawdown
    );
    println!(
        "Max. monthly loss: -${:.2?} million",
        backtestResults
            .withWTIHedge
            .monthlyPnL
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .abs()
    );
    println!(
        "Std. dev (volatility): ${:.2} million",
        backtestResults.withWTIHedge.stdDev
    );
    println!("MVHR: {:.2}", backtestResults.withWTIHedge.mvhr);
    println!(
        "Num. contracts: {:?}",
        backtestResults.withWTIHedge.numContracts as usize
    );
    println!("--------Brent---------");
    println!(
        "Max. drawdown: -${:.2} million",
        backtestResults.withBrentHedge.maxDrawdown
    );
    println!(
        "Max. monthly loss: -${:.2?} million",
        backtestResults
            .withBrentHedge
            .monthlyPnL
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .abs()
    );
    println!(
        "Std. dev (volatility): ${:.2} million",
        backtestResults.withBrentHedge.stdDev
    );
    println!("MVHR: {:.2}", backtestResults.withBrentHedge.mvhr);
    println!(
        "Num. contracts: {:?}",
        backtestResults.withBrentHedge.numContracts as usize
    );
    println!("\n-----------------------------------------------");
}

fn plotBacktestResult(backtestResults: BacktestResults) {
    let fileName = "./output/backtest.png";
    let root = BitMapBackend::new(&fileName, (1280, 720)).into_drawing_area();
    root.fill(&style::WHITE).unwrap();

    let baselineColour = style::MAGENTA.stroke_width(2);
    let wtiColour = style::RED.stroke_width(2);
    let brentColour = style::BLUE.stroke_width(2);
    let backgroundColour = &style::WHITE.mix(0.5); // Faded a bit
    let axisColour = style::BLACK.stroke_width(2);
    let n = backtestResults.noHedge.monthlyPnL.len() as f64;

    let yMin = [
        &backtestResults.noHedge.monthlyPnL,
        &backtestResults.withBrentHedge.monthlyPnL,
        &backtestResults.withWTIHedge.monthlyPnL,
    ]
    .iter()
    .flat_map(|v| v.iter())
    .fold(f64::INFINITY, |a, &b| a.min(b));
    let yMax = [
        &backtestResults.noHedge.monthlyPnL,
        &backtestResults.withBrentHedge.monthlyPnL,
        &backtestResults.withWTIHedge.monthlyPnL,
    ]
    .iter()
    .flat_map(|v| v.iter())
    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption("PnL Backtest (2007-2011)", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..n, yMin..yMax)
        .unwrap();

    chart
        .configure_mesh()
        .y_desc("PnL ($million USD)")
        .x_desc("Month")
        .draw()
        .unwrap();

    // No hedge
    chart
        .draw_series(LineSeries::new(
            backtestResults
                .noHedge
                .monthlyPnL
                .iter()
                .enumerate()
                .map(|(x, &y)| (x as f64, y)),
            baselineColour,
        ))
        .unwrap()
        .label(format!(
            "No Hedge (std. dev: ${:.2} million)",
            backtestResults.noHedge.stdDev
        ))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], baselineColour));

    // WTI
    chart
        .draw_series(LineSeries::new(
            backtestResults
                .withWTIHedge
                .monthlyPnL
                .iter()
                .enumerate()
                .map(|(x, &y)| (x as f64, y)),
            wtiColour,
        ))
        .unwrap()
        .label(format!(
            "WTI (std. dev: ${:.2} million)",
            backtestResults.withWTIHedge.stdDev
        ))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], wtiColour));

    // Brent
    chart
        .draw_series(LineSeries::new(
            backtestResults
                .withBrentHedge
                .monthlyPnL
                .iter()
                .enumerate()
                .map(|(x, &y)| (x as f64, y)),
            brentColour,
        ))
        .unwrap()
        .label(format!(
            "Brent (std. dev: ${:.2} million)",
            backtestResults.withBrentHedge.stdDev
        ))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], brentColour));

    // These are the x and y axes (crossing through 0,0). Wanted to make sure these are visible
    chart
        .draw_series(LineSeries::new(vec![(0.0, 0.0), (n, 0.0)], axisColour))
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

fn plotResult(
    rSquared: f64,
    intercept: f64,
    mvhr: &f64,
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

    println!("\nR^2 = {rSquared}, Intercept = {alpha}, MVHR = {beta}");
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

fn runBacktests() {
    println!("\n-------------------REGRESSION-------------------\n");
    let brentMVHR = runRegression("brent");
    let wtiMVHR = runRegression("wti");
    runRegression("heating_oil");

    let MONTHLY_FUEL_HEDGE: f64 = 20_000_000.0;
    let CONTRACT_SIZE: f64 = 42_000.0;
    let AMOUNT_CONTRACTS_BRENT: f64 = ((MONTHLY_FUEL_HEDGE / CONTRACT_SIZE) * brentMVHR).round();
    let AMOUNT_CONTRACTS_WTI: f64 = ((MONTHLY_FUEL_HEDGE / CONTRACT_SIZE) * wtiMVHR).round();
    let MONTHLY_FUEL_CONSUMPTION: f64 = 525_000_000.0 / 12.0;
    let SCALE_AMOUNT = 1_000_000.0;

    let baselineData = get_baseline_price_data_vector();
    let priceDataBrent = get_price_data_matrix("brent");
    let priceDataWTI = get_price_data_matrix("wti");

    let capacity = baselineData.len();

    let mut backtestResults = BacktestResults {
        noHedge: BacktestResult {
            cumulativeMonthlyPnL: 0.0,
            cumPnLPeak: 0.0,
            maxDrawdown: 0.0,
            monthlyPnL: Vec::with_capacity(capacity),
            stdDev: 0.0,
            mvhr: 1.0,         // N/A in this case
            numContracts: 0.0, // N/A again
        },
        withBrentHedge: BacktestResult {
            cumulativeMonthlyPnL: 0.0,
            cumPnLPeak: 0.0,
            maxDrawdown: 0.0,
            monthlyPnL: Vec::with_capacity(capacity),
            stdDev: 0.0,
            mvhr: brentMVHR,
            numContracts: AMOUNT_CONTRACTS_BRENT,
        },
        withWTIHedge: BacktestResult {
            cumulativeMonthlyPnL: 0.0,
            cumPnLPeak: 0.0,
            maxDrawdown: 0.0,
            monthlyPnL: Vec::with_capacity(capacity),
            stdDev: 0.0,
            mvhr: wtiMVHR,
            numContracts: AMOUNT_CONTRACTS_WTI,
        },
    };

    for (index, priceChange) in baselineData.iter().enumerate() {
        let PnL_on_underlying = MONTHLY_FUEL_CONSUMPTION * priceChange / SCALE_AMOUNT;
        let brentPriceChange = priceDataBrent[index][1];
        let wtiPriceChange = priceDataWTI[index][1];

        // No hedge
        backtestResults.noHedge.monthlyPnL.push(PnL_on_underlying);
        backtestResults.noHedge.cumulativeMonthlyPnL += PnL_on_underlying;
        if backtestResults.noHedge.cumPnLPeak < backtestResults.noHedge.cumulativeMonthlyPnL {
            backtestResults.noHedge.cumPnLPeak = backtestResults.noHedge.cumulativeMonthlyPnL;
        }
        let noHedgeDrawdown =
            backtestResults.noHedge.cumulativeMonthlyPnL - backtestResults.noHedge.cumPnLPeak;
        if noHedgeDrawdown.abs() > backtestResults.noHedge.maxDrawdown.abs() {
            backtestResults.noHedge.maxDrawdown = noHedgeDrawdown.abs();
        }

        // Brent
        let brentProfit = PnL_on_underlying
                // Note the negative sign here, because it's a hedge the price movement has the opposing effect on PnL
                + -((brentPriceChange * AMOUNT_CONTRACTS_BRENT * CONTRACT_SIZE) / SCALE_AMOUNT);
        backtestResults.withBrentHedge.monthlyPnL.push(brentProfit);
        backtestResults.withBrentHedge.cumulativeMonthlyPnL += brentProfit;
        if backtestResults.withBrentHedge.cumPnLPeak
            < backtestResults.withBrentHedge.cumulativeMonthlyPnL
        {
            backtestResults.withBrentHedge.cumPnLPeak =
                backtestResults.withBrentHedge.cumulativeMonthlyPnL;
        }
        let withBrentHedgeDrawdown = backtestResults.withBrentHedge.cumulativeMonthlyPnL
            - backtestResults.withBrentHedge.cumPnLPeak;
        if withBrentHedgeDrawdown.abs() > backtestResults.withBrentHedge.maxDrawdown.abs() {
            backtestResults.withBrentHedge.maxDrawdown = withBrentHedgeDrawdown.abs();
        }
        // WTI
        let wtiProfit = PnL_on_underlying
            // Note the negative sign here, because it's a hedge the price movement has the opposing effect on PnL
            + -((wtiPriceChange * AMOUNT_CONTRACTS_WTI * CONTRACT_SIZE) / SCALE_AMOUNT);
        backtestResults.withWTIHedge.monthlyPnL.push(wtiProfit);
        backtestResults.withWTIHedge.cumulativeMonthlyPnL += wtiProfit;
        if backtestResults.withWTIHedge.cumPnLPeak
            < backtestResults.withWTIHedge.cumulativeMonthlyPnL
        {
            backtestResults.withWTIHedge.cumPnLPeak =
                backtestResults.withWTIHedge.cumulativeMonthlyPnL;
        }
        let withWTIHedgeDrawdown = backtestResults.withWTIHedge.cumulativeMonthlyPnL
            - backtestResults.withWTIHedge.cumPnLPeak;
        if withWTIHedgeDrawdown.abs() > backtestResults.withWTIHedge.maxDrawdown.abs() {
            backtestResults.withWTIHedge.maxDrawdown = withWTIHedgeDrawdown.abs();
        }
    }

    backtestResults.noHedge.stdDev = standardDeviation(&backtestResults.noHedge.monthlyPnL);
    backtestResults.withBrentHedge.stdDev =
        standardDeviation(&backtestResults.withBrentHedge.monthlyPnL);
    backtestResults.withWTIHedge.stdDev =
        standardDeviation(&backtestResults.withWTIHedge.monthlyPnL);

    printBacktestStats(&backtestResults);
    plotBacktestResult(backtestResults);
}

fn standardDeviation(values: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;

    for entry in values.iter() {
        sum += (entry - mean).powi(2);
    }

    let stdDev = (sum / (n - 1.0)).sqrt();
    return stdDev;
}

fn runRegression(fileName: &str) -> f64 {
    println!("\nRunning regression for {}", fileName);
    let baselineData = get_baseline_price_data_vector();
    let priceData = get_price_data_matrix(fileName);

    let gramMatrix = linear_algebra::get_gram_matrix(&priceData);
    let priceDataTranspose = linear_algebra::transpose(&priceData);
    let baselineToPriceCovariance =
        linear_algebra::multiply_matrix_by_vector(&priceDataTranspose, &baselineData);
    let invertedGramMatrix = linear_algebra::invert(&gramMatrix);
    let coefficients = linear_algebra::multiply_matrix_by_vector(
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
    plotResult(rSquared, *alpha, beta, priceData, baselineData, fileName);
    return *beta;
}

fn main() {
    runBacktests();
}
