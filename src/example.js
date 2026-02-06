const transposed = [
  [1, 1, 1],
  [12, 13, 14]
]

const originalMatrix = [
  [1, 12],
  [1, 13],
  [1, 14]
]

const rowLength = originalMatrix[0].length(); // 2
const outputLength = transposed.length(); // 2
const columnLength = originalMatrix.length(); // 3

// function multiplyMatrix(matrix1, matrix2) {
//   const outputLength = matrix1.length;
//   const otherDimension = matrix1[0].length;

//   if (matrix1.length !== matrix2[0].length) {
//     throw new Error("Bad length")
//   }

//   for (let i = 0; i < otherDimension; i++) {

//   }
// }