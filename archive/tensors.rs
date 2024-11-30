struct _Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl _Tensor {
    // Constructor for a tensor with a given shape, initialized to zeros
    fn _new(shape: Vec<usize>) -> _Tensor {
        assert!(
            !shape.contains(&0),
            "Shape dimensions must be positive and cannot include zero."
        );
        let total_size: usize = shape.iter().product();
        _Tensor {
            shape,
            data: vec![0.0; total_size],
        }
    }

    // Constructor for a tensor with a given shape and initial data
    // This requires that the length of data matches the product of the shape's dimensions
    fn _from_data(mut shape: Vec<usize>, data: Vec<f64>) -> _Tensor {
        let zero_count = shape.iter().filter(|&&x| x == 0).count();
        assert!(
            zero_count <= 1,
            "Only one dimension can be inferred (zero)."
        );

        if zero_count == 1 {
            let known_product: usize = shape.iter().filter(|&&x| x != 0).product();
            let inferred_index = shape.iter().position(|&x| x == 0).unwrap();
            shape[inferred_index] = data.len() / known_product;
        }
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "The product of the shape must match the length of the data."
        );
        _Tensor { shape, data }
    }

    fn matmul(&self, other: &_Tensor) -> _Tensor {
        assert_eq!(self.shape.len(), 2, "First tensor is not a 2D matrix.");
        assert_eq!(other.shape.len(), 2, "Second tensor is not a 2D matrix.");
        assert_eq!(
            self.shape[1], other.shape[0],
            "Dimensions do not match for matrix multiplication."
        );

        let result_rows = self.shape[0];
        let result_cols = other.shape[1];
        let mut result_data = vec![0.0; result_rows * result_cols];

        for i in 0..result_rows {
            for j in 0..result_cols {
                let mut sum = 0.0;
                for k in 0..self.shape[1] {
                    sum += self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j];
                }
                result_data[i * result_cols + j] = sum;
            }
        }

        _Tensor {
            shape: vec![result_rows, result_cols],
            data: result_data,
        }
    }

    // Matrix addition
    fn matadd(&self, other: &_Tensor) -> _Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Shapes do not match for matrix addition."
        );

        let result_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        _Tensor {
            shape: self.shape.clone(),
            data: result_data,
        }
    }
    // Matrix addition
    fn matsub(&self, other: &_Tensor) -> _Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Shapes do not match for matrix addition."
        );

        let result_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x - y)
            .collect();

        _Tensor {
            shape: self.shape.clone(),
            data: result_data,
        }
    }
}

impl std::ops::Mul for _Tensor {
    type Output = _Tensor;

    fn mul(self, other: _Tensor) -> _Tensor {
        let len1 = self.shape.len();
        let len2 = other.shape.len();
        let max_len = std::cmp::max(len1, len2);
        let mut result_shape = Vec::with_capacity(max_len);

        let mut stride1 = 1;
        let mut stride2 = 1;
        let mut expanded_data1 = self.data;
        let mut expanded_data2 = other.data;

        for i in 0..max_len {
            let dim1 = if i < len1 {
                self.shape[len1 - 1 - i]
            } else {
                1
            };
            let dim2 = if i < len2 {
                other.shape[len2 - 1 - i]
            } else {
                1
            };

            assert!(
                dim1 == dim2 || dim1 == 1 || dim2 == 1,
                "Shapes could not be broadcast together."
            );

            result_shape.insert(0, std::cmp::max(dim1, dim2));

            if dim1 == 1 && dim2 != 1 {
                expanded_data1 = expanded_data1
                    .chunks(stride1)
                    .flat_map(|chunk| chunk.iter().cycle().take(stride1 * dim2))
                    .copied()
                    .collect();
            }

            if dim2 == 1 && dim1 != 1 {
                expanded_data2 = expanded_data2
                    .chunks(stride2)
                    .flat_map(|chunk| chunk.iter().cycle().take(stride2 * dim1))
                    .copied()
                    .collect();
            }

            stride1 *= dim1;
            stride2 *= dim2;
        }

        let result_data = expanded_data1
            .into_iter()
            .zip(expanded_data2)
            .map(|(x, y)| x * y)
            .collect();

        _Tensor {
            shape: result_shape,
            data: result_data,
        }
    }
}

impl Clone for _Tensor {
    fn clone(&self) -> Self {
        _Tensor {
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }
}

fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

struct NeuralNetwork {
    weights: Vec<_Tensor>,
    biases: Vec<_Tensor>,
}

fn forward(net: &NeuralNetwork, input: &_Tensor) -> _Tensor {
    let mut current_output = input.clone();

    for (weight, bias) in net.weights.iter().zip(net.biases.iter()) {
        // Apply linear transformation: weight * input + bias
        current_output = weight.matmul(&current_output).matadd(bias);
        // Apply activation function (e.g., sigmoid)
        for elem in &mut current_output.data {
            *elem = sigmoid(elem);
        }
    }
    current_output
}

fn backprop(net: &NeuralNetwork, in: &_Tensor, out: &_Tensor) {
    
}

// Example Usage
fn main() {
    // Create two tensors of shape [2, 2] and initialize them with some data
    let tensor1 = _Tensor::_from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let tensor2 = _Tensor::_from_data(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    // Multiply the two tensors
    let result = tensor1 * tensor2;

    // Print the result
    println!("Resulting shape: {:?}", result.shape);
    println!("Resulting data: {:?}", result.data);

    // Create a tensor of shape [3] and another tensor of shape [3, 1]
    let tensor1 = _Tensor::_from_data(vec![3], vec![1.0, 2.0, 3.0]);
    let tensor2 = _Tensor::_from_data(vec![3, 1], vec![4.0, 5.0, 6.0]);

    // Multiply the two tensors (broadcasting will occur)
    let result = tensor1 * tensor2;

    // Print the result
    println!("Resulting shape: {:?}", result.shape);
    println!("Resulting data: {:?}", result.data);

    // Define a 2x3 tensor (matrix)
    let tensor1 = _Tensor::_from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Define a 3x2 tensor (matrix)
    let tensor2 = _Tensor::_from_data(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    // Perform matrix multiplication
    let result = tensor1.matmul(&tensor2);

    // Print the result
    println!("Resulting shape: {:?}", result.shape);
    println!("Resulting data: {:?}", result.data);
}
