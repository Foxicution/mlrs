pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    // ========================================================================
    // Tensor creation
    // Constructor for a tensor with a given shape, initialized to zeros
    pub fn new(shape: Vec<usize>) -> Tensor {
        assert!(
            !shape.contains(&0),
            "Shape dimensions must be positive and cannot include zero."
        );
        let total_size: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![0.0; total_size],
        }
    }

    // Constructor for a tensor with a given shape and initial data
    // This requires that the length of data matches the product of the shape's dimensions
    pub fn from_data(mut shape: Vec<usize>, data: Vec<f64>) -> Tensor {
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
        Tensor { shape, data }
    }

    // ========================================================================
    // Tensor operations
    fn elemwise_with_broadcast<F>(&self, other: &Tensor, op: F) -> Tensor
    where
        F: Fn(f64, f64) -> f64,
    {
        let len1 = self.shape.len();
        let len2 = other.shape.len();
        let max_len = std::cmp::max(len1, len2);
        let mut result_shape = Vec::with_capacity(max_len);

        let mut stride1 = 1;
        let mut stride2 = 1;
        let mut expanded_data1 = self.data.clone();
        let mut expanded_data2 = other.data.clone();

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
            .map(|(x, y)| op(x, y))
            .collect();

        Tensor {
            shape: result_shape,
            data: result_data,
        }
    }

    // ========================================================================
    // Matrix operations
    pub fn matmul(&self, other: &Tensor) -> Tensor {
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

        Tensor {
            shape: vec![result_rows, result_cols],
            data: result_data,
        }
    }

    // Matrix addition
    pub fn matadd(&self, other: &Tensor) -> Tensor {
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

        Tensor {
            shape: self.shape.clone(),
            data: result_data,
        }
    }
    // Matrix subtraction
    pub fn matsub(&self, other: &Tensor) -> Tensor {
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

        Tensor {
            shape: self.shape.clone(),
            data: result_data,
        }
    }
}

impl std::ops::Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        self.elemwise_with_broadcast(&other, |x, y| x * y)
    }
}

impl std::ops::Div for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        self.elemwise_with_broadcast(&other, |x, y| x / y)
    }
}

impl std::ops::Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        self.elemwise_with_broadcast(&other, |x, y| x + y)
    }
}

impl std::ops::Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        self.elemwise_with_broadcast(&other, |x, y| x - y)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }
}
