use rand::{Rng};

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

 impl Matrix{
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix { 
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }
    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::rng();
        let mut res = Matrix::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = rng.random_range(-1.0..1.0);
            }
        }
        res
    }


    pub fn from(data: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Incompatible matrix dimensions");
        }

        let mut res = Matrix::zeros(self.rows, other.cols);
        //multiplication de deux matrice de taille Aab et Bcd, pour avoir au final une matrice de taille Cad
        for i in 0..self.rows { 
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                res.data[i][j] = sum;
            }
        }

        res
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Incompatible matrix dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }

        res
    }
    //produit scalaire entre deux matrice
    pub fn dot_multiply(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Incompatible matrix dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }

        res
    }
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Incompatible matrix dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        res
    }
    //fonction dynamique
    //map va appliquer une fonction à chaque élément de la matrice
    pub fn map(&self, function: &dyn Fn(f64) -> f64) -> Matrix {
        Matrix::from(
            (self.data)
            .clone()
            .into_iter()
            .map(|row| {
                row.into_iter()
                .map(|val| function(val)).collect()
            }).collect()
        )
    }

    pub fn transpose(&self) -> Matrix {
        let mut res = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }
        res
    }
}