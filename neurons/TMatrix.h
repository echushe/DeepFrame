#pragma once
#include "Vector.h"
#include "Shape.h"
#include "Coordinate.h"
#include "Exceptions.h"
#include <iostream>

namespace neurons
{
    /*
    Almost all calculations of deep learning depend on matrices.
    A matrix can be of one dimension, two dimensions, three dimensions or even more dimensons.
    A batch of training set or test set can be represented by a list or a vector of matrices.
    The weights of a neural network layer is a matrix.
    The bias of a neural network layer is a matrix.
    The labels/targets of a network are also saved in a matrix.
    */
    template <typename dtype = double>
    class TMatrix
    {
        // These friend classes are all functions who may operate on matrices
        //friend class Linear;
        //friend class Sigmoid;
        //friend class Tanh;
        //friend class Relu;
        //friend class Softmax;
        //friend class HalfSquareError;
        //friend class Sigmoid_CrossEntropy;
        //friend class Softmax_CrossEntropy;
        //friend class Conv_1d;
        //friend class Conv_2d;
        //friend class Pooling_2d;
        //friend class MaxPooling_2d;
        //friend class EM_1d;
        //friend class Linear_Regression;

    public:
        // Shape of this matrix
        Shape m_shape;
        // All the elements of this matrix
        dtype *m_data;

    public:
        // The default constructor.
        // The matrix created by default constructor is not usable until
        // it is assigned by a usable matrix
        TMatrix();

        // Create a matrix of a certain shape. But none of its elements are initialized
        TMatrix(const Shape & shape);

        // Create a matrix of a certain shape. All of its elements are initialized by a value
        TMatrix(const Shape & shape, dtype value);

        // Combine an array if matrices into one matrix
        explicit TMatrix(const std::vector<TMatrix> & matrices);

        // Copy constructor
        TMatrix(const TMatrix & other);

        // Move constructor
        TMatrix(TMatrix && other);

        // Create a matrix from a vector
        TMatrix(const Vector & vec, bool transpose = false);

        ~TMatrix();

    public:
        // Copy assignment
        TMatrix & operator = (const TMatrix & other);

        // Move assignment
        TMatrix & operator = (TMatrix && other);

        // Assign a scalar value to all matrix elements
        TMatrix & operator = (dtype scalar);

        // Get an element of a certain position
        dtype at(const Coordinate & pos) const;

        // Get an element of a certain position. This element can be updated
        dtype & at(const Coordinate & pos);

        // Get an element of a certain position
        dtype operator [] (const Coordinate & pos) const;

        // Get an element of a certain position. This element can be updated
        dtype & operator [] (const Coordinate & pos);

        // An implicite way to get a flatened matrix (vector)
        operator Vector() const;

        // Get a flatened matrix (vector)
        Vector flaten() const;

        TMatrix & operator += (const TMatrix & other);

        TMatrix & operator -= (const TMatrix & other);

        TMatrix & operator += (dtype scalar);

        TMatrix & operator -= (dtype scalar);

        TMatrix & operator *= (dtype scalar);

        TMatrix & operator /= (dtype scalar);

        // Randomize all elements of this matrix. Distribution of the elements complies with
        // Gaussian distribution (normal distribution).
        // dtype mu: mean of these elements
        // dtype sigma: sqrt(variance) of these elements
        TMatrix & gaussian_random(dtype mu, dtype sigma);

        // Randomize all elements of this matrix. Distribution of the elements complies with
        // uniform distribution.
        TMatrix & uniform_random(dtype min, dtype max);

        // Normalize all elements of this matrix to the range of [min, max]
        TMatrix & normalize(dtype min, dtype max);

        // Normalize all elements of this matrix to mean == 0 and variance == 1
        TMatrix & normalize();

        // Change shape of this matrix.
        // For example, we can change [20, 30] to [2, 10, 3, 5, 2], or
        // change [4, 5, 6] to [3, 5, 8].
        // However, size of this matrix will not change, which means 20 * 30 == 2 * 10 * 3 * 5 * 2,
        // or 4 * 5 * 6 == 3 * 5 * 8.
        // Order of all elements in this matrix will not change either
        TMatrix & reshape(const Shape & shape);

        // Extend one dimension of the matrix.
        // For example, [30, 50] is extended to [1, 30, 50].
        // This is a special case of reshape.
        TMatrix & left_extend_shape();

        // Extend one dimension of the matrix.
        // For example, [30, 50] is extended to [30, 50, 1].
        // This is a special case of reshape.
        TMatrix & right_extend_shape();

        // Get duplicates of left extended version of the matrix.
        // For example, [35, 45] can be extend to [12, 35, 45] in which
        // there are exactly 12 copies of [35, 45]
        TMatrix get_left_extended(lint duplicate) const;

        /* Scale one dimension of the matrix with a vector of scales
        For example, if a two dimensional matrix A is like this:

        | a11, a12, a13 |
        | a21, a22, a23 |
        | a31, a32, a33 |
        | a41, a42, a43 |

        And vector B is [ b1, b2, b3 ], vector C is [ c1, c2, c3, c4 ]
        Then the result of A.scale_one_dimension(1, B) is:

        | a11 * b1, a12 * b2, a13 * b3 |
        | a21 * b1, a22 * b2, a23 * b3 |
        | a31 * b1, a32 * b2, a33 * b3 |
        | a41 * b1, a42 * b2, a43 * b3 |

        Then the result of A.scale_one_dimension(0, C) is:

        | a11 * c1, a12 * c1, a13 * c1 |
        | a21 * c2, a22 * c2, a23 * c2 |
        | a31 * c3, a32 * c3, a33 * c3 |
        | a41 * c4, a42 * c4, a43 * c4 |

        It seems this kind of calculation is needed in back propagation.
        Perhaps there is a better way to name this function
        */
        TMatrix & scale_one_dimension(lint dim, const Vector & scales);

        // Get coordinate of the largest element
        Coordinate argmax() const;

        // Get value of the largest element
        dtype max() const;

        // Get coordinate of the lowest element
        Coordinate argmin() const;

        // Get value of the lowest element
        dtype min() const;

        // Get mean of all the elements
        dtype mean() const;

        // Get variance of all the elements of this matrix
        dtype variance() const;

        // Collapse a certain dimension of a matrix into a vector of matrices
        // For example, a matrix of shape [4, 6, 5, 7], if we collapse it with argument dim = 1,
        // it will be turned into 6 matrices of shape [4, 5, 7]
        std::vector<TMatrix> collapse(lint dim) const;

        // Collapse a certain dimension of a matrix, and merge all matrices into one matrix
        // For example, a matrix of shape [4, 6, 5, 7], if we fuse it with argument dim = 1,
        // it will be turned into a sum of 6 matrices of shape [4, 5, 7]
        TMatrix fuse(lint dim) const;

        // Collapse a certain dimension of a matrix, and get a mean of all the matrices
        // For example, a matrix of shape [4, 6, 5, 7], if we get its reduce_mean with argument dim = 1,
        // it will be turned into a mean of 6 matrices of shape [4, 5, 7]
        TMatrix reduce_mean(lint dim) const;


        dtype euclidean_norm() const;


    public:
        // Get shape of the matrix
        Shape shape() const;

    private:
        // friend bool operator == (const TMatrix &left, const TMatrix &right);

        // friend bool operator != (const TMatrix &left, const TMatrix &right);

        // friend TMatrix operator + (const TMatrix &left, const TMatrix &right);

        // friend TMatrix operator - (const TMatrix &left, const TMatrix &right);

        // friend TMatrix matrix_multiply(const TMatrix & left, const TMatrix & right, lint l_dims_merge, lint r_dims_merge);

        // friend TMatrix matrix_multiply(const TMatrix & left, const TMatrix & right);

        // friend TMatrix multiply(const TMatrix & left, const TMatrix & right);

        // friend dtype dot_product(const TMatrix & left, const TMatrix & right);

        // friend TMatrix operator * (const TMatrix &left, const TMatrix &right);

        // friend TMatrix operator * (dtype scalar, const TMatrix &right);

        // friend TMatrix operator * (const TMatrix &left, dtype scalar);

        // friend TMatrix operator / (const TMatrix &left, dtype scalar);

        // friend TMatrix transpose(const TMatrix & in);

        // friend std::ostream & operator << (std::ostream & os, const TMatrix & m);

    public:
        void print(std::ostream & os, lint dim_index, const dtype *start, size_t block_size) const;
    };

    // Overloading of a == b
    template <typename dtype>
    bool operator == (const TMatrix<dtype> &left, const TMatrix<dtype> &right)
    {
        if (left.m_shape != right.m_shape)
        {
            return false;
        }

        lint size = left.m_shape.size();

        for (lint i = 0; i < size; ++i)
        {
            if (left.m_data[i] != right.m_data[i])
            {
                return false;
            }
        }

        return true;
    }

    // Overloading of a != b
    template <typename dtype>
    bool operator != (const TMatrix<dtype> &left, const TMatrix<dtype> &right)
    {
        return !(left == right);
    }

    // Overloading of a + b
    template <typename dtype>
    TMatrix<dtype> operator + (const TMatrix<dtype> &left, const TMatrix<dtype> &right)
    {
        if (left.m_shape.size() != right.m_shape.size())
        {
            throw std::invalid_argument(invalid_shape);
        }

        lint size = left.m_shape.size();

        TMatrix<dtype> mat{ left.m_shape };
        for (lint i = 0; i < size; ++i)
        {
            mat.m_data[i] = left.m_data[i] + right.m_data[i];
        }

        return mat;
    }

    // Overloading of a - b
    template <typename dtype>
    TMatrix<dtype> operator - (const TMatrix<dtype> &left, const TMatrix<dtype> &right)
    {
        if (left.m_shape.size() != right.m_shape.size())
        {
            throw std::invalid_argument(invalid_shape);
        }

        lint size = left.m_shape.size();

        TMatrix<dtype> mat{ left.m_shape };
        for (lint i = 0; i < size; ++i)
        {
            mat.m_data[i] = left.m_data[i] - right.m_data[i];
        }

        return mat;
    }

    // TMatrix multiplication
    // Which dimensions should be merged together is manually defined here.
    // For example, two matrices: [7, 3, 2, 5, 6] and [6, 5, 6, 4], if l_dims_merge == 2 and r_dims_merge == 2,
    // the shape of result will be [7, 3, 2, 6, 4]. However, if l_dims_merge == 4 and r_dims_merge == 3,
    // the shape of output will be [7, 4].
    // Note: Total sizes (number of elements) of dimensions to merge should be equal betweem the left and right.
    template <typename dtype>
    TMatrix<dtype> matrix_multiply(const TMatrix<dtype> & left, const TMatrix<dtype> & right, lint l_dims_merge, lint r_dims_merge)
    {
        //
        if (left.m_shape.dim() < 2 || right.m_shape.dim() < 2)
        {
            throw std::invalid_argument(std::string(
                "TMatrix multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
        }

        // Check shape compatibilities between these 2 matrices
        Shape left_sh = left.m_shape;
        Shape right_sh = right.m_shape;

        Shape left_rows_sh = left_sh.sub_shape(0, left_sh.dim() - l_dims_merge - 1);
        Shape left_cols_sh = left_sh.sub_shape(left_sh.dim() - l_dims_merge, left_sh.dim() - 1);

        Shape right_rows_sh = right_sh.sub_shape(0, r_dims_merge - 1);
        Shape right_cols_sh = right_sh.sub_shape(r_dims_merge, right_sh.dim() - 1);

        lint left_columns = left_cols_sh.size();
        lint right_rows = right_rows_sh.size();

        if (left_columns != right_rows)
        {
            throw std::invalid_argument(incompatible_shape);
        }

        lint left_rows = left_rows_sh.size();
        lint right_columns = right_cols_sh.size();

        // Calculation
        TMatrix<dtype> mat{ left_rows_sh + right_cols_sh };
        dtype *left_start = left.m_data;
        dtype *right_start = right.m_data;
        dtype *mat_p = mat.m_data;

        for (lint i = 0; i < left_rows; ++i)
        {
            for (lint j = 0; j < right_columns; ++j)
            {
                *mat_p = 0.0;
                dtype * left_p = left_start;
                dtype * right_p = right_start;

                for (lint k = 0; k < left_columns; ++k)
                {
                    *mat_p += *left_p * *right_p;
                    ++left_p;
                    right_p += right_columns;
                }

                ++right_start;
                ++mat_p;
            }

            left_start += left_columns;
            right_start = right.m_data;
        }

        return mat;
    }

    // TMatrix multiplication
    // This function can automatically figure out which dimensions should be merged together.
    // For example, two matrices: [7, 3, 3, 5, 4] and [10, 6, 6, 4]
    // [3, 5, 4] of the left matrix and [10, 6] of the right matrix will be merged.
    // then, the shape of result will be [7, 3, 6, 4].
    // This function will throw out an exception if no appropriate dimensions to merge can be found.
    template <typename dtype>
    TMatrix<dtype> matrix_multiply(const TMatrix<dtype> & left, const TMatrix<dtype> & right)
    {
        /*
        if (1 == left.m_shape.dim() || 1 == right.m_shape.dim())
        {
        if (1 == left.m_shape.dim())
        {
        TMatrix extended_left{ left };
        extended_left.left_extend_shape();
        return extended_left * right;
        }
        else
        {
        TMatrix extended_right{ right };
        extended_right.right_extend_shape();
        return left * extended_right;
        }
        }
        */

        if (left.m_shape.dim() < 2 || right.m_shape.dim() < 2)
        {
            throw std::invalid_argument(std::string(
                "TMatrix multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
        }

        // Check shape compatibilities between these 2 matrices
        lint left_columns;
        lint left_rows;
        // lint right_rows;
        lint right_columns;

        Shape left_sh = left.m_shape;
        Shape right_sh = right.m_shape;

        Shape left_rows_sh;
        Shape left_cols_sh;

        Shape right_rows_sh;
        Shape right_cols_sh;

        bool can_multiply = false;

        for (lint l = left_sh.dim() - 1, r = 0; l >= 1 && r < right_sh.dim() - 1;)
        {
            left_cols_sh = left_sh.sub_shape(l, left_sh.dim() - 1);
            right_rows_sh = right_sh.sub_shape(0, r);

            if (left_cols_sh.size() == right_rows_sh.size())
            {
                left_rows_sh = left_sh.sub_shape(0, l - 1);
                left_rows = left_rows_sh.size();
                left_columns = left_cols_sh.size();

                // right_rows = left_columns;
                right_cols_sh = right_sh.sub_shape(r + 1, right_sh.dim() - 1);
                right_columns = right_cols_sh.size();

                can_multiply = true;
                break;
            }
            else if (left_cols_sh.size() > right_rows_sh.size())
            {
                ++r;
            }
            else
            {
                --l;
            }
        }

        if (!can_multiply)
        {
            throw std::invalid_argument(incompatible_shape);
        }

        // Calculation

        TMatrix<dtype> mat{ left_rows_sh + right_cols_sh };
        dtype *left_start = left.m_data;
        dtype *right_start = right.m_data;
        dtype *mat_p = mat.m_data;

        for (lint i = 0; i < left_rows; ++i)
        {
            for (lint j = 0; j < right_columns; ++j)
            {
                *mat_p = 0;
                dtype * left_p = left_start;
                dtype * right_p = right_start;

                for (lint k = 0; k < left_columns; ++k)
                {
                    *mat_p += *left_p * *right_p;
                    ++left_p;
                    right_p += right_columns;
                }

                ++right_start;
                ++mat_p;
            }

            left_start += left_columns;
            right_start = right.m_data;
        }

        return mat;
    }

    // Multiplication element by element.
    // The two matrices should have the same shape.
    template <typename dtype>
    TMatrix<dtype> multiply(const TMatrix<dtype> & left, const TMatrix<dtype> & right)
    {
        if (left.m_shape != right.m_shape)
        {
            throw std::invalid_argument(invalid_shape);
        }

        lint size = left.m_shape.size();

        TMatrix<dtype> mat{ left.m_shape };
        for (lint i = 0; i < size; ++i)
        {
            mat.m_data[i] = left.m_data[i] * right.m_data[i];
        }

        return mat;
    }

    // Dot product of two matrices
    // The two matrices should have the same amount of elements
    template <typename dtype>
    dtype dot_product(const TMatrix<dtype> & left, const TMatrix<dtype> & right)
    {
        if (left.m_shape.size() != right.m_shape.size())
        {
            throw std::invalid_argument(incompatible_size);
        }

        lint size = left.m_shape.size();
        dtype sum = 0.0;

        for (lint i = 0; i < size; ++i)
        {
            sum += left.m_data[i] * right.m_data[i];
        }

        return sum;
    }

    // TMatrix multiplication
    // This function can automatically figure out which dimensions should be merged together.
    // For example, two matrices: [7, 3, 3, 5, 4] and [10, 6, 6, 4]
    // [3, 5, 4] of the left matrix and [10, 6] of the right matrix will be merged.
    // then, the shape of result will be [7, 3, 6, 4].
    // This function will throw out an exception if no appropriate dimensions to merge can be found.
    template <typename dtype>
    TMatrix<dtype> operator * (const TMatrix<dtype> &left, const TMatrix<dtype> &right)
    {
        return matrix_multiply(left, right);
    }

    // Overloading of a * b, b is a scalar
    template <typename dtype>
    TMatrix<dtype> operator * (const TMatrix<dtype> &left, dtype scalar)
    {
        TMatrix<dtype> mat{ left.m_shape };

        lint size = left.m_shape.size();
        for (lint i = 0; i < size; ++i)
        {
            mat.m_data[i] = left.m_data[i] * scalar;
        }

        return mat;
    }

    // Overloading of a * b, a is a scalar
    template <typename dtype>
    TMatrix<dtype> operator * (dtype scalar, const TMatrix<dtype> &right)
    {
        TMatrix mat{ right.m_shape };

        lint size = right.m_shape.size();
        for (lint i = 0; i < size; ++i)
        {
            mat.m_data[i] = right.m_data[i] * scalar;
        }

        return mat;
    }

    // Overloading of a / b, b is a scalar
    template <typename dtype>
    TMatrix<dtype> operator / (const TMatrix<dtype> &left, dtype scalar)
    {
        TMatrix<dtype> mat{ left.m_shape };

        lint size = left.m_shape.size();
        for (lint i = 0; i < size; ++i)
        {
            mat.m_data[i] = left.m_data[i] / scalar;
        }

        return mat;
    }

    // Calculate power of a matrix
    template <typename dtype>
    TMatrix<dtype> matrix_pow(const TMatrix<dtype> & mat, int n)
    {
        if (n == 1)
        {
            return mat;
        }
        else if (n > 1)
        {
            TMatrix<dtype> child = matrix_pow(mat, n / 2);

            if (n % 2 == 0)
            {
                return child * child;
            }
            else
            {
                return child * child * mat;
            }
        }
        else
        {
            return TMatrix<lint>{};
        }
    }

    // Get a transposed matrix of a matrix
    template <typename dtype>
    TMatrix<dtype> transpose(const TMatrix<dtype> & in)
    {
        Shape reversed_shape{ reverse(in.m_shape) };
        TMatrix<dtype> transposed{ reversed_shape };

        lint size = in.m_shape.size();

        lint plus_pos;
        lint dim_size = reversed_shape.dim();
        lint *coord_cache = new lint[dim_size];
        lint *jump_forward_cache = new lint[dim_size];

        for (lint i = dim_size - 1; i >= 0; --i)
        {
            coord_cache[i] = 0;
            if (i < dim_size - 1)
            {
                jump_forward_cache[i] = jump_forward_cache[i + 1] * reversed_shape[i + 1];
            }
            else
            {
                jump_forward_cache[i] = 1;
            }
        }

        dtype *ele_pos = transposed.m_data;

        for (lint i = 0; i < size; ++i)
        {
            *ele_pos = in.m_data[i];

            plus_pos = 0;
            while (plus_pos < dim_size)
            {
                lint increased = coord_cache[plus_pos] + 1;
                if (increased < reversed_shape[plus_pos])
                {
                    coord_cache[plus_pos] = increased;
                    ele_pos += jump_forward_cache[plus_pos];
                    // std::cout << "forward: " << jump_forward_cache[plus_pos] << '\n';
                    break;
                }
                else
                {
                    coord_cache[plus_pos] = 0;
                    ele_pos -= jump_forward_cache[plus_pos] * (reversed_shape[plus_pos] - 1);
                    // std::cout << "backward: " << jump_forward_cache[plus_pos] * (reversed_shape[plus_pos] - 1) << '\n';
                    ++plus_pos;
                }
            }
        }

        delete[]coord_cache;
        delete[]jump_forward_cache;

        return transposed;
    }

    /* Scale one dimension of the matrix with a vector of scales
    For example, if a two dimensional matrix A is like this:

    | a11, a12, a13 |
    | a21, a22, a23 |
    | a31, a32, a33 |
    | a41, a42, a43 |

    And vector B is [ b1, b2, b3 ], vector C is [ c1, c2, c3, c4 ]
    Then the result of A.scale_one_dimension(1, B) is:

    | a11 * b1, a12 * b2, a13 * b3 |
    | a21 * b1, a22 * b2, a23 * b3 |
    | a31 * b1, a32 * b2, a33 * b3 |
    | a41 * b1, a42 * b2, a43 * b3 |

    Then the result of A.scale_one_dimension(0, C) is:

    | a11 * c1, a12 * c1, a13 * c1 |
    | a21 * c2, a22 * c2, a23 * c2 |
    | a31 * c3, a32 * c3, a33 * c3 |
    | a41 * c4, a42 * c4, a43 * c4 |

    It seems this kind of calculation is needed in back propagation.
    Perhaps there is a better way to name this function.
    */
    template <typename dtype>
    TMatrix<dtype> scale_one_dimension(const TMatrix<dtype> & in, lint in_dim, const Vector & scales)
    {
        TMatrix mat{ in };
        mat.scale_one_dimension(in_dim, scales);
        return mat;
    }

    /* Overloading of output stream << operator
    The following example is output of a matrix of shape [2, 4, 3, 5]:
    [
    [
    [
    [  -0.146382    0.13453      -1.87138     0.46065      -0.214253    ]
    [  0.163712     -0.827944    0.298595     1.05547      0.0102154    ]
    [  1.17457      -0.546841    -1.04944     0.660682     -0.625276    ]
    ]
    [
    [  1.48596      -0.829081    -2.55912     -0.888707    -0.539781    ]
    [  1.01922      -0.628956    -0.482589    0.339587     -0.121306    ]
    [  2.10886      -0.371003    -0.287389    -2.30144     -1.05935     ]
    ]
    [
    [  -0.0615274   1.45502      1.35433      0.925328     -0.243275    ]
    [  1.51561      0.197497     1.00886      0.439499     0.438945     ]
    [  0.645743     -0.128149    -1.68599     1.77643      -0.613857    ]
    ]
    [
    [  0.469861     -0.582398    0.668493     -0.103692    0.149386     ]
    [  0.624049     1.53727      1.17067      1.07825      -2.05006     ]
    [  1.17196      -1.45473     0.136395     -1.11552     -1.71463     ]
    ]
    ]
    [
    [
    [  1.12422      -1.73985     -1.47975     -1.58694     1.48247      ]
    [  -0.727862    0.754843     -0.1128      0.984235     0.326633     ]
    [  -1.03745     -0.0764704   -2.08402     0.389231     0.243215     ]
    ]
    [
    [  0.455092     0.275194     2.91628      0.272422     -3.20464     ]
    [  1.86225      -2.09501     1.05544      0.310367     -0.00122802  ]
    [  0.404831     -1.08115     1.41863      -0.400148    0.926096     ]
    ]
    [
    [  -0.358203    0.126072     0.387892     -0.569566    -0.634654    ]
    [  0.882249     -0.677104    0.204175     1.35715      -2.453       ]
    [  -0.315325    -0.379922    -0.608541    1.35717      0.0195746    ]
    ]
    [
    [  1.32359      -0.0912438   -0.208138    -1.61209     0.281664     ]
    [  0.785215     -0.316253    0.353801     -0.271609    -1.77443     ]
    [  -0.0590157   -1.53723     -0.0539041   0.386642     0.129153     ]
    ]
    ]
    ]
    */
    template <typename dtype>
    std::ostream & operator << (std::ostream & os, const TMatrix<dtype> & m)
    {
        lint size = m.m_shape.size();
        size_t block_size = 0;
        for (lint i = 0; i < size; ++i)
        {
            std::ostringstream stream;
            stream << m.m_data[i];
            size_t len = stream.str().size();
            if (len > block_size)
            {
                block_size = len;
            }
        }

        block_size += 2;

        m.print(os, 0, m.m_data, block_size);

        return os;
    }
}

#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <random>
#include <sstream>
#include <cstring>
#include <functional>

template <typename dtype>
neurons::TMatrix<dtype>::TMatrix()
    : m_shape{}, m_data{ nullptr }
{}

template <typename dtype>
neurons::TMatrix<dtype>::TMatrix(const Shape & shape)
    : m_shape{ shape }
{
    if (shape.m_size < 1)
    {
        this->m_data = nullptr;
    }
    else
    {
        m_data = new dtype[this->m_shape.m_size];
    }
}

template <typename dtype>
neurons::TMatrix<dtype>::TMatrix(const Shape & shape, dtype value)
    : TMatrix{ shape }
{
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] = value;
    }
}

template <typename dtype>
neurons::TMatrix<dtype>::TMatrix(const std::vector<TMatrix>& matrices)
    : TMatrix{}
{
    lint array_size = matrices.size();

    if (array_size > 0)
    {
        Shape mat_sh = matrices[0].m_shape;
        lint mat_size = mat_sh.m_size;

        if (mat_size > 0)
        {
            for (lint i = 1; i < array_size; ++i)
            {
                if (matrices[i].m_shape != mat_sh)
                {
                    throw std::invalid_argument(
                        std::string("TMatrix::TMatrix: invalid matrix array because of different shapes of matrices"));
                }
            }

            this->m_shape = Shape{ array_size } +mat_sh;
            this->m_data = new dtype[this->m_shape.m_size];
            dtype *this_pos = this->m_data;
            dtype *that_pos;

            for (lint i = 0; i < array_size; ++i)
            {
                that_pos = matrices[i].m_data;
                for (lint j = 0; j < mat_size; ++j)
                {
                    *this_pos = *that_pos;
                    ++this_pos;
                    ++that_pos;
                }
            }
        }
    }
}

template <typename dtype>
neurons::TMatrix<dtype>::TMatrix(const TMatrix & other)
    : m_shape{ other.m_shape }
{
    lint size = m_shape.m_size;
    m_data = new dtype[size];

    std::memcpy(m_data, other.m_data, size * sizeof(dtype));
}

template <typename dtype>
neurons::TMatrix<dtype>::TMatrix(TMatrix && other)
    : m_shape{ std::move(other.m_shape) }, m_data{ other.m_data }
{
    other.m_data = nullptr;
}

template <typename dtype>
neurons::TMatrix<dtype>::TMatrix(const Vector & vec, bool transpose)
    : m_shape{ vec.m_dim, 1 }
{
    if (transpose)
    {
        m_shape.m_data[0] = 1;
        m_shape.m_data[1] = vec.m_dim;
    }

    lint size = m_shape.m_size;
    m_data = new dtype[size];

    std::memcpy(this->m_data, vec.m_data, size * sizeof(dtype));
}

template <typename dtype>
neurons::TMatrix<dtype>::~TMatrix()
{
    delete[]this->m_data;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::operator = (const TMatrix & other)
{
    delete[]this->m_data;
    this->m_shape = other.m_shape;

    lint size = this->m_shape.m_size;
    this->m_data = new dtype[size];

    std::memcpy(this->m_data, other.m_data, size * sizeof(dtype));

    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::operator = (TMatrix && other)
{
    delete[]m_data;
    this->m_shape = std::move(other.m_shape);

    this->m_data = other.m_data;
    other.m_data = nullptr;

    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::operator = (dtype scalar)
{
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] = scalar;
    }

    return *this;
}

template <typename dtype>
dtype neurons::TMatrix<dtype>::at(const Coordinate & pos) const
{
    if (pos.m_dim != this->m_shape.m_dim)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    lint index = 0;
    for (lint i = 0; i < pos.m_dim; ++i)
    {
        if (this->m_shape.m_data[i] <= pos.m_data[i])
        {
            throw std::invalid_argument(invalid_coordinate);
        }

        index *= this->m_shape.m_data[i];
        index += pos.m_data[i];
    }

    return this->m_data[index];
}

template <typename dtype>
dtype & neurons::TMatrix<dtype>::at(const Coordinate & pos)
{
    if (pos.m_dim != this->m_shape.m_dim)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    lint index = 0;
    for (lint i = 0; i < pos.m_dim; ++i)
    {
        if (this->m_shape.m_data[i] <= pos.m_data[i])
        {
            throw std::invalid_argument(invalid_coordinate);
        }

        index *= this->m_shape.m_data[i];
        index += pos.m_data[i];
    }

    return this->m_data[index];
}

template <typename dtype>
dtype neurons::TMatrix<dtype>::operator [] (const Coordinate & pos) const
{
    lint index = 0;
    for (lint i = 0; i < pos.m_dim; ++i)
    {
        index *= this->m_shape.m_data[i];
        index += pos.m_data[i];
    }

    if (index >= this->m_shape.m_size)
    {
        std::cout << pos << '\n';
        std::cout << "fuck\n";
    }
    return this->m_data[index];
}

template <typename dtype>
dtype & neurons::TMatrix<dtype>::operator [] (const Coordinate & pos)
{
    lint index = 0;
    for (lint i = 0; i < pos.m_dim; ++i)
    {
        index *= this->m_shape.m_data[i];
        index += pos.m_data[i];
    }

    return this->m_data[index];
}

template <typename dtype>
neurons::TMatrix<dtype>::operator neurons::Vector() const
{
    lint size = this->m_shape.m_size;
    Vector vec = Vector(size);
    for (lint i = 0; i < size; ++i)
    {
        vec[i] = this->m_data[i];
    }

    return vec;
}

template <typename dtype>
neurons::Vector neurons::TMatrix<dtype>::flaten() const
{
    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::operator += (const TMatrix & other)
{
    if (m_shape.m_size != other.m_shape.m_size)
    {
        throw std::invalid_argument(invalid_shape);
    }

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] += other.m_data[i];
    }

    return *this;
}


template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::operator -= (const TMatrix & other)
{
    if (m_shape.m_size != other.m_shape.m_size)
    {
        throw std::invalid_argument(invalid_shape);
    }

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] -= other.m_data[i];
    }

    return *this;
}


template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::operator += (dtype scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] += scalar;
    }

    return *this;
}


template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::operator -= (dtype scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] -= scalar;
    }

    return *this;
}


template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::operator *= (dtype scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] *= scalar;
    }

    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::operator /= (dtype scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] /= scalar;
    }

    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::gaussian_random(dtype mu, dtype sigma)
{
    std::default_random_engine generator;
    std::normal_distribution<dtype> distribution(mu, sigma);

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        this->m_data[i] = distribution(generator);
    }

    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::uniform_random(dtype min, dtype max)
{
    if (min >= max)
    {
        throw std::invalid_argument(std::string("TMatrix::uniform_random: min should be smaller than max"));
    }

    lint size = m_shape.m_size;
    dtype range = max - min;
    for (lint i = 0; i < size; ++i)
    {
        this->m_data[i] = min + (static_cast<dtype>(rand()) / RAND_MAX) * range;
    }

    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::normalize(dtype min, dtype max)
{
    if (min >= max)
    {
        throw std::invalid_argument(std::string("TMatrix::normalize: min should be smaller than max"));
    }

    lint size = m_shape.m_size;
    dtype range = max - min;

    dtype l_max = std::numeric_limits<dtype>::max() * (-1);
    dtype l_min = std::numeric_limits<dtype>::max();

    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] > l_max)
        {
            l_max = this->m_data[i];
        }

        if (this->m_data[i] < l_min)
        {
            l_min = this->m_data[i];
        }
    }

    dtype l_range = l_max - l_min;
    if (0 == l_range)
    {
        this->uniform_random(min, max);
    }
    else
    {
        for (lint i = 0; i < size; ++i)
        {
            this->m_data[i] = min + ((this->m_data[i] - l_min) / l_range) * range;
        }
    }

    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::normalize()
{
    lint size = this->m_shape.m_size;

    if (0 == size)
    {
        throw std::bad_function_call();
    }

    dtype mean = 0;
    for (lint i = 0; i < size; ++i)
    {
        mean += this->m_data[i];
    }
    mean /= size;

    dtype var = 0;
    dtype sub;
    for (lint i = 0; i < size; ++i)
    {
        sub = this->m_data[i] - mean;
        var += sub * sub;
    }
    var /= size;
    var = sqrt(var);

    if (0 == var)
    {
        this->gaussian_random(0, 1);
    }
    else
    {
        for (lint i = 0; i < size; ++i)
        {
            this->m_data[i] = (this->m_data[i] - mean) / var;
        }
    }

    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::reshape(const Shape & shape)
{
    if (shape.m_size != this->m_shape.m_size)
    {
        throw std::invalid_argument(
            std::string("TMatrix::reshape: the new shape should be compatible with number of elements in this matrix"));
    }

    this->m_shape = shape;

    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::left_extend_shape()
{
    this->m_shape.left_extend();
    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::right_extend_shape()
{
    this->m_shape.right_extend();
    return *this;
}

template <typename dtype>
neurons::TMatrix<dtype> neurons::TMatrix<dtype>::get_left_extended(lint duplicate) const
{
    if (duplicate < 1)
    {
        throw std::invalid_argument(std::string("neurons::TMatrix::get_left_extended: duplicate should be a positive value"));
    }

    if (this->m_shape.m_size < 1)
    {
        throw std::bad_function_call();
    }

    TMatrix mat{ Shape{ duplicate } +this->m_shape };
    lint size = this->m_shape.m_size;
    dtype *mat_start = mat.m_data;

    for (lint j = 0; j < duplicate; ++j)
    {
        std::memcpy(mat_start, this->m_data, size * sizeof(dtype));
        mat_start += size;
    }

    return mat;
}

template <typename dtype>
neurons::TMatrix<dtype> & neurons::TMatrix<dtype>::scale_one_dimension(lint dim, const Vector & scales)
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("neurons::TMatrix::scale_one_dimension: dimension index out of range.");
    }

    if (this->m_shape.m_data[dim] != scales.dim())
    {
        throw std::invalid_argument(
            "neurons::TMatrix::scale_one_dimension: size of the vector is not compatible with size of this matrix dimension.");
    }

    lint size_dims_before = this->m_shape.sub_shape(0, dim - 1).m_size;
    lint size_dims_after = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1).m_size;
    lint size_dims_and_after = this->m_shape.sub_shape(dim, this->m_shape.m_dim - 1).m_size;

    if (0 == size_dims_before)
    {
        size_dims_before = 1;
    }

    if (0 == size_dims_after)
    {
        size_dims_after = 1;
    }

    dtype *start = this->m_data;

    for (lint i = 0; i < scales.dim(); ++i)
    {
        dtype scale = scales[i];
        dtype *l_start = start;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            dtype *ele = l_start;
            for (lint k = 0; k < size_dims_after; ++k)
            {
                *ele *= scale;
                ++ele;
            }
            l_start += size_dims_and_after;
        }

        start += size_dims_after;
    }

    return *this;
}

template <typename dtype>
neurons::Coordinate neurons::TMatrix<dtype>::argmax() const
{
    dtype max = std::numeric_limits<dtype>::max() * (-1);
    lint argmax_index = 0;
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] > max)
        {
            max = this->m_data[i];
            argmax_index = i;
        }
    }

    Coordinate coord{ this->m_shape };
    lint marker = argmax_index;
    for (lint i = this->m_shape.m_dim - 1; i >= 0; --i)
    {
        coord.m_data[i] = marker % this->m_shape.m_data[i];
        marker /= this->m_shape.m_data[i];
    }

    return coord;
}

template <typename dtype>
dtype neurons::TMatrix<dtype>::max() const
{
    dtype max = std::numeric_limits<dtype>::max() * (-1);
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] > max)
        {
            max = this->m_data[i];
        }
    }

    return max;
}

template <typename dtype>
neurons::Coordinate neurons::TMatrix<dtype>::argmin() const
{
    dtype min = std::numeric_limits<dtype>::max() * (-1);
    lint argmin_index = 0;
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] < min)
        {
            min = this->m_data[i];
            argmin_index = i;
        }
    }

    Coordinate coord{ this->m_shape };
    lint marker = argmin_index;
    for (lint i = this->m_shape.m_dim - 1; i >= 0; --i)
    {
        coord.m_data[i] = marker % this->m_shape.m_data[i];
        marker /= this->m_shape.m_data[i];
    }

    return coord;
}

template <typename dtype>
dtype neurons::TMatrix<dtype>::min() const
{
    dtype min = std::numeric_limits<dtype>::max();
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] < min)
        {
            min = this->m_data[i];
        }
    }

    return min;
}

template <typename dtype>
dtype neurons::TMatrix<dtype>::mean() const
{
    lint size = this->m_shape.m_size;
    dtype sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        sum += this->m_data[i];
    }

    return sum / size;
}

template <typename dtype>
dtype neurons::TMatrix<dtype>::variance() const
{
    dtype mean = this->mean();

    lint size = this->m_shape.m_size;
    dtype sum = 0;
    dtype sub;
    for (lint i = 0; i < size; ++i)
    {
        sub = this->m_data[i] - mean;
        sum += sub * sub;
    }

    return sum / size;
}

template <typename dtype>
std::vector<neurons::TMatrix<dtype>> neurons::TMatrix<dtype>::collapse(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("neurons::TMatrix::scale_one_dimension: dimension index out of range.");
    }

    Shape sh_before = this->m_shape.sub_shape(0, dim - 1);
    Shape sh_after = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1);
    Shape sh_collapsed = sh_before + sh_after;

    lint size_dims_before = sh_before.m_size;
    lint size_dims_after = sh_after.m_size;
    lint size_dims_and_after = this->m_shape.sub_shape(dim, this->m_shape.m_dim - 1).m_size;

    if (0 == size_dims_before)
    {
        size_dims_before = 1;
    }

    if (0 == size_dims_after)
    {
        size_dims_after = 1;
    }

    std::vector<TMatrix> all_collapsed;
    dtype *start = this->m_data;

    for (lint i = 0; i < this->m_shape.m_data[dim]; ++i)
    {
        TMatrix collapsed{ sh_collapsed };
        dtype *l_start = start;
        dtype *clps_ele = collapsed.m_data;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            dtype *ele = l_start;
            for (lint k = 0; k < size_dims_after; ++k)
            {
                *clps_ele = *ele;
                ++clps_ele;
                ++ele;
            }
            l_start += size_dims_and_after;
        }

        all_collapsed.push_back(collapsed);
        start += size_dims_after;
    }

    return all_collapsed;
}

template <typename dtype>
neurons::TMatrix<dtype> neurons::TMatrix<dtype>::fuse(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("neurons::TMatrix::fuse: dimension index out of range.");
    }

    Shape sh_before = this->m_shape.sub_shape(0, dim - 1);
    Shape sh_after = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1);
    Shape sh_collapsed = sh_before + sh_after;

    lint size_dims_before = sh_before.m_size;
    lint size_dims_after = sh_after.m_size;
    lint size_dims_and_after = this->m_shape.sub_shape(dim, this->m_shape.m_dim - 1).m_size;

    if (0 == size_dims_before)
    {
        size_dims_before = 1;
    }

    if (0 == size_dims_after)
    {
        size_dims_after = 1;
    }

    TMatrix fused{ sh_collapsed, 0 };
    dtype *start = this->m_data;

    for (lint i = 0; i < this->m_shape.m_data[dim]; ++i)
    {
        TMatrix collapsed{ sh_collapsed };
        dtype *l_start = start;
        dtype *clps_ele = collapsed.m_data;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            dtype *ele = l_start;
            for (lint k = 0; k < size_dims_after; ++k)
            {
                *clps_ele = *ele;
                ++clps_ele;
                ++ele;
            }
            l_start += size_dims_and_after;
        }

        fused += collapsed;
        start += size_dims_after;
    }

    return fused;
}

template <typename dtype>
neurons::TMatrix<dtype> neurons::TMatrix<dtype>::reduce_mean(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("neurons::TMatrix::fuse: dimension index out of range.");
    }

    lint dim_size = this->m_shape.m_data[dim];
    return this->fuse(dim) / static_cast<dtype>(dim_size);
}

template <typename dtype>
dtype neurons::TMatrix<dtype>::euclidean_norm() const
{
    // Calculate the norm
    dtype norm = 0;
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        norm += this->m_data[i] * this->m_data[i];
    }

    return norm;
}

template <typename dtype>
neurons::Shape neurons::TMatrix<dtype>::shape() const
{
    return m_shape;
}

template <typename dtype>
void neurons::TMatrix<dtype>::print(std::ostream & os, lint dim_index, const dtype *start, size_t block_size) const
{
    if (dim_index >= this->m_shape.m_dim)
    {
        return;
    }

    for (lint i = 0; i < dim_index; ++i)
    {
        os << "    ";
    }

    if (dim_index == this->m_shape.dim() - 1)
    {
        os << "[  ";
    }
    else
    {
        os << "[\n";
    }

    lint dim_size = this->m_shape[dim_index];
    lint jump_dist = 1;

    for (lint i = this->m_shape.dim() - 1; i > dim_index; --i)
    {
        jump_dist *= this->m_shape[i];
    }

    for (lint i = 0; i < dim_size; i++)
    {
        if (dim_index == this->m_shape.dim() - 1)
        {
            dtype val = *(start + i);
            std::ostringstream stream;
            stream << val;
            std::string str = stream.str();
            os << val;
            for (size_t j = 0; j < block_size - str.size(); ++j)
            {
                os << ' ';
            }
        }
        else
        {
            this->print(os, dim_index + 1, start + i * jump_dist, block_size);
        }
    }

    if (dim_index == this->m_shape.dim() - 1)
    {
        os << "]\n";
    }
    else
    {
        for (lint i = 0; i < dim_index; ++i)
        {
            os << "    ";
        }
        os << "]\n";
    }
}







