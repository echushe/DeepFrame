/********************************************************************

    Programmed by Chunnan Sheng

*********************************************************************/
#pragma once

#include "Vector.h"
#include "Shape.h"
#include "Coordinate.h"
#include <iostream>

/*
 This file defines all calculations of multi-dimensional arrays (matrices)
 */
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
    class Matrix
    {
        // These friend classes are all functions who may operate on matrices
        friend class Linear;
        friend class Sigmoid;
        friend class Tanh;
        friend class Relu;
        friend class Softmax;
        friend class HalfSquareError;
        friend class Sigmoid_CrossEntropy;
        friend class Softmax_CrossEntropy;
        friend class Conv_1d;
        friend class Conv_2d;
        friend class Pooling_2d;
        friend class MaxPooling_2d;
        friend class EM_1d;
        friend class Linear_Regression;
    private:
        // Shape of this matrix
        Shape m_shape;
        // All the elements of this matrix
        double *m_data;

    public:
        // The default constructor.
        // The matrix created by default constructor is not usable until
        // it is assigned by a usable matrix
        Matrix();

        // Create a matrix of a certain shape. But none of its elements are initialized
        Matrix(const Shape & shape);

        // Create a matrix of a certain shape. All of its elements are initialized by a value
        Matrix(const Shape & shape, double value);

        // Combine an array if matrices into one matrix
        explicit Matrix(const std::vector<Matrix> & matrices);
        
        // Copy constructor
        Matrix(const Matrix & other);

        // Move constructor
        Matrix(Matrix && other);

        // Create a matrix from a vector
        Matrix(const Vector & vec, bool transpose = false);

        ~Matrix();

    public:
        // Copy assignment
        Matrix & operator = (const Matrix & other);
        
        // Move assignment
        Matrix & operator = (Matrix && other);

        // Assign a scalar value to all matrix elements
        Matrix & operator = (double scalar);

        // Get an element of a certain position
        double at (const Coordinate & pos) const;

        // Get an element of a certain position. This element can be updated
        double & at (const Coordinate & pos);

        // Get an element of a certain position
        double operator [] (const Coordinate & pos) const;

        // Get an element of a certain position. This element can be updated
        double & operator [] (const Coordinate & pos);

        // An implicite way to get a flatened matrix (vector)
        operator Vector() const;

        // Get a flatened matrix (vector)
        Vector flaten() const;
        
        Matrix & operator += (const Matrix & other);

        Matrix & operator -= (const Matrix & other);

        Matrix & operator += (double scalar);

        Matrix & operator -= (double scalar);

        Matrix & operator *= (double scalar);

        Matrix & operator /= (double scalar);

        // Randomize all elements of this matrix. Distribution of the elements complies with
        // Gaussian distribution (normal distribution).
        // double mu: mean of these elements
        // double sigma: sqrt(variance) of these elements
        Matrix & gaussian_random(double mu, double sigma);

        // Randomize all elements of this matrix. Distribution of the elements complies with
        // uniform distribution.
        Matrix & uniform_random(double min, double max);

        // Normalize all elements of this matrix to the range of [min, max]
        Matrix & normalize(double min, double max);

        // Normalize all elements of this matrix to mean == 0 and variance == 1
        Matrix & normalize();

        // Change shape of this matrix.
        // For example, we can change [20, 30] to [2, 10, 3, 5, 2], or
        // change [4, 5, 6] to [3, 5, 8].
        // However, size of this matrix will not change, which means 20 * 30 == 2 * 10 * 3 * 5 * 2,
        // or 4 * 5 * 6 == 3 * 5 * 8.
        // Order of all elements in this matrix will not change either
        Matrix & reshape(const Shape & shape);

        // Extend one dimension of the matrix.
        // For example, [30, 50] is extended to [1, 30, 50].
        // This is a special case of reshape.
        Matrix & left_extend_shape();

        // Extend one dimension of the matrix.
        // For example, [30, 50] is extended to [30, 50, 1].
        // This is a special case of reshape.
        Matrix & right_extend_shape();

        // Get duplicates of left extended version of the matrix.
        // For example, [35, 45] can be extend to [12, 35, 45] in which
        // there are exactly 12 copies of [35, 45]
        Matrix get_left_extended(lint duplicate) const;

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
        Matrix & scale_one_dimension(lint dim, const Vector & scales);

        // Get coordinate of the largest element
        Coordinate argmax() const;

        // Get value of the largest element
        double max() const;

        // Get coordinate of the lowest element
        Coordinate argmin() const;

        // Get value of the lowest element
        double min() const;

        // Get mean of all the elements
        double mean() const;

        // Get variance of all the elements of this matrix
        double variance() const;

        // Collapse a certain dimension of a matrix into a vector of matrices
        // For example, a matrix of shape [4, 6, 5, 7], if we collapse it with argument dim = 1,
        // it will be turned into 6 matrices of shape [4, 5, 7]
        std::vector<Matrix> collapse(lint dim) const;

        // Collapse a certain dimension of a matrix, and merge all matrices into one matrix
        // For example, a matrix of shape [4, 6, 5, 7], if we fuse it with argument dim = 1,
        // it will be turned into a sum of 6 matrices of shape [4, 5, 7]
        Matrix fuse(lint dim) const;

        // Collapse a certain dimension of a matrix, and get a mean of all the matrices
        // For example, a matrix of shape [4, 6, 5, 7], if we get its reduce_mean with argument dim = 1,
        // it will be turned into a mean of 6 matrices of shape [4, 5, 7]
        Matrix reduce_mean(lint dim) const;


        double euclidean_norm() const;


    public:
        // Get shape of the matrix
        Shape shape() const;

        friend bool operator == (const Matrix &left, const Matrix &right);

        friend bool operator != (const Matrix &left, const Matrix &right);

        friend Matrix operator + (const Matrix &left, const Matrix &right);

        friend Matrix operator - (const Matrix &left, const Matrix &right);

        friend Matrix matrix_multiply(const Matrix & left, const Matrix & right, lint l_dims_merge, lint r_dims_merge);

        friend Matrix matrix_multiply(const Matrix & left, const Matrix & right);

        friend Matrix multiply(const Matrix & left, const Matrix & right);

        friend double dot_product(const Matrix & left, const Matrix & right);

        friend Matrix operator * (const Matrix &left, const Matrix &right);

        friend Matrix operator * (double scalar, const Matrix &right);

        friend Matrix operator * (const Matrix &left, double scalar);

        friend Matrix operator / (const Matrix &left, double scalar);

        friend Matrix transpose(const Matrix & in);

        friend std::ostream & operator << (std::ostream & os, const Matrix & m);

    private:
        void print(std::ostream & os, lint dim_index, const double *start, size_t block_size) const;
    };

    // Overloading of a == b
    bool operator == (const Matrix &left, const Matrix &right);

    // Overloading of a != b
    bool operator != (const Matrix &left, const Matrix &right);

    // Overloading of a + b
    Matrix operator + (const Matrix &left, const Matrix &right);

    // Overloading of a - b
    Matrix operator - (const Matrix &left, const Matrix &right);

    // Matrix multiplication
    // Which dimensions should be merged together is manually defined here.
    // For example, two matrices: [7, 3, 2, 5, 6] and [6, 5, 6, 4], if l_dims_merge == 2 and r_dims_merge == 2,
    // the shape of result will be [7, 3, 2, 6, 4]. However, if l_dims_merge == 4 and r_dims_merge == 3,
    // the shape of output will be [7, 4].
    // Note: Total sizes (number of elements) of dimensions to merge should be equal betweem the left and right.
    Matrix matrix_multiply(const Matrix & left, const Matrix & right, lint l_dims_merge, lint r_dims_merge);

    // Matrix multiplication
    // This function can automatically figure out which dimensions should be merged together.
    // For example, two matrices: [7, 3, 3, 5, 4] and [10, 6, 6, 4]
    // [3, 5, 4] of the left matrix and [10, 6] of the right matrix will be merged.
    // then, the shape of result will be [7, 3, 6, 4].
    // This function will throw out an exception if no appropriate dimensions to merge can be found.
    Matrix matrix_multiply(const Matrix & left, const Matrix & right);

    // Multiplication element by element.
    // The two matrices should have the same shape.
    Matrix multiply(const Matrix & left, const Matrix & right);

    // Dot product of two matrices
    // The two matrices should have the same amount of elements
    double dot_product(const Matrix & left, const Matrix & right);

    // Matrix multiplication
    // This function can automatically figure out which dimensions should be merged together.
    // For example, two matrices: [7, 3, 3, 5, 4] and [10, 6, 6, 4]
    // [3, 5, 4] of the left matrix and [10, 6] of the right matrix will be merged.
    // then, the shape of result will be [7, 3, 6, 4].
    // This function will throw out an exception if no appropriate dimensions to merge can be found.
    Matrix operator * (const Matrix &left, const Matrix &right);

    // Overloading of a * b, b is a scalar
    Matrix operator * (const Matrix &left, double scalar);

    // Overloading of a * b, a is a scalar
    Matrix operator * (double scalar, const Matrix &right);

    // Overloading of a / b, b is a scalar
    Matrix operator / (const Matrix &left, double scalar);

    // Get a transposed matrix of a matrix
    Matrix transpose(const Matrix & in);

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
    Matrix scale_one_dimension(const Matrix & in, lint in_dim, const Vector & scales);

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
    std::ostream & operator << (std::ostream & os, const Matrix & m);
}

