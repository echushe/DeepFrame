/********************************************************************

Programmed by Chunnan Sheng

*********************************************************************/
#pragma once
#include <list>

// All integers are 64bit wide here
typedef long long lint;

namespace neurons
{
    class Matrix;

    template <typename dtype>
    class TMatrix;

    /*
    class Shape represents shape of a matrix.
    For example, [ 4, 5, 6, 8 ] is shape of a four dimensional matrix, sizes of its
    dimensions are 4, 5, 6, 8 respectively.
    */
    class Shape
    {
        friend class Coordinate;
        friend class Matrix;

        friend class TMatrix<double>;
        friend class TMatrix<float>;
        friend class TMatrix<short>;
        friend class TMatrix<int>;
        friend class TMatrix<lint>;

    private:
        // How many dimensions this shape has
        lint m_dim;
        // Amount of elements this shape has
        lint m_size;
        // Detailed information of this shape
        lint *m_data;

    public:
        // A shape can be intialized by a list of dimension sizes
        Shape(std::initializer_list<lint> list);
        // Copy constructor
        Shape(const Shape & other);
        // Move constructor
        Shape(Shape && other);
        // Destructor
        ~Shape();
        // Copy assignment
        Shape & operator = (const Shape & other);
        // Move assignment
        Shape & operator = (Shape && other);
        
        Shape();

    public:
        // [] operator is to access a dimension
        lint operator [] (lint index) const;

        // [] operator to access a dimension and modify it
        // lint & operator [] (lint index);

        // Get amount of elements of this shape
        lint size() const;

        // Get amount of dimensions of this shape
        lint dim() const;

        // Add one dimension to left side of this shape.
        // Size of this shape won't change after this operation.
        // For example, shape [ 100, 120 ] will be changed to [ 1, 100, 120 ]
        Shape & left_extend();

        // Add one dimension to right side of this shape
        // Size of this shape won't change after this operation.
        // For example, shape [ 70, 80 ] will be changed to [ 70, 80, 1 ]
        Shape & right_extend();

        // Reverse the shape.
        // For example, [ 130, 180, 70, 6 ] will be changed to [ 6, 70, 180, 130 ]
        Shape & reverse();

        Shape sub_shape(lint dim_first, lint dim_last) const;

        friend bool operator == (const Shape &left, const Shape &right);
        friend bool operator != (const Shape &left, const Shape &right);
        friend Shape operator + (const Shape &left, const Shape &right);
    };


    // shape_a == shape_b
    bool operator == (const Shape &left, const Shape &right);

    // shape_a != shape_b
    bool operator != (const Shape &left, const Shape &right);

    Shape operator + (const Shape &left, const Shape &right);

    // This function returns the reversed shape without changing the original shape
    Shape reverse(const Shape & sh);
}
