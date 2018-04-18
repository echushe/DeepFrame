#pragma once
#include "Shape.h"
#include <memory>

namespace neurons
{
    /*
    This class helps locate a place in a matrix.
    For example, coordinate [ 0, 2 ] is one of the places in a matrix of shape [ 3, 4 ]
    */
    class Coordinate
    {
        friend class Matrix;

        friend class TMatrix<double>;
        friend class TMatrix<float>;
        friend class TMatrix<short>;
        friend class TMatrix<int>;
        friend class TMatrix<lint>;

    private:
        // Number of dimensions
        lint m_dim;
        // Detailed data of this coordinate
        lint *m_data;

        // The shape used to restrict behavior of the coordinate
        // The coordinate can only change within the range defined by the shape
        std::unique_ptr<Shape> m_shape;

    public:
        // The coordinate can be created by a list of numbers.
        // These numbers should be no less than 0.
        Coordinate(std::initializer_list<lint> list);

        // Create a coordinate by a list of numbers.
        // But these numbers should agree with the shape.
        // For example, [ 0, 90 ] is not a location in shape [ 30, 20, 75 ]
        Coordinate(std::initializer_list<lint> list, const Shape & shape);

        // Create a coordinate with a shape
        // In this circumstance, the coordinate will be initialized by zeros
        Coordinate(const Shape & shape);

        // Copy constructor
        Coordinate(const Coordinate & other);
        // Move constructor
        Coordinate(Coordinate && other);
        // Destructor
        ~Coordinate();
        // Copy assignment
        Coordinate & operator = (const Coordinate & other);
        // Move assignment
        Coordinate & operator = (Coordinate && other);

    public:
        // Access a certain dimension of this coordinate
        lint operator [] (lint index) const;
        // Access a certain dimansion of this coordinate and update it
        lint & operator [] (lint index);
        // Plus the coordinate to traverse a matrix
        Coordinate & operator ++ ();
        // Plus the coordinate to 
        Coordinate operator ++ (int);

        // Get amount of dimensions of this coordinate
        lint dim() const;

        // Reverse the coordinate
        // For example, reverse of [ 7, 5, 3, 0, 44 ] is [ 44, 0, 3, 5, 7 ]
        Coordinate & reverse();

        Coordinate & transposed_plus();

        friend bool operator == (const Coordinate &left, const Coordinate &right);
        friend bool operator != (const Coordinate &left, const Coordinate &right);
        friend std::ostream & operator << (std::ostream & os, const Coordinate & co);
    };

    // coordinate_a == coordinate_b
    bool operator == (const Coordinate &left, const Coordinate &right);

    // coordinate_a != coordinate_b
    bool operator != (const Coordinate &left, const Coordinate &right);

    std::ostream & operator << (std::ostream & os, const Coordinate & co);

    // Get reverse of the coordinate without changing the original one
    Coordinate reverse(const Coordinate & sh);
}
