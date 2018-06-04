/*********************************************
*
*          COMP6771 Assignment 2
*             Chunnan Sheng
*               z5100764
*
*********************************************/

#pragma once
#include <vector>
#include <list>
#include <ostream>
#include <string>
#include <limits>

// All integers are 64bit wide here
typedef long long lint;

// Definition of namespace evec
namespace neurons
{
    template <typename dtype>
    class TMatrix;

    // Exception descriptions
    // const std::string zero_dim("Dimension of zero size is invalid for Vector!");
    const std::string out_range("Vector: Index out of range!");
    const std::string diff_dim("Vector: Numbers of dimensons should be the same!");
    // const std::string zero_div("Vector: Divisor should not be zero!");
    const std::string zero_norm("Vector: Unit vector of a vector whose norm is zero cannot be acquired!");
    // const std::string assn_self("Vector: Cannot copy, assign or move an instance to itself!");

    const double inf = std::numeric_limits<double>::infinity();
    const double minus_inf = std::numeric_limits<double>::infinity() * (-1);

    // Definition of class Vector
    class Vector
    {
        friend class TMatrix<double>;
        friend class TMatrix<float>;
        friend class TMatrix<short>;
        friend class TMatrix<int>;
        friend class TMatrix<lint>;

        friend class Linear_Regression;
    private:
        // long long (64bit integer) as amount of dimensions
        lint m_dim;
        // Pointer to the dynamic norm value of this vector
        // default value of this pointer is nullptr until there is once calculation
        // of norm of this vector.
        // The double instance of this pointer will be deleted and this pointer will 
        // be changed back to nullptr if there is a modification of this vector.
        mutable double *m_norm;
        // This pointer will point to to the dynamic memory of vector dimensions
        double *m_data;

    public:
        // A default constructor that takes number of dimensions
        Vector(lint dim = 1);
        
        // A constructor that takes number of dimensions and an arbitrary value for all dimensions
        Vector(lint dim, double m);

        // A constructor that takes an iterator of a vector<double>
        Vector(const std::vector<double>::iterator &begin, const std::vector<double>::iterator &end);

        // A constructor that takes a const iterator of a vector<double>
        Vector(const std::vector<double>::const_iterator &begin, const std::vector<double>::const_iterator &end);

        // A constructor that takes an iterator of a list<double>
        Vector(const std::list<double>::iterator &begin, const std::list<double>::iterator &end);

        // A constructor that takes a const iterator of a list<double>
        Vector(const std::list<double>::const_iterator &begin, const std::list<double>::const_iterator &end);

        // A construtor that takes a list of double values
        Vector(std::initializer_list<double> list);

        //Copy constructor
        Vector(const Vector &other);

        //Move constructor
        Vector(Vector &&other);

        //Destructor
        ~Vector();

    public:
        //Copy assignment
        Vector & operator = (const Vector &other);

        //Move assignment
        Vector & operator = (Vector &&other);

        // A [] operator that would change the value of a dimension
        double & operator [] (lint index);

        // A [] operator accessing value of a dimension
        double operator [] (lint index) const;

        // Example: b = -a
        Vector operator - () const;

        // Adding right vectors to itself
        Vector & operator += (const Vector &right);

        // Subtracting right vectors from itself
        Vector & operator -= (const Vector &right);

        // Scalar multiplication
        Vector & operator *= (double scalar);

        // Scalar division
        Vector & operator /= (double scalar);

        // Implicitly convert into a list<double> instance
        operator std::list<double>() const;

        // Implicitly convert into a vector<double> instance
        operator std::vector<double>() const;

    public:

        lint getNumDimensions() const;

        double get(lint index) const;

        // This function caches calculated norm value.
        // The cache will be removed if the Vector is updated.
        double getEuclideanNorm() const;

        // Create and return a unit vector
        Vector createUnitVector() const;

        //
        lint dim() const
        {
            return m_dim;
        }

    private:
        void destroyMe();
        void destroyMeWithoutMemoryRelease();
    };

    /************************************************************
          I have tried my best not to use friend functions
    *************************************************************/

    // Overloading of a == b
    bool operator == (const Vector &left, const Vector &right);

    // Overloading of a != b
    bool operator != (const Vector &left, const Vector &right);

    // Overloading of a + b
    Vector operator + (const Vector &left, const Vector &right);

    // Overloading of a - b
    Vector operator - (const Vector &left, const Vector &right);

    // Overloading of a * b (dot product)
    double operator * (const Vector &left, const Vector &right);

    // Overloading of a * b, b is a double
    Vector operator * (const Vector &left, double scalar);

    // Overloading of a * b, a is a double
    Vector operator * (double scalar, const Vector &right);

    // Overloading of a / b, b is a double
    Vector operator / (const Vector &left, double scalar);

    // Overloading of output stream << operator
    std::ostream& operator<<(std::ostream& os, const Vector &v);
}

