/*********************************************
*
*          COMP6771 Assignment 2
*             Chunnan Sheng
*               z5100764
*
*********************************************/

#include "Vector.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cstring>

neurons::Vector::Vector(lint dim)
    : Vector(dim, 0.0)
{
}

neurons::Vector::Vector(lint dim, double m)
    : m_dim(dim), m_norm(nullptr), m_data(nullptr)
{
    // Zero dimension is allowed here
    if (this->m_dim <= 0)
    {
        this->m_dim = 0;
        return;
    }

    this->m_data = new double[this->m_dim];
    for (lint i = 0; i < this->m_dim; i++)
    {
        this->m_data[i] = m;
    }
}

neurons::Vector::Vector(
    const std::vector<double>::iterator &begin, const std::vector<double>::iterator &end)
    : m_dim(end - begin), m_norm(nullptr), m_data(nullptr)
{
    // Zero dimension is allowed here
    if (0 == this->m_dim)
    {
        return;
    }

    this->m_data = new double[this->m_dim];
    
    std::copy(begin, end, this->m_data);
}

neurons::Vector::Vector(
    const std::vector<double>::const_iterator &begin, const std::vector<double>::const_iterator &end)
    : m_dim(end - begin), m_norm(nullptr), m_data(nullptr)
{
    // Zero dimension is allowed here
    if (0 == this->m_dim)
    {
        return;
    }

    this->m_data = new double[this->m_dim];

    std::copy(begin, end, this->m_data);
}

neurons::Vector::Vector(
    const std::list<double>::iterator &begin, const std::list<double>::iterator &end)
    : m_dim(0), m_norm(nullptr), m_data(nullptr)
{
    // We do not know how many elements are there in the list
    // So that we should traverse the list to figure out
    // length of this list
    std::list<double>::iterator it = begin;
    for (; it != end; it++)
    {
        m_dim++;
    }

    // Zero dimension is allowed here
    if (0 == this->m_dim)
    {
        return;
    }

    this->m_data = new double[this->m_dim];

    std::copy(begin, end, this->m_data);
}

neurons::Vector::Vector(
    const std::list<double>::const_iterator &begin, const std::list<double>::const_iterator &end)
    : m_dim(0), m_norm(nullptr), m_data(nullptr)
{
    // We do not know how many elements are there in the list
    // So that we should traverse the list to figure out
    // length of this list
    std::list<double>::const_iterator it = begin;
    for (; it != end; it++)
    {
        m_dim++;
    }

    // Zero dimension is allowed here
    if (0 == this->m_dim)
    {
        return;
    }

    this->m_data = new double[this->m_dim];

    std::copy(begin, end, this->m_data);
}

neurons::Vector::Vector(std::initializer_list<double> list)
    : m_dim(list.end() - list.begin()), m_norm(nullptr), m_data(nullptr)
{
    // Zero dimension is allowed here
    if (0 == this->m_dim)
    {
        return;
    }

    this->m_data = new double[this->m_dim];

    std::copy(list.begin(), list.end(), this->m_data);
}

// All objects associated with the original Vector should be
// Copied to the new one
neurons::Vector::Vector(const Vector & other)
    : m_dim(other.m_dim), m_norm(nullptr), m_data(nullptr)
{
    // Zero dimension is allowed here
    if (0 == this->m_dim)
    {
        return;
    }

    if (other.m_norm)
    {
        this->m_norm = new double(*(other.m_norm));
    }

    this->m_data = new double[this->m_dim];

    std::memcpy(this->m_data, other.m_data, this->m_dim * sizeof(double));
}

// The move constructor just borrow the object instances from the original Vector
neurons::Vector::Vector(Vector && other)
    : m_dim(other.m_dim), m_norm(other.m_norm), m_data(other.m_data)
{
    // Content of the original Vector is cleared
    // It is not allowed that two vector instances share the
    // same memory
    other.destroyMeWithoutMemoryRelease();
}

neurons::Vector::~Vector()
{
    this->destroyMe();
}

neurons::Vector & neurons::Vector::operator = (const Vector & other)
{
    // Skip copying, moving or assigning to itself
    if (&other == this)
    {
        return *this;
    }

    // std::cout << "assignment!!!!!!!" << std::endl;

    // All content of myself should be destroyed
    // before the new assignment
    this->destroyMe();

    // Zero dimension is allowed here
    if (0 == other.m_dim)
    {
        return *this;
    }

    this->m_dim = other.m_dim;

    if (other.m_norm)
    {
        this->m_norm = new double(*(other.m_norm));
    }

    this->m_data = new double[this->m_dim];
    std::memcpy(this->m_data, other.m_data, this->m_dim * sizeof(double));

    return *this;
}


neurons::Vector & neurons::Vector::operator = (Vector && other)
{
    // Skip copying, moving or assigning to itself
    if (&other == this)
    {
        return *this;
    }

    // std::cout << "move!!!!!!!" << std::endl;
    // All content of myself should be destroyed
    // before the new assignment
    this->destroyMe();

    this->m_dim = other.m_dim;
    this->m_norm = other.m_norm;
    this->m_data = other.m_data;

    other.destroyMeWithoutMemoryRelease();

    return *this;
}


double & neurons::Vector::operator[](lint index)
{
    if (index < 0 || index >= this->m_dim)
    {
        throw std::out_of_range(out_range);
    }

    // Value of this dimension may be modified
    // then the norm would be affected.
    // Therefore the norm should be re-calculated next time.
    if (this->m_norm)
    {
        delete this->m_norm;
        this->m_norm = nullptr;
    }

    return this->m_data[index];
}

double neurons::Vector::operator[](lint index) const
{
    if (index < 0 || index >= this->m_dim)
    {
        throw std::out_of_range(out_range);
    }

    return this->m_data[index];
}

neurons::Vector neurons::Vector::operator-() const
{
    return *this * (-1);
}

neurons::Vector & neurons::Vector::operator+=(const Vector & right)
{
    if (right.m_dim != this->m_dim)
    {
        throw std::invalid_argument(diff_dim);
    }

    // The norm is affected so it should be
    // re-calculated next time
    if (this->m_norm)
    {
        delete this->m_norm;
        this->m_norm = nullptr;
    }

    for (lint i = 0; i < this->m_dim; i++)
    {
        this->m_data[i] += right.m_data[i];
    }

    return *this;
}

neurons::Vector & neurons::Vector::operator-=(const Vector & right)
{
    if (right.m_dim != this->m_dim)
    {
        throw std::invalid_argument(diff_dim);
    }

    // The norm is affected so it should be
    // re-calculated next time
    if (this->m_norm)
    {
        delete this->m_norm;
        this->m_norm = nullptr;
    }

    for (lint i = 0; i < this->m_dim; i++)
    {
        this->m_data[i] -= right.m_data[i];
    }

    return *this;
}

neurons::Vector & neurons::Vector::operator*=(double scalar)
{
    // The norm is affected so it should be
    // re-calculated next time
    if (this->m_norm)
    {
        delete this->m_norm;
        this->m_norm = nullptr;
    }

    for (lint i = 0; i < this->m_dim; i++)
    {
        this->m_data[i] *= scalar;
    }

    return *this;
}

neurons::Vector & neurons::Vector::operator/=(double scalar)
{
    // The norm is affected so it should be
    // re-calculated next time
    if (this->m_norm)
    {
        delete this->m_norm;
        this->m_norm = nullptr;
    }

    for (lint i = 0; i < this->m_dim; i++)
    {
        this->m_data[i] /= scalar;
    }

    return *this;
}

neurons::Vector::operator std::list<double>() const
{
    std::list<double> the_list;

    for (lint i = 0; i < this->m_dim; i++)
    {
        the_list.push_back(this->m_data[i]);
    }

    return the_list;
}

neurons::Vector::operator std::vector<double>() const
{
    std::vector<double> the_vector;

    for (lint i = 0; i < this->m_dim; i++)
    {
        the_vector.push_back(this->m_data[i]);
    }

    return the_vector;
}

lint neurons::Vector::getNumDimensions() const
{
    return this->m_dim;
}

double neurons::Vector::get(lint index) const
{
    if (index < 0 || index >= this->m_dim)
    {
        throw std::out_of_range(out_range);
    }

    return this->m_data[index];
}

double neurons::Vector::getEuclideanNorm() const
{
    // Get the norm directly if there is one
    if (this->m_norm)
    {
        return *(this->m_norm);
    }

    // Calculate the norm if there is not one
    double norm = 0;
    for (lint i = 0; i < this->m_dim; i++)
    {
        norm += this->m_data[i] * this->m_data[i];
    }

    norm = sqrt(norm);
    // Cache the new norm value
    this->m_norm = new double(norm);

    return norm;
}


neurons::Vector neurons::Vector::createUnitVector() const
{
    double norm = this->getEuclideanNorm();

    // A vector of zero norm cannot be converted to a unit vector
    if (0.0 == norm)
    {
        throw std::invalid_argument(zero_norm);
    }

    // Copy the original vector to a new one
    Vector unit_vector(*this);

    // Convert the new vector to a unit vector
    for (lint i = 0; i < this->m_dim; i++)
    {
        unit_vector.m_data[i] /= norm;
    }

    if (unit_vector.m_norm)
    {
        *(unit_vector.m_norm) = 1.0;
    }

    return unit_vector;
}

void neurons::Vector::destroyMe()
{
    // std::cout << "destroy me start" << std::endl;
    // std::cout << *this << std::endl;
    if (this->m_norm)
    {
        delete this->m_norm;
        this->m_norm = nullptr;
    }

    // this->m_data should be deleted as an array
    if (this->m_data)
    {
        delete[]this->m_data;
        this->m_data = nullptr;
    }

    this->m_dim = 0;
    // std::cout << "destroy me end" << std::endl;
}

void neurons::Vector::destroyMeWithoutMemoryRelease()
{
    this->m_dim = 0;
    this->m_norm = nullptr;
    this->m_data = nullptr;
}

bool neurons::operator==(const neurons::Vector & left, const neurons::Vector & right)
{
    if (left.getNumDimensions() != right.getNumDimensions())
    {
        return false;
    }

    lint dim = left.getNumDimensions();
    for (lint i = 0; i < dim; i++)
    {
        if (left[i] != right[i])
        {
            return false;
        }
    }

    return true;
}

bool neurons::operator!=(const neurons::Vector & left, const neurons::Vector & right)
{
    return !(left == right);
}

neurons::Vector neurons::operator+(const neurons::Vector & left, const neurons::Vector & right)
{
    if (left.getNumDimensions() != right.getNumDimensions())
    {
        throw std::invalid_argument(diff_dim);
    }

    lint dim = left.getNumDimensions();
    neurons::Vector result(dim);

    for (lint i = 0; i < dim; i++)
    {
        result[i] = left[i] + right[i];
    }

    return result;
}

neurons::Vector neurons::operator - (const neurons::Vector & left, const neurons::Vector & right)
{
    if (left.getNumDimensions() != right.getNumDimensions())
    {
        throw std::invalid_argument(diff_dim);
    }

    lint dim = left.getNumDimensions();
    neurons::Vector result(dim);

    for (lint i = 0; i < dim; i++)
    {
        result[i] = left[i] - right[i];
    }

    return result;
}

double neurons::operator * (const neurons::Vector & left, const neurons::Vector & right)
{
    if (left.getNumDimensions() != right.getNumDimensions())
    {
        throw std::invalid_argument(diff_dim);
    }

    double sum = 0.0;
    lint dim = left.getNumDimensions();

    for (lint i = 0; i < dim; i++)
    {
        sum += left[i] * right[i];
    }

    return sum;
}

neurons::Vector neurons::operator * (const neurons::Vector & left, double scalar)
{
    lint dim = left.getNumDimensions();
    neurons::Vector result(left.getNumDimensions());

    for (lint i = 0; i < dim; i++)
    {
        result[i] = left[i] * scalar;
    }

    return result;
}

neurons::Vector neurons::operator * (double scalar, const neurons::Vector & right)
{
    lint dim = right.getNumDimensions();
    neurons::Vector result(right.getNumDimensions());

    for (lint i = 0; i < dim; i++)
    {
        result[i] = scalar * right[i];
    }

    return result;
}

neurons::Vector neurons::operator / (const Vector & left, double scalar)
{
    lint dim = left.getNumDimensions();
    neurons::Vector result(left.getNumDimensions());

    for (lint i = 0; i < dim; i++)
    {
        result[i] = left[i] / scalar;
    }

    return result;
}

std::ostream & neurons::operator << (std::ostream & os, const neurons::Vector &v)
{
    os << '[';
    lint dim = v.getNumDimensions();
    
    for (lint i = 0; i < dim; i++)
    {
        os << v[i];

        if (i < dim - 1)
        {
            os << ' ';
        }
    }

    os << ']';

    return os;
}
