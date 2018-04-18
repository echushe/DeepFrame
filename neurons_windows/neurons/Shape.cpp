#include "Shape.h"
#include "Exceptions.h"
#include <functional>
#include <cstring>
#include <algorithm>

neurons::Shape::Shape()
    : m_dim{ 0 }, m_size{ 0 }, m_data{ nullptr }
{}

neurons::Shape::Shape(std::initializer_list<lint> list)
    : m_dim(list.end() - list.begin()), m_size{ 1 }
{
    if (this->m_dim < 1)
    {
        this->m_dim = 0;
        this->m_size = 0;
        this->m_data = nullptr;
    }

    this->m_data = new lint[this->m_dim];
    lint *p = this->m_data;
    for (auto itr = list.begin(); itr != list.end(); ++itr, ++p)
    {
        if (*itr <= 0)
        {
            throw std::invalid_argument(invalid_shape_val);
        }
        *p = *itr;
    }

    for (lint i = 0; i < this->m_dim; ++i)
    {
        this->m_size *= this->m_data[i];
    }
}

neurons::Shape::Shape(const Shape & other)
    : m_dim{ other.m_dim }, m_size(other.m_size)
{
    this->m_data = new lint[this->m_dim];

    std::memcpy(this->m_data, other.m_data, this->m_dim * sizeof(lint));
}

neurons::Shape::Shape(Shape && other)
    : m_dim{ other.m_dim }, m_size(other.m_size), m_data{ other.m_data }
{
    other.m_dim = 0;
    other.m_size = 0;
    other.m_data = nullptr;
}

neurons::Shape::~Shape()
{
    delete[]this->m_data;
}

neurons::Shape & neurons::Shape::operator = (const Shape & other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_size = other.m_size;

    this->m_data = new lint[m_dim];
    std::memcpy(this->m_data, other.m_data, m_dim * sizeof(lint));

    return *this;
}

neurons::Shape & neurons::Shape::operator = (Shape && other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_size = other.m_size;
    this->m_data = other.m_data;

    other.m_dim = 0;
    other.m_size = 0;
    other.m_data = nullptr;

    return *this;
}

lint neurons::Shape::operator [] (lint index) const
{
    return this->m_data[index];
}

/*
lint & neurons::Shape::operator [] (lint index)
{
    return this->m_data[index];
}
*/

lint neurons::Shape::size() const
{
    return this->m_size;
}

lint neurons::Shape::dim() const
{
    return this->m_dim;
}

neurons::Shape & neurons::Shape::left_extend()
{
    if (0 == this->m_dim)
    {
        throw std::bad_function_call();
    }

    ++this->m_dim;
    lint *new_data = new lint[this->m_dim];
    *new_data = 1;

    for (lint i = 1; i < this->m_dim; ++i)
    {
        new_data[i] = this->m_data[i - 1];
    }
    delete[]this->m_data;
    this->m_data = new_data;

    return *this;
}

neurons::Shape & neurons::Shape::right_extend()
{
    if (0 == this->m_dim)
    {
        throw std::bad_function_call();
    }

    ++this->m_dim;
    lint *new_data = new lint[this->m_dim];

    for (lint i = 0; i < this->m_dim - 1; ++i)
    {
        new_data[i] = this->m_data[i];
    }
    new_data[this->m_dim - 1] = 1;
    delete[]this->m_data;
    this->m_data = new_data;

    return *this;
}

neurons::Shape & neurons::Shape::reverse()
{
    for (lint i = 0; i < this->m_dim / 2; ++i)
    {
        lint k = this->m_data[i];
        this->m_data[i] = this->m_data[this->m_dim - 1 - i];
        this->m_data[this->m_dim - 1 - i] = k;
    }

    return *this;
}

neurons::Shape neurons::Shape::sub_shape(lint dim_first, lint dim_last) const
{
    Shape sub_shape;

    if (dim_first > dim_last)
    {
        return sub_shape;
    }

    if (dim_first < 0 || dim_last > this->m_dim - 1)
    {
        throw std::invalid_argument(
            std::string("Shape::sub_shape: index of dimension out of range")
        );
    }

    sub_shape.m_dim = dim_last - dim_first + 1;
    sub_shape.m_data = new lint[sub_shape.m_dim];
    sub_shape.m_size = 1;

    for (lint i = dim_first; i <= dim_last; ++i)
    {
        sub_shape.m_data[i - dim_first] = this->m_data[i];
    }

    for (lint i = 0; i < sub_shape.m_dim; ++i)
    {
        sub_shape.m_size *= sub_shape.m_data[i];
    }

    return sub_shape;
}

// Overloading of a == b
bool neurons::operator == (const Shape &left, const Shape &right)
{
    if (left.m_dim != right.m_dim)
    {
        return false;
    }

    for (lint i = 0; i < left.m_dim; ++i)
    {
        if (left.m_data[i] != right.m_data[i])
        {
            return false;
        }
    }

    return true;
}

// Overloading of a != b
bool neurons::operator != (const Shape &left, const Shape &right)
{
    return !(left == right);
}

neurons::Shape neurons::operator + (const Shape & left, const Shape & right)
{
    Shape merged;
    merged.m_dim = left.m_dim + right.m_dim;
    merged.m_size = std::max<lint>(left.m_size, 1) * std::max<lint>(right.m_size, 1);
    merged.m_data = new lint[merged.m_dim];

    for (lint i = 0; i < left.m_dim; ++i)
    {
        merged.m_data[i] = left.m_data[i];
    }

    for (lint i = left.m_dim; i < merged.m_dim; ++i)
    {
        merged.m_data[i] = right.m_data[i - left.m_dim];
    }

    return merged;
}

neurons::Shape neurons::reverse(const Shape & sh)
{
    Shape to_reverse{ sh };
    return to_reverse.reverse();
}

std::ostream & neurons::operator<<(std::ostream & os, const Shape & sh)
{
    os << '[';
    for (lint i = 0; i < sh.m_dim; ++i)
    {
        os << sh.m_data[i];
        if (i < sh.m_dim - 1)
        {
            os << ", ";
        }
    }
    os << ']';

    return os;
}
