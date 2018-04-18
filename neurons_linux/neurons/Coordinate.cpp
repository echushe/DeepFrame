#include "Coordinate.h"
#include "Exceptions.h"
#include <functional>
#include <cstring>

neurons::Coordinate::Coordinate(std::initializer_list<lint> list)
    : m_dim(list.end() - list.begin())
{
    if (this->m_dim < 1)
    {
        throw std::invalid_argument(invalid_coord_dim);
    }

    this->m_data = new lint[this->m_dim];

    lint *p = this->m_data;
    for (auto itr = list.begin(); itr != list.end(); ++itr, ++p)
    {
        if (*itr < 0)
        {
            throw std::invalid_argument(invalid_coord_val);
        }
        *p = *itr;
    }
}

neurons::Coordinate::Coordinate(std::initializer_list<lint> list, const Shape & shape)
    : Coordinate(list)
{
    if (shape.m_dim != this->m_dim)
    {
        delete[]this->m_data;
        throw std::invalid_argument(std::string("Shape and coordinate incompatible"));
    }

    for (lint i = 0; i < this->m_dim; ++i)
    {
        if (this->m_data[i] >= shape.m_data[i])
        {
            delete[]this->m_data;
            throw std::invalid_argument(std::string("Shape and coordinate incompatible"));
        }
    }

    this->m_shape = std::make_unique<Shape>(shape);
}


neurons::Coordinate::Coordinate(const Shape & shape)
    : m_dim{ shape.m_dim }, m_shape{ std::make_unique<Shape>(shape) }
{
    this->m_data = new lint[this->m_dim];
    for (lint i = 0; i < this->m_dim; ++i)
    {
        this->m_data[i] = 0;
    }
}


neurons::Coordinate::Coordinate(const Coordinate & other)
    : m_dim{ other.m_dim }, m_shape{ std::make_unique<Shape>(*(other.m_shape)) }
{
    this->m_data = new lint[this->m_dim];

    std::memcpy(this->m_data, other.m_data, this->m_dim * sizeof(lint));
}


neurons::Coordinate::Coordinate(Coordinate && other)
    : m_dim{ other.m_dim }, m_data{ other.m_data }, m_shape{ std::move(other.m_shape) }
{
    other.m_dim = 0;
    other.m_data = nullptr;
}

neurons::Coordinate::~Coordinate()
{
    delete[]this->m_data;
}

neurons::Coordinate & neurons::Coordinate::operator = (const Coordinate & other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_shape = std::make_unique<Shape>(*(other.m_shape));

    this->m_data = new lint[m_dim];
    std::memcpy(this->m_data, other.m_data, m_dim * sizeof(lint));

    return *this;
}

neurons::Coordinate & neurons::Coordinate::operator = (Coordinate && other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_shape = std::move(other.m_shape);
    this->m_data = other.m_data;

    other.m_dim = 0;
    other.m_data = nullptr;

    return *this;
}

lint neurons::Coordinate::operator [] (lint index) const
{
    return this->m_data[index];
}

lint & neurons::Coordinate::operator [] (lint index)
{
    return this->m_data[index];
}

neurons::Coordinate & neurons::Coordinate::operator++()
{
    if (nullptr == this->m_shape)
    {
        throw std::bad_function_call();
    }

    lint plus_pos = this->m_dim - 1;
    while (plus_pos >= 0)
    {
        lint increased = this->m_data[plus_pos] + 1;
        if (increased < this->m_shape->m_data[plus_pos])
        {
            this->m_data[plus_pos] = increased;
            break;
        }
        else
        {
            this->m_data[plus_pos] = 0;
            --plus_pos;
        }
    }

    return *this;
}

neurons::Coordinate neurons::Coordinate::operator++(int)
{
    Coordinate copy{ *this };
    ++(*this);
    return copy;
}

neurons::Coordinate & neurons::Coordinate::transposed_plus()
{
    if (nullptr == this->m_shape)
    {
        throw std::bad_function_call();
    }

    lint plus_pos = 0;
    while (plus_pos < this->m_dim)
    {
        lint increased = this->m_data[plus_pos] + 1;
        if (increased < this->m_shape->m_data[plus_pos])
        {
            this->m_data[plus_pos] = increased;
            break;
        }
        else
        {
            this->m_data[plus_pos] = 0;
            ++plus_pos;
        }
    }

    return *this;
}

lint neurons::Coordinate::dim() const
{
    return this->m_dim;
}

neurons::Coordinate & neurons::Coordinate::reverse()
{
    if (nullptr != this->m_shape)
    {
        this->m_shape->reverse();
    }

    for (lint i = 0; i < this->m_dim / 2; ++i)
    {
        lint k = this->m_data[i];
        this->m_data[i] = this->m_data[this->m_dim - 1 - i];
        this->m_data[this->m_dim - 1 - i] = k;
    }

    return *this;
}

bool neurons::operator == (const Coordinate & left, const Coordinate & right)
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

bool neurons::operator != (const Coordinate & left, const Coordinate & right)
{
    return !(left == right);
}

std::ostream & neurons::operator<<(std::ostream & os, const Coordinate & co)
{
    os << '[';
    for (lint i = 0; i < co.m_dim; ++i)
    {
        os << co.m_data[i];
        if (i < co.m_dim - 1)
        {
            os << ", ";
        }
    }
    os << ']';

    return os;
}

neurons::Coordinate neurons::reverse(const Coordinate & sh)
{
    Coordinate to_reverse{ sh };
    return to_reverse.reverse();
}
