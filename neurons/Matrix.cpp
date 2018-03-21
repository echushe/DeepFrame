#include "Matrix.h"
#include "Exceptions.h"
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <random>
#include <sstream>
#include <cstring>
#include <functional>


neurons::Matrix::Matrix()
    : m_shape{}, m_data{ nullptr }
{}

neurons::Matrix::Matrix(const Shape & shape)
    : m_shape{ shape }
{
    if (shape.m_size < 1)
    {
        this->m_data = nullptr;
    }
    else
    {
        m_data = new double[this->m_shape.m_size];
    }
}

neurons::Matrix::Matrix(const Shape & shape, double value)
    : Matrix{ shape }
{
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] = value;
    }
}

neurons::Matrix::Matrix(const std::vector<Matrix>& matrices)
    : Matrix{}
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
                        std::string("Matrix::Matrix: invalid matrix array because of different shapes of matrices"));
                }
            }

            this->m_shape = Shape{ array_size } + mat_sh;
            this->m_data = new double[this->m_shape.m_size];
            double *this_pos = this->m_data;
            double *that_pos;

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
        
neurons::Matrix::Matrix (const Matrix & other)
    : m_shape {other.m_shape}
{
    lint size = m_shape.m_size;
    m_data = new double[size];

    std::memcpy(m_data, other.m_data, size * sizeof(double));
}

neurons::Matrix::Matrix (Matrix && other)
    : m_shape {std::move(other.m_shape)}, m_data {other.m_data}
{
    other.m_data = nullptr;
}

neurons::Matrix::Matrix (const Vector & vec, bool transpose)
    : m_shape {vec.m_dim, 1}
{
    if (transpose)
    {
        m_shape.m_data[0] = 1;
        m_shape.m_data[1] = vec.m_dim;
    }

    lint size = m_shape.m_size;
    m_data = new double[size];

    std::memcpy(this->m_data, vec.m_data, size * sizeof(double));
}

neurons::Matrix::~Matrix()
{
    delete[]this->m_data;
}

neurons::Matrix & neurons::Matrix::operator = (const Matrix & other)
{
    delete []this->m_data;
    this->m_shape = other.m_shape;

    lint size = this->m_shape.m_size;
    this->m_data = new double[size];

    std::memcpy(this->m_data, other.m_data, size * sizeof(double));

    return *this;
}

neurons::Matrix & neurons::Matrix::operator = (Matrix && other)
{
    delete []m_data;
    this->m_shape = std::move(other.m_shape);
    
    this->m_data = other.m_data;
    other.m_data = nullptr;

    return *this;
}

neurons::Matrix & neurons::Matrix::operator = (double scalar)
{
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] = scalar;
    }

    return *this;
}

double neurons::Matrix::at(const Coordinate & pos) const
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

double & neurons::Matrix::at(const Coordinate & pos)
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

double neurons::Matrix::operator [] (const Coordinate & pos) const
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

double & neurons::Matrix::operator [] (const Coordinate & pos)
{
    lint index = 0;
    for (lint i = 0; i < pos.m_dim; ++i)
    {
        index *= this->m_shape.m_data[i];
        index += pos.m_data[i];
    }

    return this->m_data[index];
}


neurons::Matrix::operator neurons::Vector() const
{
    lint size = this->m_shape.m_size;
    Vector vec = Vector(size);
    for (lint i = 0; i < size; ++i)
    {
        vec[i] = this->m_data[i];
    }

    return vec;
}

neurons::Vector neurons::Matrix::flaten() const
{
    return *this;
}

neurons::Matrix & neurons::Matrix::operator += (const Matrix & other)
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

neurons::Matrix & neurons::Matrix::operator -= (const Matrix & other)
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


neurons::Matrix & neurons::Matrix::operator += (double scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] += scalar;
    }

    return *this;
}


neurons::Matrix & neurons::Matrix::operator -= (double scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] -= scalar;
    }

    return *this;
}


neurons::Matrix & neurons::Matrix::operator *= (double scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] *= scalar;
    }

    return *this;
}

neurons::Matrix & neurons::Matrix::operator /= (double scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] /= scalar;
    }

    return *this;
}

neurons::Matrix & neurons::Matrix::gaussian_random(double mu, double sigma)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mu, sigma);

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        this->m_data[i] = distribution(generator);
    }

    return *this;
}

neurons::Matrix & neurons::Matrix::uniform_random(double min, double max)
{
    if (min >= max)
    {
        throw std::invalid_argument(std::string("Matrix::uniform_random: min should be smaller than max"));
    }

    lint size = m_shape.m_size;
    double range = max - min;
    for (lint i = 0; i < size; ++i)
    {
        this->m_data[i] = min + (static_cast<double>(rand()) / RAND_MAX) * range;
    }

    return *this;
}

neurons::Matrix & neurons::Matrix::normalize(double min, double max)
{
    if (min >= max)
    {
        throw std::invalid_argument(std::string("Matrix::normalize: min should be smaller than max"));
    }

    lint size = m_shape.m_size;
    double range = max - min;

    double l_max = std::numeric_limits<double>::max() * (-1);
    double l_min = std::numeric_limits<double>::max();

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

    double l_range = l_max - l_min;
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

neurons::Matrix & neurons::Matrix::normalize()
{
    lint size = this->m_shape.m_size;

    if (0 == size)
    {
        throw std::bad_function_call();
    }

    double mean = 0;
    for (lint i = 0; i < size; ++i)
    {
        mean += this->m_data[i];
    }
    mean /= size;

    double var = 0;
    double sub;
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

neurons::Matrix & neurons::Matrix::reshape(const Shape & shape)
{
    if (shape.m_size != this->m_shape.m_size)
    {
        throw std::invalid_argument(
            std::string("Matrix::reshape: the new shape should be compatible with number of elements in this matrix"));
    }

    this->m_shape = shape;

    return *this;
}

neurons::Matrix & neurons::Matrix::left_extend_shape()
{
    this->m_shape.left_extend();
    return *this;
}

neurons::Matrix & neurons::Matrix::right_extend_shape()
{
    this->m_shape.right_extend();
    return *this;
}

neurons::Matrix neurons::Matrix::get_left_extended(lint duplicate) const
{
    if (duplicate < 1)
    {
        throw std::invalid_argument(std::string("neurons::Matrix::get_left_extended: duplicate should be a positive value"));
    }

    if (this->m_shape.m_size < 1)
    {
        throw std::bad_function_call();
    }

    Matrix mat{ Shape{duplicate} + this->m_shape };
    lint size = this->m_shape.m_size;
    double *mat_start = mat.m_data;

    for (lint j = 0; j < duplicate; ++j)
    {
        std::memcpy(mat_start, this->m_data, size * sizeof(double));
        mat_start += size;
    }

    return mat;
}

neurons::Matrix & neurons::Matrix::scale_one_dimension(lint dim, const Vector & scales)
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("neurons::Matrix::scale_one_dimension: dimension index out of range.");
    }

    if (this->m_shape.m_data[dim] != scales.dim())
    {
        throw std::invalid_argument(
            "neurons::Matrix::scale_one_dimension: size of the vector is not compatible with size of this matrix dimension.");
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

    double *start = this->m_data;
    
    for (lint i = 0; i < scales.dim(); ++i)
    {
        double scale = scales[i];
        double *l_start = start;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            double *ele = l_start;
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

neurons::Coordinate neurons::Matrix::argmax() const
{
    double max = std::numeric_limits<double>::max() * (-1);
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

double neurons::Matrix::max() const
{
    double max = std::numeric_limits<double>::max() * (-1);
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

neurons::Coordinate neurons::Matrix::argmin() const
{
    double min = std::numeric_limits<double>::max() * (-1);
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

double neurons::Matrix::min() const
{
    double min = std::numeric_limits<double>::max();
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

double neurons::Matrix::mean() const
{
    lint size = this->m_shape.m_size;
    double sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        sum += this->m_data[i];
    }

    return sum / size;
}

double neurons::Matrix::variance() const
{
    double mean = this->mean();

    lint size = this->m_shape.m_size;
    double sum = 0;
    double sub;
    for (lint i = 0; i < size; ++i)
    {
        sub = this->m_data[i] - mean;
        sum += sub * sub;
    }

    return sum / size;
}

std::vector<neurons::Matrix> neurons::Matrix::collapse(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("neurons::Matrix::scale_one_dimension: dimension index out of range.");
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

    std::vector<Matrix> all_collapsed;
    double *start = this->m_data;

    for (lint i = 0; i < this->m_shape.m_data[dim]; ++i)
    {
        Matrix collapsed{ sh_collapsed };
        double *l_start = start;
        double *clps_ele = collapsed.m_data;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            double *ele = l_start;
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

neurons::Matrix neurons::Matrix::fuse(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("neurons::Matrix::fuse: dimension index out of range.");
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

    Matrix fused{ sh_collapsed, 0 };
    double *start = this->m_data;

    for (lint i = 0; i < this->m_shape.m_data[dim]; ++i)
    {
        Matrix collapsed{ sh_collapsed };
        double *l_start = start;
        double *clps_ele = collapsed.m_data;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            double *ele = l_start;
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


neurons::Matrix neurons::Matrix::reduce_mean(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("neurons::Matrix::fuse: dimension index out of range.");
    }

    lint dim_size = this->m_shape.m_data[dim];
    return this->fuse(dim) / static_cast<double>(dim_size);
}

double neurons::Matrix::euclidean_norm() const
{
    // Calculate the norm
    double norm = 0;
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        norm += this->m_data[i] * this->m_data[i];
    }

    return norm;
}


neurons::Shape neurons::Matrix::shape() const
{
    return m_shape;
}


void neurons::Matrix::print(std::ostream & os, lint dim_index, const double *start, size_t block_size) const
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
            double val = *(start + i);
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

// Overloading of a == b
bool neurons::operator == (const Matrix &left, const Matrix &right)
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
bool neurons::operator != (const Matrix &left, const Matrix &right)
{
    return !(left == right);
}

// Overloading of a + b
neurons::Matrix neurons::operator + (const Matrix &left, const Matrix &right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(invalid_shape);
    }

    lint size = left.m_shape.size();

    Matrix mat{ left.m_shape };
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = left.m_data[i] + right.m_data[i];
    }

    return mat;
}

// Overloading of a - b
neurons::Matrix neurons::operator - (const Matrix &left, const Matrix &right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(invalid_shape);
    }

    lint size = left.m_shape.size();

    Matrix mat{ left.m_shape };
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = left.m_data[i] - right.m_data[i];
    }

    return mat;
}

// Matrix multiplication
neurons::Matrix neurons::matrix_multiply(const Matrix & left, const Matrix & right, lint l_dims_merge, lint r_dims_merge)
{
    //
    if (left.m_shape.dim() < 2 || right.m_shape.dim() < 2)
    {
        throw std::invalid_argument(std::string(
            "Matrix multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
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
    Matrix mat{ left_rows_sh + right_cols_sh };
    double *left_start = left.m_data;
    double *right_start = right.m_data;
    double *mat_p = mat.m_data;

    for (lint i = 0; i < left_rows; ++i)
    {
        for (lint j = 0; j < right_columns; ++j)
        {
            *mat_p = 0.0;
            double * left_p = left_start;
            double * right_p = right_start;

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

neurons::Matrix neurons::matrix_multiply(const Matrix & left, const Matrix & right)
{
    /*
    if (1 == left.m_shape.dim() || 1 == right.m_shape.dim())
    {
        if (1 == left.m_shape.dim())
        {
            Matrix extended_left{ left };
            extended_left.left_extend_shape();
            return extended_left * right;
        }
        else
        {
            Matrix extended_right{ right };
            extended_right.right_extend_shape();
            return left * extended_right;
        }
    }
    */

    if (left.m_shape.dim() < 2 || right.m_shape.dim() < 2)
    {
        throw std::invalid_argument(std::string(
            "Matrix multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
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

    Matrix mat{ left_rows_sh + right_cols_sh };
    double *left_start = left.m_data;
    double *right_start = right.m_data;
    double *mat_p = mat.m_data;

    for (lint i = 0; i < left_rows; ++i)
    {
        for (lint j = 0; j < right_columns; ++j)
        {
            *mat_p = 0.0;
            double * left_p = left_start;
            double * right_p = right_start;

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

//
neurons::Matrix neurons::multiply(const Matrix & left, const Matrix & right)
{
    if (left.m_shape != right.m_shape)
    {
        throw std::invalid_argument(invalid_shape);
    }

    lint size = left.m_shape.size();

    Matrix mat{ left.m_shape };
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = left.m_data[i] * right.m_data[i];
    }

    return mat;
}

double neurons::dot_product(const Matrix & left, const Matrix & right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(incompatible_size);
    }

    lint size = left.m_shape.size();
    double sum = 0.0;

    for (lint i = 0; i < size; ++i)
    {
        sum += left.m_data[i] * right.m_data[i];
    }

    return sum;
}

// Overloading of a * b (dot product)
neurons::Matrix neurons::operator * (const Matrix &left, const Matrix &right)
{
    return matrix_multiply(left, right);
}

// Overloading of a * b, b is a double
neurons::Matrix neurons::operator * (const Matrix &left, double scalar)
{
    Matrix mat{ left.m_shape };

    lint size = left.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = left.m_data[i] * scalar;
    }

    return mat;
}

// Overloading of a * b, a is a double
neurons::Matrix neurons::operator * (double scalar, const Matrix &right)
{
    Matrix mat{ right.m_shape };

    lint size = right.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = right.m_data[i] * scalar;
    }

    return mat;
}

// Overloading of a / b, b is a double
neurons::Matrix neurons::operator / (const Matrix &left, double scalar)
{
    Matrix mat{ left.m_shape };

    lint size = left.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = left.m_data[i] / scalar;
    }

    return mat;
}

neurons::Matrix neurons::transpose(const Matrix & in)
{
    Shape reversed_shape{reverse(in.m_shape)};
    Matrix transposed{ reversed_shape };

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

    double *ele_pos = transposed.m_data;

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


neurons::Matrix neurons::scale_one_dimension(const Matrix & in, lint in_dim, const Vector & scales)
{
    Matrix mat{ in };
    mat.scale_one_dimension(in_dim, scales);
    return mat;
}


// Overloading of output stream << operator
std::ostream & neurons::operator << (std::ostream& os, const Matrix & m)
{
    lint size = m.m_shape.size();
    size_t block_size = 0;
    for (lint i = 0; i < size; ++i)
    {
        std::ostringstream stream;
        stream << m.m_data[i];
        size_t len = stream.str().size();
        if ( len > block_size)
        {
            block_size = len;
        }
    }

    block_size += 2;

    m.print(os, 0, m.m_data, block_size);

    return os;
}

