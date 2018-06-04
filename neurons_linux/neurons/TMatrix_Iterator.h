#pragma once

#include <iterator>
#include <cassert>

namespace neurons
{
    template <typename dtype> class TMatrix;

    template <typename dtype = double>
    class TMatrix_Iterator : public std::iterator<std::bidirectional_iterator_tag, dtype>
    {
    protected:
        lint m_mat_size;
        dtype * m_mat_data;
        lint m_ele_index;

    public:
        TMatrix_Iterator(lint mat_size, dtype * mat_data, lint ele_index);

        TMatrix_Iterator(const TMatrix_Iterator & other);

        TMatrix_Iterator(TMatrix_Iterator && other);

        TMatrix_Iterator & operator = (const TMatrix_Iterator & other);

        TMatrix_Iterator & operator = (TMatrix_Iterator && other);

    public:

        TMatrix_Iterator & operator ++ ();

        TMatrix_Iterator operator ++ (int);

        TMatrix_Iterator & operator -- ();

        TMatrix_Iterator operator -- (int);

        dtype & operator * () const;

        dtype * operator -> () const;

        bool operator == (const TMatrix_Iterator & right) const;

        bool operator != (const TMatrix_Iterator & right) const;

    };

    template<typename dtype>
    inline TMatrix_Iterator<dtype>::TMatrix_Iterator(lint mat_size, dtype * mat_data, lint ele_index)
        : m_mat_size{ mat_size }, m_mat_data{ mat_data }, m_ele_index{ ele_index }
    {}

    template<typename dtype>
    inline TMatrix_Iterator<dtype>::TMatrix_Iterator(const TMatrix_Iterator & other)
        : m_mat_size{ other.m_mat_size }, m_mat_data{ other.m_mat_data }, m_ele_index{ other.m_ele_index }
    {}

    template<typename dtype>
    inline TMatrix_Iterator<dtype>::TMatrix_Iterator(TMatrix_Iterator && other)
        : m_mat_size{ other.m_mat_size }, m_mat_data{ other.m_mat_data }, m_ele_index{ other.m_ele_index }
    {
        other.m_mat_size = 0;
        other.m_mat_data = nullptr;
        other.m_ele_index = 0;
    }

    template<typename dtype>
    inline TMatrix_Iterator<dtype> & TMatrix_Iterator<dtype>::operator=(const TMatrix_Iterator & other)
    {
        this->m_mat_size = other.mat_size;
        this->m_mat_data = other.m_mat_data;
        this->m_ele_index = other.m_ele_index;
    }

    template<typename dtype>
    inline TMatrix_Iterator<dtype> & TMatrix_Iterator<dtype>::operator=(TMatrix_Iterator && other)
    {
        this->m_mat_size = other.mat_size;
        this->m_mat_data = other.m_mat_data;
        this->m_ele_index = other.m_ele_index;

        other.m_mat_data = nullptr;
    }

    template<typename dtype>
    inline TMatrix_Iterator<dtype> & TMatrix_Iterator<dtype>::operator++()
    {
        assert(this->m_mat_index < this->m_mat_size);

        ++this->m_ele_index;

        return *this;
    }

    template<typename dtype>
    inline TMatrix_Iterator<dtype> TMatrix_Iterator<dtype>::operator++(int)
    {
        assert(this->m_mat_index < this->m_mat_size);

        TMatrix_Iterator old{ *this };

        ++this->m_ele_index;

        return old;
    }

    template<typename dtype>
    inline TMatrix_Iterator<dtype> & TMatrix_Iterator<dtype>::operator--()
    {
        assert(this->m_mat_index > 0);

        --this->m_ele_index;

        return *this;
    }

    template<typename dtype>
    inline TMatrix_Iterator<dtype> TMatrix_Iterator<dtype>::operator--(int)
    {
        assert(this->m_mat_index > 0);

        TMatrix_Iterator old{ *this };

        --this->m_ele_index;

        return old;
    }

    template<typename dtype>
    inline dtype & TMatrix_Iterator<dtype>::operator*() const
    {
        return this->m_mat_data[this->m_ele_index];
    }

    template<typename dtype>
    inline dtype * TMatrix_Iterator<dtype>::operator->() const
    {
        return this->m_mat_data + this->m_ele_index;
    }

    template<typename dtype>
    inline bool TMatrix_Iterator<dtype>::operator==(const TMatrix_Iterator & right) const
    {
        return (this->m_ele_index == right.m_ele_index) &&
            (this->m_mat_data == right.m_mat_data) &&
            (this->m_mat_size == right.m_mat_size);
    }

    template<typename dtype>
    inline bool TMatrix_Iterator<dtype>::operator!=(const TMatrix_Iterator & right) const
    {
        return !(*this == right);
    }
}