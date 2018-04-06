#pragma once
#include "TMatrix.h"

namespace neurons
{
    # define M_PI          3.141592653589793238462643383279502884L /* pi */

    class Activation
    {
    public:
        Activation() {}

        virtual std::unique_ptr<Activation> clone() = 0;

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in) = 0;
    };


    class Linear : public Activation
    {
    public:
        Linear() {};

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);
    };


    class Sigmoid : public Activation
    {
    public:
        Sigmoid() {}

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);
    };


    class Tanh : public Activation
    {
    public:
        Tanh() {}

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);
    };


    class Relu : public Activation
    {
    public:
        Relu() {}

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);
    };


    class Softmax : public Activation
    {
    public:
        Softmax() {};

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);
    };


    class ErrorFunction
    {
    public:
        ErrorFunction() {}

        virtual std::unique_ptr<ErrorFunction> clone() = 0;

        virtual double operator () (TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input) = 0;
        virtual double operator () (TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input) = 0;

    };


    class HalfSquareError : public ErrorFunction
    {
    private:
        std::unique_ptr<Activation> m_act_func;

        mutable TMatrix<> m_act;

    public:
        HalfSquareError(const std::unique_ptr<Activation> & act_func);

        HalfSquareError(Activation * act_func);

        virtual std::unique_ptr<ErrorFunction> clone();

        virtual double operator () (TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);
        virtual double operator () (TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);

        TMatrix<> & get_activation() const;
    };


    class Sigmoid_CrossEntropy : public ErrorFunction
    {
    private:
        mutable TMatrix<> m_sigmoid;

    public:
        Sigmoid_CrossEntropy() {}

        virtual std::unique_ptr<ErrorFunction> clone();

        virtual double operator () (TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);
        virtual double operator () (TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);

        TMatrix<> & get_sigmoid() const;
    };


    class Softmax_CrossEntropy : public ErrorFunction
    {
    private:
        mutable TMatrix<> m_softmax;

    public:
        Softmax_CrossEntropy() {}

        virtual std::unique_ptr<ErrorFunction> clone();

        virtual double operator () (TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);
        virtual double operator () (TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);

        TMatrix<> & get_SoftMax() const;
    };


    lint now_in_seconds();

    lint now_in_milliseconds();

    double gaussian_function(double mu, double sigma, double x);
}
