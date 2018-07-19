#pragma once
#include "TMatrix.h"

namespace neurons
{
    # define M_PI          3.141592653589793238462643383279502884L /* pi */

    class Activation
    {
    public:
        static const std::string LINEAR;
        static const std::string SIGMOID;
        static const std::string TANH;
        static const std::string RELU;
        static const std::string LEAKYRELU;
        static const std::string ARCTAN;
        static const std::string SIN;
        static const std::string SOFTSIGN;
        static const std::string SOFTMAX;
        static const std::string NULL_FUNC;

        static std::unique_ptr<Activation> get_function_by_name(const std::string & func_name);

    public:
        Activation() {}

        virtual std::unique_ptr<Activation> clone() = 0;

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in) = 0;

        virtual std::string to_string() const = 0;
    };


    class Linear : public Activation
    {
    public:
        Linear() {};

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);

        virtual std::string to_string() const;
    };


    class Sigmoid : public Activation
    {
    public:
        Sigmoid() {}

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);

        virtual std::string to_string() const;
    };


    class Tanh : public Activation
    {
    public:
        Tanh() {}

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);

        virtual std::string to_string() const;
    };


    class Relu : public Activation
    {
    public:
        Relu() {}

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);

        virtual std::string to_string() const;
    };


    class LeakyRelu : public Activation
    {
    public:
        LeakyRelu() {}

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);

        virtual std::string to_string() const;
    };


    class Arctan : public Activation
    {
    public:
        Arctan() {}

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);

        virtual std::string to_string() const;
    };


    class Sin : public Activation
    {
    public:
        Sin() {}

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);

        virtual std::string to_string() const;
    };


    class Softsign : public Activation
    {
    public:
        Softsign() {}

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);

        virtual std::string to_string() const;
    };


    class Softmax : public Activation
    {
    public:
        Softmax() {};

        virtual std::unique_ptr<Activation> clone();

        virtual void operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in);

        virtual std::string to_string() const;
    };


    class ErrorFunction
    {
    protected:
        std::unique_ptr<Activation> m_act_func;
        mutable TMatrix<> m_act;

    public:
        static const std::string HALF_SQUARE_ERROR;
        static const std::string SIGMOID_CROSS_ENTROPY;
        static const std::string SOFTMAX_CROSS_ENTROPY;
        static const std::string NULL_FUNC;

        static std::unique_ptr<ErrorFunction> get_function_by_name(std::string & func_name);

    public:
        ErrorFunction() {}

        virtual std::unique_ptr<ErrorFunction> clone() = 0;

        virtual double operator () (TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input) = 0;
        virtual double operator () (TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input) = 0;

        virtual std::string to_string() const = 0;

        virtual TMatrix<> & get_activation() const = 0;

        std::unique_ptr<Activation> get_act_func() const;

    };


    class HalfSquareError : public ErrorFunction
    {
    private:

    public:
        HalfSquareError(const std::unique_ptr<Activation> & act_func);

        HalfSquareError(Activation * act_func);

        virtual std::unique_ptr<ErrorFunction> clone();

        virtual double operator () (TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);
        virtual double operator () (TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);

        virtual TMatrix<> & get_activation() const;

        virtual std::string to_string() const;
    };


    class Sigmoid_CrossEntropy : public ErrorFunction
    {
    private:
        // mutable TMatrix<> m_sigmoid;

    public:
        Sigmoid_CrossEntropy();

        virtual std::unique_ptr<ErrorFunction> clone();

        virtual double operator () (TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);
        virtual double operator () (TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);

        virtual TMatrix<> & get_activation() const;

        virtual std::string to_string() const;
    };


    class Softmax_CrossEntropy : public ErrorFunction
    {
    private:
        // mutable TMatrix<> m_softmax;

    public:
        Softmax_CrossEntropy();

        virtual std::unique_ptr<ErrorFunction> clone();

        virtual double operator () (TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);
        virtual double operator () (TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input);

        virtual TMatrix<> & get_activation() const;

        virtual std::string to_string() const;
    };


    lint now_in_seconds();

    lint now_in_milliseconds();

    double gaussian_function(double mu, double sigma, double x);
}
