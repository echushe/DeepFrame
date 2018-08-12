#include "Functions.h"
#include <math.h>
#include <iostream>
#include <chrono>

const std::string neurons::Activation::LINEAR{ "Linear" };
const std::string neurons::Activation::SIGMOID{ "Sigmoid" };
const std::string neurons::Activation::TANH{ "Tanh" };
const std::string neurons::Activation::RELU{ "Relu" };
const std::string neurons::Activation::LEAKYRELU{ "LeakyRelu" };
const std::string neurons::Activation::ARCTAN{ "Arctan" };
const std::string neurons::Activation::SIN{ "Sin" };
const std::string neurons::Activation::SOFTSIGN{ "Softsign" };
const std::string neurons::Activation::SOFTMAX{ "Softmax" };
const std::string neurons::Activation::NULL_FUNC{ "NULL" };

const std::string neurons::ErrorFunction::HALF_SQUARE_ERROR{ "HalfSquareError" };
const std::string neurons::ErrorFunction::SIGMOID_CROSS_ENTROPY{ "Sigmoid_CrossEntropy" };
const std::string neurons::ErrorFunction::SOFTMAX_CROSS_ENTROPY{ "Softmax_CrossEntropy" };
const std::string neurons::ErrorFunction::NULL_FUNC{ "NULL" };

std::unique_ptr<neurons::Activation> neurons::Activation::get_function_by_name(const std::string & func_name)
{
    if (func_name == LINEAR)
    {
        return std::make_unique<Linear>();
    }
    else if (func_name == SIGMOID)
    {
        return std::make_unique<Sigmoid>();
    }
    else if (func_name == TANH)
    {
        return std::make_unique<Tanh>();
    }
    else if (func_name == RELU)
    {
        return std::make_unique<Relu>();
    }
    else if (func_name == LEAKYRELU)
    {
        return std::make_unique<LeakyRelu>();
    }
    else if (func_name == ARCTAN)
    {
        return std::make_unique<Arctan>();
    }
    else if (func_name == SIN)
    {
        return std::make_unique<Sin>();
    }
    else if (func_name == SOFTSIGN)
    {
        return std::make_unique<Softsign>();
    }
    else if (func_name == SOFTMAX)
    {
        return std::make_unique<Softmax>();
    }
    else if (func_name == NULL_FUNC)
    {
        return nullptr;
    }
    else
    {
        return std::make_unique<Sigmoid>();
    }
}

std::unique_ptr<neurons::Activation> neurons::Linear::clone()
{
    return std::make_unique<neurons::Linear>();
}

void neurons::Linear::operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in)
{
    TMatrix<> l_output{ in.m_shape };
    TMatrix<> l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        l_output.m_data[i] = in.m_data[i];
        l_diff.m_data[i] = 1;
    }

    output = std::move(l_output);
    diff = std::move(l_diff);
}

std::string neurons::Linear::to_string() const
{
    return Activation::LINEAR;
}


std::unique_ptr<neurons::Activation> neurons::Sigmoid::clone()
{
    return std::make_unique<neurons::Sigmoid>();
}

void neurons::Sigmoid::operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in)
{
    TMatrix<> l_output{ in.m_shape };
    TMatrix<> l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        double y = 1.0 / (1.0 + exp(-in.m_data[i]));
        l_output.m_data[i] = y;
        l_diff.m_data[i] = y * (1 - y);
    }

    output = std::move(l_output);
    diff = std::move(l_diff);
}

std::string neurons::Sigmoid::to_string() const
{
    return Activation::SIGMOID;
}


std::unique_ptr<neurons::Activation> neurons::Tanh::clone()
{
    return std::make_unique<neurons::Tanh>();
}

void neurons::Tanh::operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in)
{
    output = TMatrix<>{ in.m_shape };
    diff = TMatrix<>{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        double y = tanh(in.m_data[i]);
        output.m_data[i] = y;
        diff.m_data[i] = 1 - y * y;
    }
}

std::string neurons::Tanh::to_string() const
{
    return Activation::TANH;
}


std::unique_ptr<neurons::Activation> neurons::Relu::clone()
{
    return std::make_unique<neurons::Relu>();
}


void neurons::Relu::operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in)
{
    output = TMatrix<>{ in.m_shape };
    diff = TMatrix<>{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        double x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
            diff.m_data[i] = 1;
        }
        else
        {
            output.m_data[i] = 0;
            diff.m_data[i] = 0;
        }
    }
}

std::string neurons::Relu::to_string() const
{
    return Activation::RELU;
}


std::unique_ptr<neurons::Activation> neurons::LeakyRelu::clone()
{
    return std::make_unique<neurons::LeakyRelu>();
}


void neurons::LeakyRelu::operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in)
{
    output = TMatrix<>{ in.m_shape };
    diff = TMatrix<>{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        double x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
            diff.m_data[i] = 1;
        }
        else
        {
            output.m_data[i] = 0.01 * x;
            diff.m_data[i] = 0.01;
        }
    }
}

std::string neurons::LeakyRelu::to_string() const
{
    return Activation::LEAKYRELU;
}



std::unique_ptr<neurons::Activation> neurons::Arctan::clone()
{
    return std::make_unique<neurons::Arctan>();
}

void neurons::Arctan::operator()(TMatrix<>& output, TMatrix<>& diff, const TMatrix<>& in)
{
    output = TMatrix<>{ in.m_shape };
    diff = TMatrix<>{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = atan(in.m_data[i]);
        diff.m_data[i] = 1 / (1 + in.m_data[i] * in.m_data[i]);
    }
}


std::string neurons::Arctan::to_string() const
{
    return Activation::ARCTAN;
}


std::unique_ptr<neurons::Activation> neurons::Sin::clone()
{
    return std::make_unique<neurons::Sin>();
}

void neurons::Sin::operator()(TMatrix<>& output, TMatrix<>& diff, const TMatrix<>& in)
{
    output = TMatrix<>{ in.m_shape };
    diff = TMatrix<>{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = sin(in.m_data[i]);
        diff.m_data[i] = cos(in.m_data[i]);
    }
}


std::string neurons::Sin::to_string() const
{
    return Activation::SIN;
}


std::unique_ptr<neurons::Activation> neurons::Softsign::clone()
{
    return std::make_unique<neurons::Softsign>();
}

void neurons::Softsign::operator()(TMatrix<>& output, TMatrix<>& diff, const TMatrix<>& in)
{
    output = TMatrix<>{ in.m_shape };
    diff = TMatrix<>{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        double d = 1 + fabs(in.m_data[i]);
        output.m_data[i] = in.m_data[i] / d;
        diff.m_data[i] = 1 / (d * d);
    }
}


std::string neurons::Softsign::to_string() const
{
    return Activation::SOFTSIGN;
}


std::unique_ptr<neurons::Activation> neurons::Softmax::clone()
{
    return std::make_unique<neurons::Softmax>();
}


void neurons::Softmax::operator () (TMatrix<> & output, TMatrix<> & diff, const TMatrix<> & in)
{
    output = TMatrix<>{ in.m_shape };
    diff = TMatrix<>{ in.m_shape };
    lint size = in.m_shape.size();

    double sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = exp(in.m_data[i]);
        sum += output.m_data[i];
    }

    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] /= sum;
        diff.m_data[i] = output.m_data[i] * (1 - output.m_data[i]);
    }
}

std::string neurons::Softmax::to_string() const
{
    return Activation::SOFTMAX;
}

std::unique_ptr<neurons::ErrorFunction> neurons::ErrorFunction::get_function_by_name(std::string & func_name)
{
    std::istringstream buf(func_name);
    std::istream_iterator<std::string> beg(buf), end;

    std::vector<std::string> tokens(beg, end);

    if (tokens[0] == SIGMOID_CROSS_ENTROPY)
    {
        return std::make_unique<Sigmoid_CrossEntropy>();
    }
    else if (tokens[0] == SOFTMAX_CROSS_ENTROPY)
    {
        return std::make_unique<Softmax_CrossEntropy>();
    }
    else if (tokens[0] == HALF_SQUARE_ERROR)
    {
        return std::make_unique<HalfSquareError>(Activation::get_function_by_name(tokens[1]));
    }
    else if (tokens[0] == NULL_FUNC)
    {
        return nullptr;
    }
    else
    {
        return std::make_unique<Softmax_CrossEntropy>();
    }
}

std::unique_ptr<neurons::Activation> neurons::ErrorFunction::get_act_func() const
{
    return this->m_act_func->clone();
}


neurons::HalfSquareError::HalfSquareError(const std::unique_ptr<Activation> & act_func)
{
    this->m_act_func = act_func->clone();
}

neurons::HalfSquareError::HalfSquareError(Activation * act_func)
{
    this->m_act_func.reset(act_func);
}


std::unique_ptr<neurons::ErrorFunction> neurons::HalfSquareError::clone()
{
    return std::make_unique<neurons::HalfSquareError>(this->m_act_func->clone());
}

double neurons::HalfSquareError::operator()(TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input)
{
    if (target.shape().size() != input.shape().size())
    {
        throw std::invalid_argument(std::string("ErrorFunction: target and pred should be of the same size."));
    }
    // Execute the activation function
    // The activation function may be sigmoid, tanh, relu, etc
    this->m_act_func->operator()(this->m_act, diff, input);

    TMatrix<> l_diff{ target.m_shape };
    lint size = target.m_shape.size();

    double sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        double sub = this->m_act.m_data[i] - target.m_data[i];
        sum += sub * sub;
        
        l_diff.m_data[i] = sub;
    }

    sum /= 2;
    diff = neurons::multiply(diff, l_diff);

    return sum;
}

double neurons::HalfSquareError::operator()(TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input)
{
    double loss = this->operator()(diff, target, input);
    pred = this->m_act;

    return loss;
}

neurons::TMatrix<> & neurons::HalfSquareError::get_activation() const
{
    return this->m_act;
}

std::string neurons::HalfSquareError::to_string() const
{
    return std::string(ErrorFunction::HALF_SQUARE_ERROR + " " + this->m_act_func->to_string());
}


neurons::Sigmoid_CrossEntropy::Sigmoid_CrossEntropy()
{
    this->m_act_func.reset(new Sigmoid);
}

std::unique_ptr<neurons::ErrorFunction> neurons::Sigmoid_CrossEntropy::clone()
{
    return std::make_unique<neurons::Sigmoid_CrossEntropy>();
}

double neurons::Sigmoid_CrossEntropy::operator()(TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input)
{
    if (target.shape().size() != input.shape().size())
    {
        throw std::invalid_argument(std::string("ErrorFunction: target and pred should be of the same size."));
    }

    diff = TMatrix<>{ target.m_shape };


    if (this->m_act.m_shape.size() != target.m_shape.size())
    {
        this->m_act = TMatrix<>{ target.m_shape };
    }
    else if (this->m_act.m_shape != target.m_shape)
    {
        this->m_act.reshape(target.m_shape);
    }


    lint size = target.m_shape.size();

    double sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        double y = 1.0 / (1.0 + exp(input.m_data[i] * (-1)));
        
        this->m_act.m_data[i] = y;
        sum += target.m_data[i] * log(y);
        diff.m_data[i] = y - target.m_data[i];
    }

    sum *= -1;

    return sum;
}

double neurons::Sigmoid_CrossEntropy::operator()(TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input)
{
    double loss = this->operator()(diff, target, input);
    pred = this->m_act;

    return loss;
}


neurons::TMatrix<> & neurons::Sigmoid_CrossEntropy::get_activation() const
{
    return this->m_act;
}

std::string neurons::Sigmoid_CrossEntropy::to_string() const
{
    return ErrorFunction::SIGMOID_CROSS_ENTROPY;
}


neurons::Softmax_CrossEntropy::Softmax_CrossEntropy()
{
    this->m_act_func.reset(new Softmax);
}

std::unique_ptr<neurons::ErrorFunction> neurons::Softmax_CrossEntropy::clone()
{
    return std::make_unique<neurons::Softmax_CrossEntropy>();
}


double neurons::Softmax_CrossEntropy::operator()(TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input)
{
    if (target.shape().size() != input.shape().size())
    {
        throw std::invalid_argument(std::string("ErrorFunction: target and pred should be of the same size."));
    }

    diff = TMatrix<>{ target.m_shape };

    if (this->m_act.m_shape.size() != target.m_shape.size())
    {
        this->m_act = TMatrix<>{ target.m_shape };
    }
    else if (this->m_act.m_shape != target.m_shape)
    {
        this->m_act.reshape(target.m_shape);
    }

    lint size = target.m_shape.size();
    double softmax_sum = 0;

    for (lint i = 0; i < size; ++i)
    {
        this->m_act.m_data[i] = exp(input.m_data[i]);
        softmax_sum += this->m_act.m_data[i];
    }

    double centropy_sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        this->m_act.m_data[i] /= softmax_sum;

        centropy_sum += target.m_data[i] * log(this->m_act.m_data[i]);

        diff.m_data[i] = this->m_act.m_data[i] - target.m_data[i];
    }

    centropy_sum *= -1;

    return centropy_sum;
}

double neurons::Softmax_CrossEntropy::operator()(TMatrix<> & pred, TMatrix<> & diff, const TMatrix<> & target, const TMatrix<> & input)
{
    double loss = this->operator()(diff, target, input);
    pred = this->m_act;

    return loss;
}

neurons::TMatrix<> & neurons::Softmax_CrossEntropy::get_activation() const
{
    return this->m_act;
}

std::string neurons::Softmax_CrossEntropy::to_string() const
{
    return ErrorFunction::SOFTMAX_CROSS_ENTROPY;
}

lint neurons::now_in_seconds()
{
    lint ms =
        std::chrono::duration_cast <std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    
    return ms / 1000;
}

lint neurons::now_in_milliseconds()
{
    return std::chrono::duration_cast <std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

double neurons::gaussian_function(double mu, double sigma, double x)
{
    double x_minus_mu_sq = (x - mu) * (x - mu);

    double left = 1.0 / (sqrt(2 * M_PI) * sigma);
    double right = exp(x_minus_mu_sq * (-1) / (2 * sigma * sigma));

    return left * right;
}


