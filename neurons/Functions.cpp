#include "Functions.h"
#include <math.h>
#include <iostream>
#include <chrono>

std::unique_ptr<neurons::Activation> neurons::Linear::clone()
{
    return std::make_unique<neurons::Linear>();
}

void neurons::Linear::operator () (Matrix & output, Matrix & diff, const Matrix & in)
{
    Matrix l_output{ in.m_shape };
    Matrix l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        l_output.m_data[i] = in.m_data[i];
        l_diff.m_data[i] = 1;
    }

    output = std::move(l_output);
    diff = std::move(l_diff);
}


std::unique_ptr<neurons::Activation> neurons::Sigmoid::clone()
{
    return std::make_unique<neurons::Sigmoid>();
}

void neurons::Sigmoid::operator () (Matrix & output, Matrix & diff, const Matrix & in)
{
    Matrix l_output{ in.m_shape };
    Matrix l_diff{ in.m_shape };
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


std::unique_ptr<neurons::Activation> neurons::Tanh::clone()
{
    return std::make_unique<neurons::Tanh>();
}

void neurons::Tanh::operator () (Matrix & output, Matrix & diff, const Matrix & in)
{
    Matrix l_output{ in.m_shape };
    Matrix l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        double y = tanh(in.m_data[i]);
        l_output.m_data[i] = y;
        l_diff.m_data[i] = (1 + y) * (1 - y);
    }

    output = std::move(l_output);
    diff = std::move(l_diff);
}


std::unique_ptr<neurons::Activation> neurons::Relu::clone()
{
    return std::make_unique<neurons::Relu>();
}

void neurons::Relu::operator () (Matrix & output, Matrix & diff, const Matrix & in)
{
    Matrix l_output{ in.m_shape };
    Matrix l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        double x = in.m_data[i];

        if (x >= 0)
        {
            l_output.m_data[i] = x;
            l_diff.m_data[i] = 1;
        }
        else
        {
            l_output.m_data[i] = 0;
            l_diff.m_data[i] = 0;
        }
    }

    output = std::move(l_output);
    diff = std::move(l_diff);
}


std::unique_ptr<neurons::Activation> neurons::Softmax::clone()
{
    return std::make_unique<neurons::Softmax>();
}

void neurons::Softmax::operator () (Matrix & output, Matrix & diff, const Matrix & in)
{
    Matrix l_output{ in.m_shape };
    Matrix l_diff{ in.m_shape };
    lint size = in.m_shape.size();

    double sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        l_output.m_data[i] = exp(in.m_data[i]);
        sum += l_output.m_data[i];
    }

    for (lint i = 0; i < size; ++i)
    {
        l_output.m_data[i] /= sum;
        l_diff.m_data[i] = l_output.m_data[i] * (1 - l_output.m_data[i]);
    }

    output = std::move(l_output);
    diff = std::move(l_diff);
}


std::unique_ptr<neurons::ErrorFunction> neurons::HalfSquareError::clone()
{
    return std::make_unique<neurons::HalfSquareError>();
}

double neurons::HalfSquareError::operator()(Matrix & diff, const Matrix & target, const Matrix & input)
{
    if (target.shape().size() != input.shape().size())
    {
        throw std::invalid_argument(std::string("ErrorFunction: target and pred should be of the same size."));
    }

    Matrix l_diff{ target.m_shape };
    lint size = target.m_shape.size();

    double sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        double sub = input.m_data[i] - target.m_data[i];
        sum += sub * sub;
        
        l_diff.m_data[i] = sub;
    }

    sum /= 2;
    diff = std::move(l_diff);

    return sum;
}

double neurons::HalfSquareError::operator()(Matrix & pred, Matrix & diff, const Matrix & target, const Matrix & input)
{
    double loss = this->operator()(diff, target, input);
    pred = input;

    return loss;
}


std::unique_ptr<neurons::ErrorFunction> neurons::Sigmoid_CrossEntropy::clone()
{
    return std::make_unique<neurons::Sigmoid_CrossEntropy>();
}

double neurons::Sigmoid_CrossEntropy::operator()(Matrix & diff, const Matrix & target, const Matrix & input)
{
    if (target.shape().size() != input.shape().size())
    {
        throw std::invalid_argument(std::string("ErrorFunction: target and pred should be of the same size."));
    }

    Matrix l_diff{ target.m_shape };


    if (this->m_sigmoid.m_shape.size() != target.m_shape.size())
    {
        this->m_sigmoid = Matrix{ target.m_shape };
    }
    else if (this->m_sigmoid.m_shape != target.m_shape)
    {
        this->m_sigmoid.reshape(target.m_shape);
    }


    lint size = target.m_shape.size();

    double sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        double y = 1.0 / (1.0 + exp(input.m_data[i] * (-1)));
        
        this->m_sigmoid.m_data[i] = y;
        sum += target.m_data[i] * log(y);
        l_diff.m_data[i] = y - target.m_data[i];
    }

    sum *= -1;
    diff = std::move(l_diff);

    return sum;
}

double neurons::Sigmoid_CrossEntropy::operator()(Matrix & pred, Matrix & diff, const Matrix & target, const Matrix & input)
{
    double loss = this->operator()(diff, target, input);
    pred = this->get_sigmoid();

    return loss;
}


neurons::Matrix & neurons::Sigmoid_CrossEntropy::get_sigmoid() const
{
    return this->m_sigmoid;
}


std::unique_ptr<neurons::ErrorFunction> neurons::Softmax_CrossEntropy::clone()
{
    return std::make_unique<neurons::Softmax_CrossEntropy>();
}


double neurons::Softmax_CrossEntropy::operator()(Matrix & diff, const Matrix & target, const Matrix & input)
{
    if (target.shape().size() != input.shape().size())
    {
        throw std::invalid_argument(std::string("ErrorFunction: target and pred should be of the same size."));
    }

    Matrix l_diff{ target.m_shape };

    if (this->m_softmax.m_shape.size() != target.m_shape.size())
    {
        this->m_softmax = Matrix{ target.m_shape };
    }
    else if (this->m_softmax.m_shape != target.m_shape)
    {
        this->m_softmax.reshape(target.m_shape);
    }

    lint size = target.m_shape.size();
    double softmax_sum = 0;

    for (lint i = 0; i < size; ++i)
    {
        this->m_softmax.m_data[i] = exp(input.m_data[i]);
        softmax_sum += this->m_softmax.m_data[i];
    }

    double centropy_sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        this->m_softmax.m_data[i] /= softmax_sum;
        centropy_sum += target.m_data[i] * log(this->m_softmax.m_data[i]);
        l_diff.m_data[i] = this->m_softmax.m_data[i] - target.m_data[i];
    }

    centropy_sum *= -1;
    diff = std::move(l_diff);

    return centropy_sum;
}

double neurons::Softmax_CrossEntropy::operator()(Matrix & pred, Matrix & diff, const Matrix & target, const Matrix & input)
{
    double loss = this->operator()(diff, target, input);
    pred = this->get_SoftMax();

    return loss;
}

neurons::Matrix & neurons::Softmax_CrossEntropy::get_SoftMax() const
{
    return this->m_softmax;
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
