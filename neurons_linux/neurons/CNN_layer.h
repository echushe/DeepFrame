/********************************************************************

Programmed by Chunnan Sheng

*********************************************************************/
#pragma once
#include "Functions.h"
#include "Traditional_NN_layer.h"
#include "Convolution.h"

namespace neurons
{
    /*
    This is class definition of a convolutional neural network layer
    */
    class CNN_layer : public Traditional_NN_layer
    {
    private:
        // A module of 2-dimensional convolution algorithm
        Conv_2d m_conv2d;

    public:
        static std::string from_binary_data(
            char * binary_data, lint & data_size, TMatrix<> & w, TMatrix<> & b,
            lint & stride, lint & padding,
            std::unique_ptr<Activation> & act_func, std::unique_ptr<ErrorFunction> & err_func,
            char *& residual_data, lint & residual_len
        );

        // Default constructor does almost nothing here.
        // However, default constructor is essential for containers like std::vector, std::list, std::map, etc.
        // Behaviors of operations on a CNN_layer that is created by default constructor would be undefined.
        // You should assign a valid CNN_layer instance to the object created by default constructor before
        // using it.
        CNN_layer();

        // This is the Constructor to create a functional CNN layer
        CNN_layer(
            double mmt_rate,
            lint rows,  // Number of rows for one input sample
            lint cols,  // Number of columns for one input sample
            lint chls,  // Number of channels (depth) for one input sample
            lint filters,  // Number of filters (kernels)
            lint filter_rows,  // Number of rows for one kernel
            lint filter_cols,  // Number of columns for one kernel
            lint stride,  // Stride size (>= 1) 
            lint padding,  // Zero padding size (>= 0)
            lint threads,  // Number of threads while training of the network layer
            neurons::Activation *act_func,  // Activation function of this layer
            neurons::ErrorFunction *err_func = nullptr // Cost function or error function of this layer
        );

        CNN_layer(
            double mmt_rate,
            lint rows, lint cols, lint chls, lint stride, lint padding, lint threads,
            const TMatrix<> & w, const TMatrix<> & b,
            std::unique_ptr<Activation> & act_func, std::unique_ptr<ErrorFunction> & err_func);

        CNN_layer(const CNN_layer & other);

        CNN_layer(CNN_layer && other);

        CNN_layer & operator = (const CNN_layer & other);

        CNN_layer & operator = (CNN_layer && other);

        Shape output_shape() const;

        virtual std::string nn_type() const { return NN_layer::CNN; }

        virtual std::unique_ptr<char[]> to_binary_data(lint & data_size) const;
    };

    class CNN_layer_op : public Traditional_NN_layer_op
    {
    private:

        std::vector<TMatrix<>> m_conv_to_x_diffs;
        std::vector<TMatrix<>> m_conv_to_w_diffs;
        std::vector<TMatrix<>> m_act_diffs;

        Conv_2d m_conv2d;

    public:

        CNN_layer_op();

        CNN_layer_op(
            const Conv_2d & conv2d,
            const TMatrix<> &w,
            const TMatrix<> &b,
            const std::unique_ptr<Activation> &act_func,
            const std::unique_ptr<ErrorFunction> &err_func);

        CNN_layer_op(const CNN_layer_op & other);

        CNN_layer_op(CNN_layer_op && other);

        CNN_layer_op & operator = (const CNN_layer_op & other);

        CNN_layer_op & operator = (CNN_layer_op && other);

        //--------------------------------------------
        // Forward propagation via batch learning
        //--------------------------------------------

        virtual std::vector<TMatrix<>> batch_forward_propagate(const std::vector<TMatrix<>> & inputs);

        virtual std::vector<TMatrix<>> batch_forward_propagate(
            const std::vector<TMatrix<>> & inputs, const std::vector<TMatrix<>> & targets);

        //--------------------------------------------
        // Back propagation via batch learning
        //--------------------------------------------

        virtual std::vector<TMatrix<>> batch_back_propagate(double l_rate, const std::vector<TMatrix<>> & E_to_y_diffs);

        virtual std::vector<TMatrix<>> batch_back_propagate(double l_rate);

        virtual Shape output_shape() const;
    };
}

