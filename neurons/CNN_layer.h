#pragma once
#include "Functions.h"
#include "Traditional_NN_layer.h"
#include "Convolution.h"

namespace neurons
{
    class CNN_layer : public Traditional_NN_layer
    {
    private:
        Conv_2d m_conv2d;

    public:
        CNN_layer();

        CNN_layer(
            lint rows,
            lint cols,
            lint chls,
            lint filters,
            lint filter_rows,
            lint filter_cols,
            lint stride,
            lint padding,
            lint threads,
            neurons::Activation *act_func,
            neurons::ErrorFunction *err_func = nullptr);

        CNN_layer(const CNN_layer & other);

        CNN_layer(CNN_layer && other);

        CNN_layer & operator = (const CNN_layer & other);

        CNN_layer & operator = (CNN_layer && other);

        Shape output_shape() const;
    };

    class CNN_layer_op : public Traditional_NN_layer_op
    {
    private:

        std::vector<Matrix> m_conv_to_x_diffs;
        std::vector<Matrix> m_conv_to_w_diffs;
        std::vector<Matrix> m_act_diffs;

        Conv_2d m_conv2d;

    public:

        CNN_layer_op();

        CNN_layer_op(
            const Conv_2d & conv2d,
            const Matrix &w,
            const Matrix &b,
            const std::unique_ptr<Activation> &act_func,
            const std::unique_ptr<ErrorFunction> &err_func);

        CNN_layer_op(const CNN_layer_op & other);

        CNN_layer_op(CNN_layer_op && other);

        CNN_layer_op & operator = (const CNN_layer_op & other);

        CNN_layer_op & operator = (CNN_layer_op && other);

        //--------------------------------------------
        // Forward propagation via batch learning
        //--------------------------------------------

        virtual std::vector<Matrix> batch_forward_propagate(const std::vector<Matrix> & inputs);

        virtual std::vector<Matrix> batch_forward_propagate(
            const std::vector<Matrix> & inputs, const std::vector<Matrix> & targets);

        //--------------------------------------------
        // Backward propagation via batch learning
        //--------------------------------------------

        virtual std::vector<Matrix> batch_backward_propagate(double l_rate, const std::vector<Matrix> & E_to_y_diffs);

        virtual std::vector<Matrix> batch_backward_propagate(double l_rate);

        virtual Shape output_shape() const;
    };
}

