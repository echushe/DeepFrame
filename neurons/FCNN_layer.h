#pragma once
#include "Functions.h"
#include "NN_layer.h"

namespace neurons
{
    /*
    This is definition of a single layer of a fully connected neural network
    */
    class FCNN_layer : public NN_layer
    {
    public:
        FCNN_layer();

        FCNN_layer(
            lint input_size,
            lint output_size,
            lint threads,
            neurons::Activation *act_func,
            neurons::ErrorFunction *err_func = nullptr);


        FCNN_layer(const FCNN_layer & other);

        FCNN_layer(FCNN_layer && other);

        FCNN_layer & operator = (const FCNN_layer & other);

        FCNN_layer & operator = (FCNN_layer && other);

        virtual Shape output_shape() const;
    };

    class FCNN_layer_op : public NN_layer_op
    {
    private:
        // The input data
        std::vector<Matrix> m_x;

    public:
        FCNN_layer_op();

        FCNN_layer_op(
            const Matrix &w,
            const Matrix &b,
            const std::unique_ptr<Activation> &act_func,
            const std::unique_ptr<ErrorFunction> &err_func);

        FCNN_layer_op(const FCNN_layer_op & other);

        FCNN_layer_op(FCNN_layer_op && other);

        FCNN_layer_op & operator = (const FCNN_layer_op & other);

        FCNN_layer_op & operator = (FCNN_layer_op && other);

        virtual std::vector<Matrix> forward_propagate(const std::vector<Matrix> & inputs);

        virtual std::vector<Matrix> forward_propagate(
            const std::vector<Matrix> & inputs, const std::vector<Matrix> & targets);

        virtual std::vector<Matrix> backward_propagate(double l_rate, const std::vector<Matrix> & E_to_y_diffs);

        virtual std::vector<Matrix> backward_propagate(double l_rate);

        virtual Shape output_shape() const;

    };

}