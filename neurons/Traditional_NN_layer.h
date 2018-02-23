#pragma once
#include "Matrix.h"
#include "Functions.h"
#include "NN_layer.h"

namespace neurons
{
    class Traditional_NN_layer_op;

    class Traditional_NN_layer : public NN_layer
    {
    protected:
        Matrix m_w;
        Matrix m_b;

        // The pointer of activation function (logist, softmax, etc)
        std::unique_ptr<Activation> m_act_func;
        // The pointer of error function (sigmoid_crossentropy, softmax_crossentropy, etc)
        std::unique_ptr<ErrorFunction> m_err_func;

    public:
        Traditional_NN_layer();
        Traditional_NN_layer(const Shape &w_sh, const Shape &b_sh, lint threads, Activation *act_func, ErrorFunction *err_func = nullptr);

        Traditional_NN_layer(const Traditional_NN_layer & other);

        Traditional_NN_layer(Traditional_NN_layer && other);

        Traditional_NN_layer & operator = (const Traditional_NN_layer & other);

        Traditional_NN_layer & operator = (Traditional_NN_layer && other);

        virtual double commit_training();

        virtual double commit_testing();

        virtual Shape output_shape() const = 0;
    };

    class Traditional_NN_layer_op : public NN_layer_op
    {
    protected:
        Matrix m_w;
        Matrix m_b;

        // The pointer of activation function (logist, softmax, etc)
        std::unique_ptr<Activation> m_act_func;
        // The pointer of error function (sigmoid_crossentropy, softmax_crossentropy, etc)
        std::unique_ptr<ErrorFunction> m_err_func;

        mutable Matrix m_w_gradient;
        mutable Matrix m_b_gradient;

        // Differentiation of the activation function dy/dz
        // in which y is output of activation, z is x * w  + b
        std::vector<Matrix> m_act_diffs;

    public:
        Traditional_NN_layer_op() {}

        Traditional_NN_layer_op(
            const Matrix &w,
            const Matrix &b,
            const std::unique_ptr<Activation> &act_func,
            const std::unique_ptr<ErrorFunction> &err_func);

        Traditional_NN_layer_op(const Traditional_NN_layer_op & other);

        Traditional_NN_layer_op(Traditional_NN_layer_op && other);

        Traditional_NN_layer_op & operator = (const Traditional_NN_layer_op & other);

        Traditional_NN_layer_op & operator = (Traditional_NN_layer_op && other);

    public:

        //--------------------------------------------
        // Forward propagation
        //--------------------------------------------

        virtual Matrix forward_propagate(const Matrix &input);

        virtual Matrix forward_propagate(const Matrix &input, const Matrix &target);

        //--------------------------------------------
        // Backward propagation
        //--------------------------------------------

        virtual Matrix back_propagate(double l_rate, const Matrix & E_to_y_diff);

        virtual Matrix back_propagate(double l_rate);

        Matrix& get_weight_gradient() const;

        Matrix& get_bias_gradient() const;

        void update_w_and_b(const Matrix &w, const Matrix &b);
    };
}


