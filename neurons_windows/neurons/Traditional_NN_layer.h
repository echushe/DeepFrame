#pragma once
#include "TMatrix.h"
#include "Functions.h"
#include "NN_layer.h"

namespace neurons
{
    class Traditional_NN_layer_op;

    class Traditional_NN_layer : public NN_layer
    {
    protected:
        TMatrix<> m_w;
        TMatrix<> m_b;

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
        TMatrix<> m_w;
        TMatrix<> m_b;

        // The pointer of activation function (logist, softmax, etc)
        std::unique_ptr<Activation> m_act_func;
        // The pointer of error function (sigmoid_crossentropy, softmax_crossentropy, etc)
        std::unique_ptr<ErrorFunction> m_err_func;

        mutable TMatrix<> m_w_gradient;
        mutable TMatrix<> m_b_gradient;

        // Differentiation of the activation function dy/dz
        // in which y is output of activation, z is x * w  + b
        std::vector<TMatrix<>> m_act_diffs;

    public:
        Traditional_NN_layer_op() {}

        Traditional_NN_layer_op(
            const TMatrix<> &w,
            const TMatrix<> &b,
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

        virtual TMatrix<> forward_propagate(const TMatrix<> &input);

        virtual TMatrix<> forward_propagate(const TMatrix<> &input, const TMatrix<> &target);

        //--------------------------------------------
        // Backward propagation
        //--------------------------------------------

        virtual TMatrix<> back_propagate(double l_rate, const TMatrix<> & E_to_y_diff);

        virtual TMatrix<> back_propagate(double l_rate);

        TMatrix<>& get_weight_gradient() const;

        TMatrix<>& get_bias_gradient() const;

        void update_w_and_b(const TMatrix<> &w, const TMatrix<> &b);
    };
}


