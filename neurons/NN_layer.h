#pragma once
#include "Matrix.h"
#include "Functions.h"

namespace neurons
{
    class NN_layer_op;

    class NN_layer
    {
    protected:
        Matrix m_w;
        Matrix m_b;

        // The pointer of activation function (logist, softmax, etc)
        std::unique_ptr<Activation> m_act_func;
        // The pointer of error function (sigmoid_crossentropy, softmax_crossentropy, etc)
        std::unique_ptr<ErrorFunction> m_err_func;

        mutable std::vector<std::shared_ptr<NN_layer_op>> m_ops;

    public:
        NN_layer();
        NN_layer(const Shape &w_sh, const Shape &b_sh, lint threads, Activation *act_func, ErrorFunction *err_func = nullptr);

        NN_layer(const NN_layer & other);

        NN_layer(NN_layer && other);

        NN_layer & operator = (const NN_layer & other);

        NN_layer & operator = (NN_layer && other);

        std::vector<std::shared_ptr<NN_layer_op>>& operation_instances() const;

        double commit_training();

        double commit_testing();

        virtual Shape output_shape() const = 0;
    };

    class NN_layer_op
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

        // loss calculated in the training
        double m_loss;

        // Differentiation of the activation function dy/dz
        // in which y is output of activation, z is x * w  + b
        std::vector<Matrix> m_act_diffs;

    public:
        NN_layer_op() {}

        NN_layer_op(
            const Matrix &w,
            const Matrix &b,
            const std::unique_ptr<Activation> &act_func,
            const std::unique_ptr<ErrorFunction> &err_func);

        NN_layer_op(const NN_layer_op & other);

        NN_layer_op(NN_layer_op && other);

        NN_layer_op & operator = (const NN_layer_op & other);

        NN_layer_op & operator = (NN_layer_op && other);

    public:

        virtual std::vector<Matrix> forward_propagate(const std::vector<Matrix> & inputs) = 0;

        virtual std::vector<Matrix> forward_propagate(
            const std::vector<Matrix> & inputs, const std::vector<Matrix> & targets) = 0;

        virtual std::vector<Matrix> backward_propagate(double l_rate, const std::vector<Matrix> & E_to_y_diffs) = 0;

        virtual std::vector<Matrix> backward_propagate(double l_rate) = 0;

        virtual Shape output_shape() const = 0;

        Matrix& get_weight_gradient() const;

        Matrix& get_bias_gradient() const;

        double get_loss() const;

        void update_w_and_b(const Matrix &w, const Matrix &b);
    };
}

