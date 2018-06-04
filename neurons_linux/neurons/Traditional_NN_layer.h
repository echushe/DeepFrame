#pragma once
#include "TMatrix.h"
#include "Functions.h"
#include "NN_layer.h"

namespace neurons
{
    class FCNN_layer;
    class CNN_layer;
    class Traditional_NN_layer_op;

    class Traditional_NN_layer : public NN_layer
    {
    protected:
        double m_mmt_rate;

        TMatrix<> m_w;
        TMatrix<> m_b;

        TMatrix<> m_w_mmt;
        TMatrix<> m_b_mmt;

        // The pointer of activation function (logist, softmax, etc)
        std::unique_ptr<Activation> m_act_func;
        // The pointer of error function (sigmoid_crossentropy, softmax_crossentropy, etc)
        std::unique_ptr<ErrorFunction> m_err_func;

    public:
        static std::string from_binary_data(
            char * binary_data, lint & data_size, TMatrix<> & w, TMatrix<> & b,
            std::unique_ptr<Activation> & act_func, std::unique_ptr<ErrorFunction> & err_func,
            char *& residual_data, lint & residual_len
        );

        Traditional_NN_layer();

        Traditional_NN_layer(double mmt_rate, lint threads, const TMatrix<> & w, const TMatrix<> & b,
            std::unique_ptr<Activation> & act_func, std::unique_ptr<ErrorFunction> & err_func);

        Traditional_NN_layer(
            double mmt_rate, const Shape &w_sh, const Shape &b_sh, lint threads, 
            Activation *act_func, ErrorFunction *err_func = nullptr);

        Traditional_NN_layer(const Traditional_NN_layer & other);

        Traditional_NN_layer(Traditional_NN_layer && other);

        Traditional_NN_layer & operator = (const Traditional_NN_layer & other);

        Traditional_NN_layer & operator = (Traditional_NN_layer && other);

        TMatrix<> weights() const;

        TMatrix<> bias() const;

        virtual double commit_training();

        virtual double commit_testing();

        virtual Shape output_shape() const = 0;

        virtual std::unique_ptr<char[]> to_binary_data(lint & data_size) const;
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
        // Forward propagation via a single training sample can be recognized as
        // forward propagation via a batch of training samples where size of this batch is 1
        //--------------------------------------------

        virtual TMatrix<> forward_propagate(const TMatrix<> &input);

        virtual TMatrix<> forward_propagate(const TMatrix<> &input, const TMatrix<> &target);

        //--------------------------------------------
        // Back propagation
        // Back propagation via a single training sample can be recognized as
        // back propagation via a batch of training samples where size of this batch is 1
        //--------------------------------------------

        virtual TMatrix<> back_propagate(double l_rate, const TMatrix<> & E_to_y_diff);

        virtual TMatrix<> back_propagate(double l_rate);

        TMatrix<>& get_weight_gradient() const;

        TMatrix<>& get_bias_gradient() const;

        void update_w_and_b(const TMatrix<> &w, const TMatrix<> &b);
    };
}


