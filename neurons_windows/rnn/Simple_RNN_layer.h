#pragma once
#include "Functions.h"
#include "NN_layer.h"
#include "RNN_unit.h"

namespace neurons
{
    /*
    This is definition of a single layer of a fully connected neural network
    */
    class Simple_RNN_layer : public NN_layer
    {
    private:

        lint m_output_size;
        // The pointer of activation function (logist, softmax, etc)
        std::unique_ptr<Activation> m_act_func;
        // The pointer of error function (sigmoid_crossentropy, softmax_crossentropy, etc)
        std::unique_ptr<ErrorFunction> m_err_func;

    public:
        Simple_RNN_layer();

        Simple_RNN_layer(
            lint input_size,
            lint output_size,
            lint bptt_len,
            lint threads,
            neurons::Activation *act_func,
            neurons::ErrorFunction *err_func = nullptr);


        Simple_RNN_layer(const Simple_RNN_layer & other);

        Simple_RNN_layer(Simple_RNN_layer && other);

        Simple_RNN_layer & operator = (const Simple_RNN_layer & other);

        Simple_RNN_layer & operator = (Simple_RNN_layer && other);

        virtual Shape output_shape() const;

        virtual std::unique_ptr<char[]> to_binary_data(lint & data_size) const;

        virtual std::string nn_type() const { return NN_layer::RNN; }
    };

    class Simple_RNN_layer_op : public NN_layer_op
    {
    private:
        RNN_unit m_rnn;
        size_t m_samples;

    public:
        Simple_RNN_layer_op();

        Simple_RNN_layer_op(
            lint input_size,
            lint output_size,
            lint bptt_len,
            const std::unique_ptr<Activation> &act_func,
            const std::unique_ptr<ErrorFunction> &err_func);

        //----------------------------
        // Copy and move operations
        //----------------------------

        Simple_RNN_layer_op(const Simple_RNN_layer_op & other);

        Simple_RNN_layer_op(Simple_RNN_layer_op && other);

        Simple_RNN_layer_op & operator = (const Simple_RNN_layer_op & other);

        Simple_RNN_layer_op & operator = (Simple_RNN_layer_op && other);

        virtual void forget_all();

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

