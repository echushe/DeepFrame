#pragma once
#include "Functions.h"


namespace neurons
{

    /*
    This is definition of a single RNN (Recurrent neural network) unit.
    A single RNN unit itself can become a simple RNN.
    Multiple RNN units can construct a more implicated RNN block,
    such as LSTM (long short term memory) or GRU (Gated recurrent unit)
    */
    class RNN_unit
    {
    private:
        // The weights which will multiply with context layer (old output layer)
        Matrix m_u;

        // The weights which will multiply with input
        Matrix m_w;

        Matrix m_b;

        // The previous output data
        Matrix m_old_y_bak;

        // The previous output data (input)
        Matrix m_old_y_in;

        // The pointer of activation function (logist, softmax, etc)
        std::unique_ptr<Activation> m_act_func;

        // The pointer of error function (sigmoid_crossentropy, softmax_crossentropy, etc)
        std::unique_ptr<ErrorFunction> m_err_func;

        // The input data (cached)
        Matrix m_x;

        // The output data
        Matrix m_y;

        // Differentiation of the activation function dy/dz
        // in which y is output of activation, z is x * w  + b
        Matrix m_act_diffs;

    public:
        RNN_unit();

        RNN_unit(
            lint input_size,
            lint output_size,
            neurons::Activation *act_func,
            neurons::ErrorFunction *err_func = nullptr);

        RNN_unit(
            lint input_size,
            lint output_size,
            std::unique_ptr<neurons::Activation> &act_func,
            std::unique_ptr<neurons::ErrorFunction> &err_func);


        //----------------------------
        // Copy and move operations
        //----------------------------

        RNN_unit(const RNN_unit & other);

        RNN_unit(RNN_unit && other);

        RNN_unit & operator = (const RNN_unit & other);

        RNN_unit & operator = (RNN_unit && other);



        Shape output_shape() const;

        //---------------------------
        // other functions
        //---------------------------

        Matrix forward_propagate(const Matrix &input);

        Matrix forward_propagate(double &loss, const Matrix &input, const Matrix &targets);

        Matrix backward_propagate(double l_rate, const Matrix &E_to_y_diffs);

        Matrix backward_propagate(double l_rate);
    };

}