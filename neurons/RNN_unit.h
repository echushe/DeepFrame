#pragma once
#include "Functions.h"
#include <deque>

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

        // The previous output data (input)
        Matrix m_old_y;

        // The pointer of activation function (logist, softmax, etc)
        std::unique_ptr<Activation> m_act_func;

        // The pointer of error function (sigmoid_crossentropy, softmax_crossentropy, etc)
        std::unique_ptr<ErrorFunction> m_err_func;

    public:
        RNN_unit();

        RNN_unit(
            lint input_size,
            lint output_size,
            lint bptt_len,
            neurons::Activation *act_func,
            neurons::ErrorFunction *err_func = nullptr);

        RNN_unit(
            lint input_size,
            lint output_size,
            lint bptt_len,
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

        Matrix forward_propagate(const Matrix & input);

        Matrix forward_propagate(double & loss, const Matrix & input, const Matrix & targets);

        std::vector<Matrix> backward_propagate_through_time(double l_rate, const Matrix & E_to_y_diff, lint len = 0);

        std::vector<Matrix> backward_propagate_through_time(double l_rate, lint len = 0);

        void forget_all();
    
    private:

        //-----------------------------------------------------
        // Following definitions are for
        // back propagation and back propagation through time
        //-----------------------------------------------------

        lint m_bptt_len;
        //

        struct cache_item
        {
            Matrix m_y_in;
            Matrix m_x;
            Matrix m_act_diff;

            cache_item(const Matrix & y_in, const Matrix & x, const Matrix & act_diff)
                :
                m_y_in { y_in },
                m_x { x },
                m_act_diff{ act_diff }
            {}
        };

        std::deque<cache_item> m_cache_for_bptt;

    private:
        size_t m_counter;
    };

}