/********************************************************************

Programmed by Chunnan Sheng

*********************************************************************/
#pragma once
#include "Functions.h"
#include "Traditional_NN_layer.h"

namespace neurons
{
    /*
    This is definition of a single layer of a fully connected neural network.
    FCNN stands for "Fully Connected Neural Networks"
    */
    class FCNN_layer : public Traditional_NN_layer
    {
    public:
        // Default constructor does almost nothing here.
        // However, default constructor is essential for containers like std::vector, std::list, std::map, etc.
        // Behaviors of operations on a FCNN_layer that is created by default constructor would be undefined.
        // You should assign a valid FCNN_layer instance to the object created by default constructor before
        // using it.
        FCNN_layer();

        // This is the Constructor to create a functional FCNN layer
        FCNN_layer(
            lint input_size,     // Size of input
            lint output_size,    // Size of output
            lint threads,        // Number of threads while training this layer
            neurons::Activation *act_func,    // Activation function of this layer
            neurons::ErrorFunction *err_func = nullptr   // Cost function or error function of this layer
        );


        FCNN_layer(const FCNN_layer & other);

        FCNN_layer(FCNN_layer && other);

        FCNN_layer & operator = (const FCNN_layer & other);

        FCNN_layer & operator = (FCNN_layer && other);

        virtual Shape output_shape() const;
    };

    class FCNN_layer_op : public Traditional_NN_layer_op
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

        //----------------------------
        // Copy and move operations
        //----------------------------

        FCNN_layer_op(const FCNN_layer_op & other);

        FCNN_layer_op(FCNN_layer_op && other);

        FCNN_layer_op & operator = (const FCNN_layer_op & other);

        FCNN_layer_op & operator = (FCNN_layer_op && other);

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