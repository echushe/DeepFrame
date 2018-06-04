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
            double mmt_rate,
            lint input_size,     // Size of input
            lint output_size,    // Size of output
            lint threads,        // Number of threads while training this layer
            neurons::Activation *act_func,    // Activation function of this layer
            neurons::ErrorFunction *err_func = nullptr   // Cost function or error function of this layer
        );

        // This is the constructor to create a functional FCNN layer from weight, bias and functions directly 
        FCNN_layer(double mmt_rate, lint threads, TMatrix<> & w, TMatrix<> & b,
            std::unique_ptr<Activation> & act_func, std::unique_ptr<ErrorFunction> & err_func);


        FCNN_layer(const FCNN_layer & other);

        FCNN_layer(FCNN_layer && other);

        FCNN_layer & operator = (const FCNN_layer & other);

        FCNN_layer & operator = (FCNN_layer && other);

        virtual Shape output_shape() const;

        virtual std::string nn_type() const { return NN_layer::FCNN; }
    };

    class FCNN_layer_op : public Traditional_NN_layer_op
    {
    private:
        // The input data
        std::vector<TMatrix<>> m_x;

    public:
        FCNN_layer_op();

        FCNN_layer_op(
            const TMatrix<> &w,
            const TMatrix<> &b,
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

        virtual std::vector<TMatrix<>> batch_forward_propagate(const std::vector<TMatrix<>> & inputs);

        virtual std::vector<TMatrix<>> batch_forward_propagate(
            const std::vector<TMatrix<>> & inputs, const std::vector<TMatrix<>> & targets);

        //--------------------------------------------
        // Backward propagation via batch learning
        //--------------------------------------------

        virtual std::vector<TMatrix<>> batch_back_propagate(double l_rate, const std::vector<TMatrix<>> & E_to_y_diffs);

        virtual std::vector<TMatrix<>> batch_back_propagate(double l_rate);

        virtual Shape output_shape() const;

    };

}