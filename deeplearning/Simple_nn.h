/********************************************************************

Programmed by Chunnan Sheng

A simple NN without any hidden layers

*********************************************************************/

#pragma once
#include "NN.h"
#include "FCNN_layer.h"
#include <iostream>

class Simple_nn : public NN
{
private:

public:
    Simple_nn(
        double l_rate,
        lint batch_size,
        lint threads,
        lint steps,
        lint epoch_size,
        lint secs_allowed,
        const dataset::Dataset &d_set);

    Simple_nn(const Simple_nn & other) = delete;
    Simple_nn(Simple_nn && other) = delete;
    Simple_nn & operator = (const Simple_nn & other) = delete;
    Simple_nn & operator = (Simple_nn && other) = delete;

public:

    virtual void print_layers(std::ostream & os) const;

private:

    virtual std::vector<neurons::Matrix> foward_propagate(
        const std::vector<neurons::Matrix> & inputs,
        const std::vector<neurons::Matrix> & targets,
        lint thread_id);

    virtual std::vector<neurons::Matrix> gradient_descent(
        const std::vector<neurons::Matrix> & inputs,
        const std::vector<neurons::Matrix> & targets,
        lint thread_id);

};


