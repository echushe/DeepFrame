/********************************************************************

Programmed by Chunnan Sheng

This class is definition of a multi-layer NN of online learning

To Xu Ning:
    You can copy this code to Batch_multi_layer_nn and add
    batch learning algorithms to it.

*********************************************************************/

#pragma once
#include "NN.h"
#include "FCNN_layer.h"
#include <iostream>


class Multi_layer_nn : public NN
{
private:

public:
    Multi_layer_nn(
        double l_rate,
        lint batch_size,
        lint threads,
        lint steps,
        lint epoch_size,
        lint secs_allowed,
        const dataset::Dataset &d_set);

    // Copies and moves are prohibited

    Multi_layer_nn(const Multi_layer_nn & other) = delete;
    Multi_layer_nn(Multi_layer_nn && other) = delete;
    Multi_layer_nn & operator = (const Multi_layer_nn & other) = delete;
    Multi_layer_nn & operator = (Multi_layer_nn && other) = delete;

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

