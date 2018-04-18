/********************************************************************

Programmed by Chunnan Sheng

*********************************************************************/

#pragma once
#include "NN.h"
#include "Convolution.h"
#include "FCNN_layer.h"
#include "CNN_layer.h"
#include <iostream>


class Conv_NN : public NN
{
private:

    // std::vector<neurons::CNN_layer> m_conv_layers;
    // neurons::FCNN_layer m_nn_layer;

public:
    Conv_NN(
        double l_rate,
        lint batch_size,
        lint threads,
        lint steps,
        lint epoch_size,
        lint secs_allowed,
        const dataset::Dataset &d_set);

    // Copies and moves are prohibited

    Conv_NN(const Conv_NN & other) = delete;
    Conv_NN(Conv_NN && other) = delete;
    Conv_NN & operator = (const Conv_NN & other) = delete;
    Conv_NN & operator = (Conv_NN && other) = delete;

public:

    virtual void print_layers(std::ostream & os) const;

private:

    virtual std::vector<neurons::TMatrix<>> test(
        const std::vector<neurons::TMatrix<>> & inputs,
        const std::vector<neurons::TMatrix<>> & targets,
        lint thread_id);

    virtual std::vector<neurons::TMatrix<>> optimise(
        const std::vector<neurons::TMatrix<>> & inputs,
        const std::vector<neurons::TMatrix<>> & targets,
        lint thread_id);

};



