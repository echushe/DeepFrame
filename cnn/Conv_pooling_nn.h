#pragma once
/********************************************************************

Programmed by Chunnan Sheng

*********************************************************************/

#pragma once
#include "NN.h"
#include "Convolution.h"
#include "FCNN_layer.h"
#include "CNN_layer.h"
#include "Pooling.h"

class Conv_pooling_nn : public NN
{
private:

    //std::vector<neurons::CNN_layer> m_conv_layers;
    std::vector<neurons::Pooling_layer> m_pooling_layers;
    //neurons::FCNN_layer m_nn_layer;

public:
    Conv_pooling_nn(
        double l_rate,
        lint batch_size,
        lint threads,
        lint steps,
        lint epoch_size,
        lint secs_allowed,
        const dataset::Dataset &d_set);

    Conv_pooling_nn(const Conv_pooling_nn & other) = delete;
    Conv_pooling_nn(Conv_pooling_nn && other) = delete;
    Conv_pooling_nn & operator = (const Conv_pooling_nn & other) = delete;
    Conv_pooling_nn & operator = (Conv_pooling_nn && other) = delete;

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


