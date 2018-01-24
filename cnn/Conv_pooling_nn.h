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

class Conv_Pooling_NN : public NN
{
private:

    //std::vector<neurons::CNN_layer> m_conv_layers;
    std::vector<neurons::Pooling_layer> m_pooling_layers;
    //neurons::FCNN_layer m_nn_layer;

public:
    Conv_Pooling_NN(
        double l_rate,
        lint batch_size,
        lint threads,
        lint steps,
        lint epoch_size,
        lint secs_allowed,
        const dataset::Dataset &d_set);

    Conv_Pooling_NN(const Conv_Pooling_NN & other) = delete;
    Conv_Pooling_NN(Conv_Pooling_NN && other) = delete;
    Conv_Pooling_NN & operator = (const Conv_Pooling_NN & other) = delete;
    Conv_Pooling_NN & operator = (Conv_Pooling_NN && other) = delete;

public:

    virtual void print_layers(std::ostream & os) const;

private:

    virtual std::vector<neurons::Matrix> test(
        const std::vector<neurons::Matrix> & inputs,
        const std::vector<neurons::Matrix> & targets,
        lint thread_id);

    virtual std::vector<neurons::Matrix> optimise(
        const std::vector<neurons::Matrix> & inputs,
        const std::vector<neurons::Matrix> & targets,
        lint thread_id);

};


