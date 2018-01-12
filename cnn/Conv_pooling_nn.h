#pragma once
/********************************************************************

Programmed by Chunnan Sheng

*********************************************************************/

#pragma once
#include "Conv_nn.h"
#include "Pooling.h"

class Conv_pooling_nn : public Mnist_nn
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
        const std::string & train_file,
        const std::string & train_label,
        const std::string & test_file,
        const std::string & test_label);

    Conv_pooling_nn(const Conv_nn & other) = delete;
    Conv_pooling_nn(Conv_nn && other) = delete;
    Conv_pooling_nn & operator = (const Conv_nn & other) = delete;
    Conv_pooling_nn & operator = (Conv_nn && other) = delete;

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


