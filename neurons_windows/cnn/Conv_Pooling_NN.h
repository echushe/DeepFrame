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
        double mmt_rate,
        lint threads,
        const std::string & model_file,
        const dataset::Dataset &d_set);

    Conv_Pooling_NN(const Conv_Pooling_NN & other) = delete;
    Conv_Pooling_NN(Conv_Pooling_NN && other) = delete;
    Conv_Pooling_NN & operator = (const Conv_Pooling_NN & other) = delete;
    Conv_Pooling_NN & operator = (Conv_Pooling_NN && other) = delete;

public:

    virtual void print_layers(std::ostream & os) const;

    virtual void save_layers_as_images() const;
    
    virtual bool load(const std::string & file_name);

    virtual bool load_until(const std::string & file_name, lint layer_index);

    virtual void initialize_model();

    virtual void save(const std::string & file_name) const;

private:

    virtual std::vector<neurons::TMatrix<>> test(
        const std::vector<neurons::TMatrix<>> & inputs,
        const std::vector<neurons::TMatrix<>> & targets,
        lint thread_id);

    virtual std::vector<neurons::TMatrix<>> optimise(
        const std::vector<neurons::TMatrix<>> & inputs,
        const std::vector<neurons::TMatrix<>> & targets,
        lint thread_id);

    virtual std::vector<neurons::TMatrix<>> predict(
        const std::vector<neurons::TMatrix<>> & inputs,
        lint thread_id) const;

};


