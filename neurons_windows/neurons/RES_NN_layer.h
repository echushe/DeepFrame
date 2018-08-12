#pragma once
#include "TMatrix.h"
#include "Functions.h"
#include "NN_layer.h"

namespace neurons
{
    // Each Residual layer includes 2 NN layers
    class RES_NN_layer : public NN_layer
    {
    private:
        double m_mmt_rate;

        TMatrix<> m_w_1;
        TMatrix<> m_b_1;

        TMatrix<> m_w_2;
        TMatrix<> m_b_2;

        TMatrix<> m_w_1_mmt;
        TMatrix<> m_b_1_mmt;

        TMatrix<> m_w_2_mmt;
        TMatrix<> m_b_2_mmt;

        // The pointer of activation function (logist, softmax, etc)
        std::unique_ptr<Activation> m_act_func;
        // The pointer of error function (sigmoid_crossentropy, softmax_crossentropy, etc)
        std::unique_ptr<ErrorFunction> m_err_func;

    public:
        RES_NN_layer();

        RES_NN_layer(
            double mmt_rate, lint threads,
            const TMatrix<> & w_1, const TMatrix<> & b_1,
            const TMatrix<> & w_2, const TMatrix<> & b_2,
            std::unique_ptr<Activation> & act_func, std::unique_ptr<ErrorFunction> & err_func);

        RES_NN_layer(
            double mmt_rate,
            const Shape &w_sh, const Shape &b_sh, lint threads,
            Activation *act_func, ErrorFunction *err_func = nullptr);

        RES_NN_layer(const RES_NN_layer & other);

        RES_NN_layer(RES_NN_layer && other);

        RES_NN_layer & operator = (const RES_NN_layer & other);

        RES_NN_layer & operator = (RES_NN_layer && other);

        ~RES_NN_layer();
    };
}

