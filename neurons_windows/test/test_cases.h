#pragma once
#include "Vector.h"
#include "TMatrix.h"
#include "Functions.h"
#include "Convolution.h"
#include "Pooling.h"
#include "MixtureModel.h"
#include "Dataset.h"
#include "RNN_unit.h"
#include "Mnist.h"
#include "PGM.h"
#include "Review.h"
#include "LinearRegression.h"
#include <iostream>
#include <vector>
#include <list>

void vector_cases()
{
    neurons::Vector a(2);

    std::list<double> l{ 1,2,3 };
    neurons::Vector b(l.begin(), l.end());

    std::vector<double> v2{ 4,5,6,7 };
    neurons::Vector c{ v2.begin(),v2.end() };

    std::vector<double> a1{ 5,4,3,2,1 };
    neurons::Vector d{ a1.begin(),a1.end() };

    std::list<double> a2{ 9,0,8,6,7 };
    neurons::Vector e{ a2.begin(),a2.end() };

    // use the copy constructor
    neurons::Vector f{ e };

    std::cout << a.getNumDimensions() << ": " << a << "\n";
    std::cout << "D1:" << b.get(1) << " " << b << "\n";
    std::cout << c << " Euclidean Norm = " << c.getEuclideanNorm() << "\n";
    std::cout << d << " Unit Vector: " << d.createUnitVector() << " L = " << d.createUnitVector().getEuclideanNorm() << "\n";
    std::cout << e << "\n";
    std::cout << f << "\n";

    // test the move constructor
    neurons::Vector g = std::move(f);
    std::cout << g << "\n";
    std::cout << f << "\n";

    // try operator overloading
    e += d;
    std::cout << e << "\n";

    neurons::Vector h = e - g;
    std::cout << h << "\n";

    // test scalar multiplication
    h *= 2;
    std::cout << h << "\n";

    neurons::Vector j = b / 2;
    std::cout << j << "\n";

    std::cout << "dot product = " << j * b << "\n";

    if (g == (e - d)) std::cout << "true" << "\n";
    if (j != b) std::cout << "false" << "\n";

    j[0] = 1;
    std::cout << j << "\n";

    // type cast from Vector to a std::vector
    std::vector<double> vj = j;

    // type cast from Vector to a std::vector
    std::list<double> lj = j;

    for (auto d : lj)
    {
        std::cout << d << "\n";
    }

    // list initialisation
    neurons::Vector k{ 1, 2, 3 };
    std::cout << k << "\n";

    std::cout << "========================= my own test cases =========================" << "\n";

    neurons::Vector scn(2, 7);
    neurons::Vector scn0{ 3, 3 };
    neurons::Vector scn1{ 9, 8, 7, 6, 5 };
    neurons::Vector scn2{ 1, 2, 3, 4, 5, 6 };
    neurons::Vector scn3(scn1);
    neurons::Vector scn4;

    std::cout << scn << "\n";
    scn0 = std::move(scn1);
    std::cout << scn0 << "\n";
    std::cout << scn1 << "\n";

    scn0 = -1 * scn0;
    std::cout << scn0 << "\n";
    scn0 = scn2 * (scn3 * scn0);
    std::cout << scn0 << "\n";
    std::cout << scn3 << "\n";

    scn4 = scn2;

    std::cout << scn4 << "\n";

    scn4 = (scn2 + (scn0 - scn2) * 3.33) / 12.33;

    std::cout << scn4 << "\n";

    scn4 += scn2;

    std::cout << scn4 << "\n";

    scn4 *= 2;

    std::cout << scn4 << "\n";

    scn4 /= 100;

    std::cout << scn4 << "\n";

    scn4 -= scn0;

    std::cout << scn4 << "\n";

    scn4 = scn4.createUnitVector();

    std::cout << scn4 << "\n";

    try
    {
        neurons::Vector scn5 = scn + scn2;
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        scn += scn2;
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        neurons::Vector scn5 = scn - scn2;
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        scn -= scn2;
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        double dot = scn * scn2;
        std::cout << dot << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        scn1 = scn1;
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        scn1 = std::move(scn1);
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        std::cout << scn0[100] << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        std::cout << scn0.get(-2) << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        std::cout << (scn0 /= 0) << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        std::cout << (scn0 / 0) << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        neurons::Vector zero_v{ 0, 0, 0, 0, 0, 0 };
        std::cout << (zero_v.createUnitVector()) << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }
}


void test_matrix_constructor()
{
    std::cout << "=================== test_matrix_constructor ==================" << "\n";

    neurons::TMatrix<> mat1(neurons::Shape{ 6 }, 3);
    std::cout << "mat1:\n" << mat1 << '\n';

    neurons::TMatrix<> mat2{ neurons::Shape{ 1, 6 }, 3 };
    std::cout << "mat2:\n" << mat2 << '\n';

    neurons::TMatrix<> mat3{ neurons::Shape{ 6, 1 }, 4.4 };
    std::cout << "mat3:\n" << mat3 << '\n';

    neurons::TMatrix<> mat4{ neurons::Shape{ 6, 5, 4 }, -3 };
    std::cout << "mat4:\n" << mat4 << '\n';

    try
    {
        neurons::TMatrix<> mat6{ neurons::Shape{ 3, 0, 9 }, -9 };
    }
    catch (std::exception & e)
    {
        std::cout << e.what() << '\n';
    }

    neurons::TMatrix<> mat7(neurons::Vector{ 1, 3, 5, 7, 9, 11, 13 }, false);
    std::cout << "mat7:\n" << mat7 << '\n';

    neurons::TMatrix<> mat8(neurons::Vector{ 1, 3, 5, 7, 9, 11, 13 }, true);
    std::cout << "mat8:\n" << mat8 << '\n';

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                mat4[{i, j, k}] = i * j * k;
            }
        }
    }

    neurons::TMatrix<> mat9{ mat4 };
    std::cout << "mat9:\n" << mat9 << '\n';

    neurons::TMatrix<> mat10{ std::move(mat4) };
    std::cout << "mat10:\n" << mat10 << '\n';
    std::cout << "mat4:\n" << mat4 << '\n';

    neurons::TMatrix<> mat11{ neurons::Shape{ 9, 5 }, -1 };
    mat11 = mat10;
    std::cout << "mat11:\n" << mat11 << '\n';

    neurons::TMatrix<> mat12{ neurons::Shape{ 2, 5 }, -1 };
    mat12 = std::move(mat10);
    std::cout << "mat12:\n" << mat12 << '\n';
    std::cout << "mat10:\n" << mat10 << '\n';

    neurons::TMatrix<> mat13{ neurons::Shape{ 7, 8 }, -1 };
    neurons::TMatrix<> mat14{ neurons::Shape{ 7, 8 }, 3 };
    neurons::TMatrix<> mat15{ neurons::Shape{ 7, 8 }, 9 };
    std::vector<neurons::TMatrix<>> vec;
    vec.push_back(mat13);
    vec.push_back(mat14);
    vec.push_back(mat15);

    neurons::TMatrix<> mat16{ vec };
    std::cout << "mat16:\n" << mat16 << '\n';
}


void test_matrix_indexing()
{
    std::cout << "=================== test_matrix_indexing ==================" << "\n";
    neurons::TMatrix<> mat1{ neurons::Shape{ 6, 7 }, 8.88 };
    std::cout << mat1 << '\n';

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            mat1[{i, j}] = i * j;
        }
    }

    std::cout << mat1 << '\n';

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            mat1[neurons::Coordinate{ i, j }] = i - j;
        }
    }

    std::cout << mat1 << '\n';
}

void test_matrix_self_cal()
{
    std::cout << "=================== test_matrix_self_cal ==================" << "\n";
    neurons::TMatrix<> mat1{ neurons::Shape{ 6, 5, 5 }, 8 };
    neurons::TMatrix<> mat2{ neurons::Shape{ 6, 5, 5 }, 2 };
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                mat1[neurons::Coordinate{ i, j, k }] = j - k;
            }
        }
    }

    mat1 += mat2;
    std::cout << "mat1:\n" << mat1 << '\n';

    mat1 -= mat2;
    std::cout << "mat1:\n" << mat1 << '\n';

    mat1 *= 2;
    std::cout << "mat1:\n" << mat1 << '\n';

    mat1 /= 2;
    std::cout << "mat1:\n" << mat1 << '\n';
}

void test_matrix_mul()
{
    std::cout << "=================== test_matrix_multiply ==================" << "\n";

    neurons::TMatrix<> mat1{ neurons::Shape{ 6, 5 } };
    neurons::TMatrix<> mat2{ neurons::Shape{ 5, 3 } };

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            mat1[{i, j}] = i * 5 + j;
        }
    }

    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            mat2[{i, j}] = i * 3 + j;
        }
    }

    neurons::TMatrix<> mat3 = neurons::matrix_multiply(mat1, mat2);

    std::cout << "mat3:\n" << mat3 << '\n';

    try
    {
        neurons::TMatrix<> mat4{ neurons::Shape{ 5 }, 7 };
        neurons::TMatrix<> mat5{ neurons::Shape{ 3 }, 4 };
        mat3 = neurons::matrix_multiply(mat4, mat5);
        std::cout << "mat3:\n" << mat3 << '\n';
    }
    catch (std::invalid_argument & ex)
    {
        std::cout << ex.what() << '\n';
    }

    neurons::TMatrix<> mat6{ neurons::Shape{ 5, 1 }, 7 };
    neurons::TMatrix<> mat7{ neurons::Shape{ 1, 3 }, 4 };
    mat3 = neurons::matrix_multiply(mat6, mat7);
    std::cout << "mat3:\n" << mat3 << '\n';

    neurons::TMatrix<> mat8{ neurons::Shape{ 1, 5 }, 7 };
    neurons::TMatrix<> mat9{ neurons::Shape{ 5, 1 }, 4 };
    mat3 = neurons::matrix_multiply(mat8, mat9);
    std::cout << "mat3:\n" << mat3 << '\n';

    neurons::TMatrix<> mat10{ neurons::Shape{ 7, 2, 6 } };
    neurons::TMatrix<> mat11{ neurons::Shape{ 3, 4, 5 } };

    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 6; ++k)
            {
                mat10[{i, j, k}] = i * j * k;
            }
        }
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                mat11[{i, j, k}] = i * j * k;
            }
        }
    }

    mat3 = neurons::matrix_multiply(mat10, mat11);
    std::cout << "mat10:\n" << mat10 << '\n';
    std::cout << "mat11:\n" << mat11 << '\n';
    std::cout << "mat3:\n" << mat3 << '\n';

    neurons::TMatrix<> mat12{ neurons::Shape{ 5, 4, 6, 3 }, 1 };
    neurons::TMatrix<> mat13{ neurons::Shape{ 2, 9, 2, 2, 7 }, 1 };
    mat3 = neurons::matrix_multiply(mat12, mat13);
    std::cout << "mat12:\n" << mat12 << '\n';
    std::cout << "mat13:\n" << mat13 << '\n';
    std::cout << "mat3:\n" << mat3 << '\n';
    mat3 = neurons::matrix_multiply(mat12, mat13, 3, 4);
    std::cout << "mat3:\n" << mat3 << '\n';
}



void test_multiply_and_dot_product()
{
    std::cout << "=================== test_multiply_and_dot_product ==================" << "\n";

    neurons::TMatrix<> mat10{ neurons::Shape{ 4, 3 }, 3 };
    neurons::TMatrix<> mat11{ neurons::Shape{ 4, 3 }, 5 };

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            mat10[{i, j}] = i * 3 + j;
            mat11[{i, j}] = i * 3 + j;
        }
    }

    neurons::TMatrix<> mat3 = neurons::multiply(mat10, mat11);
    std::cout << "mat3:\n" << mat3 << '\n';

    neurons::TMatrix<> mat12{ neurons::Shape{ 4, 3 }, 3 };
    neurons::TMatrix<> mat13{ neurons::Shape{ 2, 3, 2, 1 }, 2 };
    double dot = neurons::dot_product(mat12, mat13);
    std::cout << "results of dot product:\n" << dot << '\n';

}

void test_matrix_dim_scale_up()
{
    std::cout << "=================== test_matrix_dim_scale_up ==================" << "\n";

    neurons::TMatrix<> mat1{ neurons::Shape{ 5, 6 }, 1 };
    neurons::Vector vec1{ 1, 2, 3, 4, 5, 6 };
    mat1.scale_one_dimension(1, vec1);

    std::cout << "mat1:\n" << mat1 << '\n';

    neurons::TMatrix<> mat2{ neurons::Shape{ 5, 6 }, 1 };
    neurons::Vector vec2{ 1, 2, 3, 4, 5 };
    mat2.scale_one_dimension(0, vec2);

    std::cout << "mat2:\n" << mat2 << '\n';

    neurons::TMatrix<> mat3{ neurons::Shape{ 4, 5, 6 }, 1 };
    neurons::Vector vec3{ 5, 4, 3, 2, 1 };
    mat3.scale_one_dimension(1, vec3);

    std::cout << "mat3:\n" << mat3 << '\n';

    neurons::TMatrix<> mat4{ neurons::Shape{ 4, 5, 6 }, 1 };
    neurons::Vector vec4{ 7, 6, 5, 4, 3, 2 };
    mat4.scale_one_dimension(2, vec4);

    std::cout << "mat3:\n" << mat4 << '\n';
}

void test_matrix_other_cal()
{
    std::cout << "=================== test_matrix_other_cal ==================" << "\n";

    neurons::TMatrix<> mat1{ neurons::Shape{ 6, 5 } };
    neurons::TMatrix<> mat2{ neurons::Shape{ 6, 5 } };

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            mat1[{i, j}] = i * 5 + j;
        }
    }

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            mat2[{i, j}] = (i * 5 + j) * (-1);
        }
    }

    neurons::TMatrix<> mat3 = mat1 + mat2;
    std::cout << "mat1:\n" << mat1 << '\n';
    std::cout << "mat2:\n" << mat2 << '\n';
    std::cout << "mat3:\n" << mat3 << '\n';

    mat3 = mat1 - mat2;
    std::cout << "mat3:\n" << mat3 << '\n';

    mat3 = mat3 * 3.0;
    std::cout << "mat3:\n" << mat3 << '\n';

    mat3 = 2.0 * mat3;
    std::cout << "mat3:\n" << mat3 << '\n';

    mat3 = mat3 / 2.0;
    std::cout << "mat3:\n" << mat3 << '\n';
}

void test_matrix_random_normal()
{
    std::cout << "=================== test_matrix_random_normal ==================" << "\n";
    neurons::TMatrix<> mat1{ neurons::Shape{ 5, 4, 3 } };
    mat1.uniform_random(0, 1);
    std::cout << "mat1:\n" << mat1 << '\n';
    mat1.gaussian_random(0, 1);
    std::cout << "mat1:\n" << mat1 << '\n';
    mat1.normalize(0, 100);
    std::cout << "mat1:\n" << mat1 << '\n';
}


void test_matrix_transpose()
{
    std::cout << "=================== test_matrix_transpose ==================" << "\n";
    neurons::TMatrix<> mat1{ neurons::Shape{ 4, 3 } };
    mat1.uniform_random(0, 1);
    std::cout << "mat1:\n" << mat1 << '\n';
    neurons::TMatrix<> mat2 = neurons::transpose(mat1);
    std::cout << "mat2 is transpose of mat1:\n" << mat2 << '\n';
}


void test_matrix_collapse()
{
    std::cout << "=================== test_matrix_collapse ==================" << "\n";
    neurons::TMatrix<> mat1{ neurons::Shape{ 5, 4, 3, 6 } };
    lint index = 0;
    for (lint i = 0; i < 5; ++i)
    {
        for (lint j = 0; j < 4; ++j)
        {
            for (lint k = 0; k < 3; ++k)
            {
                for (lint l = 0; l < 6; ++l)
                {
                    mat1[{i, j, k, l}] = static_cast<double>(index++);
                }
            }
        }
    }

    std::vector<neurons::TMatrix<>> vec_1 = mat1.collapse(0);
    std::vector<neurons::TMatrix<>> vec_2 = mat1.collapse(1);
    std::vector<neurons::TMatrix<>> vec_3 = mat1.collapse(2);
    std::vector<neurons::TMatrix<>> vec_4 = mat1.collapse(3);

    for (size_t i = 0; i < vec_1.size(); ++i)
    {
        std::cout << vec_1[i] << '\n';
    }
    std::cout << "-------------------------------------" << '\n';

    for (size_t i = 0; i < vec_2.size(); ++i)
    {
        std::cout << vec_2[i] << '\n';
    }
    std::cout << "-------------------------------------" << '\n';

    for (size_t i = 0; i < vec_3.size(); ++i)
    {
        std::cout << vec_3[i] << '\n';
    }
    std::cout << "-------------------------------------" << '\n';

    for (size_t i = 0; i < vec_4.size(); ++i)
    {
        std::cout << vec_4[i] << '\n';
    }
}


void test_matrix_fuse()
{
    std::cout << "=================== test_matrix_fuse ==================" << "\n";
    neurons::TMatrix<> mat1{ neurons::Shape{ 5, 4, 3, 6 } };
    lint index = 0;
    for (lint i = 0; i < 5; ++i)
    {
        for (lint j = 0; j < 4; ++j)
        {
            for (lint k = 0; k < 3; ++k)
            {
                for (lint l = 0; l < 6; ++l)
                {
                    mat1[{i, j, k, l}] = static_cast<double>(index++);
                }
            }
        }
    }

    neurons::TMatrix<> f_1 = mat1.fuse(0);
    neurons::TMatrix<> f_2 = mat1.fuse(1);
    neurons::TMatrix<> f_3 = mat1.fuse(2);
    neurons::TMatrix<> f_4 = mat1.fuse(3);

    std::cout << "mat1:\n" << mat1 << '\n';
    std::cout << "fuse 1\n" << f_1 << '\n';
    std::cout << "fuse 2\n" << f_2 << '\n';
    std::cout << "fuse 3\n" << f_3 << '\n';
    std::cout << "fuse 4\n" << f_4 << '\n';
}

#pragma optimize("", off)
void test_of_basic_neuron_operations()
{
    std::cout << "=================== test_of_basic_neuron_operations ==================" << "\n";
    std::vector<neurons::TMatrix<>> images;
    std::vector<neurons::TMatrix<>> labels;

    std::string dataset_dir = "D:/develop/my_neurons/dataset/mnist/";

    dataset::Mnist mnist{
        dataset_dir + "train-images-idx3-ubyte",
        dataset_dir + "train-labels-idx1-ubyte",
        dataset_dir + "t10k-images-idx3-ubyte",
        dataset_dir + "t10k-labels-idx1-ubyte"
    };

    mnist.get_test_set(images, labels, 10);

    lint rows = images[0].shape()[0];
    lint cols = images[0].shape()[1];
    lint targets = labels[0].shape()[0];

    neurons::Sigmoid sigmoid;
    neurons::Tanh tanh;
    neurons::Relu relu;
    neurons::Softmax softmax;
    neurons::HalfSquareError hse{ std::make_unique<neurons::Tanh>() };
    neurons::Sigmoid_CrossEntropy sig_cross;
    neurons::Softmax_CrossEntropy sm_cross;

    neurons::TMatrix<> weights{ neurons::Shape{ rows, cols, targets } };
    weights.gaussian_random(0, 0.0001);
    std::cout << weights << '\n';

    for (size_t i = 0; i < images.size(); ++i)
    {
        images[i].reshape(neurons::Shape{ 1, rows, cols });
        labels[i].reshape(neurons::Shape{ 1, targets });
    }

    for (size_t i = 0; i < images.size(); ++i)
    {
        std::cout << "============== Information of image " << i << "================\n";
        std::cout << images[i] << std::endl;
        neurons::TMatrix<> product = neurons::matrix_multiply(images[i], weights, 2, 2);
        std::cout << product << std::endl;
        neurons::TMatrix<> preds, diff;

        std::cout << "sigmoid and its differentiation:\n";
        sigmoid(preds, diff, product);
        std::cout << preds << std::endl;
        std::cout << diff << std::endl;

        std::cout << "tanh and its differentiation:\n";
        tanh(preds, diff, product);
        std::cout << preds << std::endl;
        std::cout << diff << std::endl;

        std::cout << "relu and its differentiation:\n";
        relu(preds, diff, product);
        std::cout << preds << std::endl;
        std::cout << diff << std::endl;

        std::cout << "softmax and its differentiation:\n";
        softmax(preds, diff, product);
        std::cout << preds << std::endl;
        std::cout << diff << std::endl;

        std::cout << "The label:\n";
        std::cout << labels[i] << std::endl;

        std::cout << "The half square error and its differentiation:\n";
        double loss = hse(diff, labels[i], preds);
        std::cout << loss << '\n';
        std::cout << diff << '\n';

        /*
        std::cout << "The sigmoid_crossentropy and its differentiation:\n";
        loss = sig_cross(diff, labels[i], product);
        std::cout << loss << '\n';
        std::cout << diff << '\n';

        std::cout << "The softmax_crossentropy and its differentiation:\n";
        loss = sm_cross(diff, labels[i], product);
        std::cout << loss << '\n';
        std::cout << diff << '\n';
        */
    }
}
#pragma optimize("", on)

void test_conv_1d()
{
    neurons::Conv_1d conv_1d{ neurons::Shape{2, 8, 3}, neurons::Shape{5, 3, 4} };

    neurons::TMatrix<> input{ neurons::Shape{2, 8, 3} };

    for (lint b = 0; b < 2; ++b)
    {
        for (lint i = 0; i < 8; ++i)
        {
            for (lint j = 0; j < 3; ++j)
            {
                input[{b, i, j}] = static_cast<double>(i * j + j + 1);
            }
        }
    }

    neurons::TMatrix<> weights{ neurons::Shape{5, 3, 4} };
    for (lint i = 0; i < 5; ++i)
    {
        for (lint j = 0; j < 3; ++j)
        {
            for (lint k = 0; k < 4; ++k)
            {
                weights[{i, j, k}] = static_cast<double>(k + j + i + 1);
            }
        }
    }

    neurons::TMatrix<> bias{ neurons::Shape{ 1, 4 } };

    for (lint i = 0; i < 4; ++i)
    {
        bias[{0, i}] = static_cast<double>(i * 10000);
    }

    std::cout << "Input: " << '\n';
    std::cout << input << '\n';

    std::cout << "Weights: " << '\n';
    std::cout << weights << '\n';
    neurons::TMatrix<> output = conv_1d(input, weights, bias);

    std::cout << "Output: " << '\n';
    std::cout << output << '\n';

    std::cout << "d(conv)/d(w)\n";
    std::cout << conv_1d.get_diff_to_weights() << '\n';
}


void test_conv_2d()
{
    neurons::Conv_2d conv_2d{ neurons::Shape{ 1, 4, 4, 3 }, neurons::Shape{ 3, 3, 3, 5 }, 1, 1, 2, 2 };

    neurons::TMatrix<> input{ neurons::Shape{ 1, 4, 4, 3 } };

    for (lint b = 0; b < 1; ++b)
    {
        for (lint r = 0; r < 4; ++r)
        {
            for (lint c = 0; c < 4; ++c)
            {
                for (lint j = 0; j < 3; ++j)
                {
                    input[{b, r, c, j}] = static_cast<double>(r + c + j + 1);
                }
            }
        }
    }

    neurons::TMatrix<> weights{ neurons::Shape{ 3, 3, 3, 5 } };
    for (lint r = 0; r < 3; ++r)
    {
        for (lint c = 0; c < 3; ++c)
        {
            for (lint j = 0; j < 3; ++j)
            {
                for (lint k = 0; k < 5; ++k)
                {
                    weights[{r, c, j, k}] = static_cast<double>(r + c + j + k + 1);
                }
            }
        }
    }

    neurons::TMatrix<> bias{ neurons::Shape{1, 5} };

    for (lint i = 0; i < 5; ++i)
    {
        bias[{0, i}] = static_cast<double>(i * 10000);
    }

    std::cout << "Input: " << '\n';
    std::cout << input << '\n';

    std::cout << "Weights: " << '\n';
    std::cout << weights << '\n';
    neurons::TMatrix<> output = conv_2d(input, weights, bias);

    std::cout << "Output: " << '\n';
    std::cout << output << '\n';

    std::cout << "d(conv)/d(w)\n";
    std::cout << conv_2d.get_diff_to_weights() << '\n';

    std::cout << "d(conv)/d(x)\n";
    std::cout << conv_2d.get_diff_to_input() << '\n';

    /*
    input.reshape(input.shape().sub_shape(1, 3));
    weights.reshape(weights.shape().sub_shape(0, 2));
    double dot = neurons::dot_product(input, weights);

    std::cout << "Dot: " << '\n';
    std::cout << dot << '\n';
    */
}


void test_pooling_2d()
{
    neurons::Shape in_sh{ 1, 4, 6, 3 };
    neurons::Shape k_sh{ 1, 2, 3, 3 };

    neurons::MaxPooling_2d pool{ in_sh, k_sh };
    
    neurons::TMatrix<> input{ in_sh };
    input.uniform_random(0, 10);

    neurons::TMatrix<> output = pool(input);
    neurons::TMatrix<> diff;


    std::cout << "Input:\n";
    std::cout << input << '\n';
    std::cout << "Output:\n";
    std::cout << output << '\n';
}

void test_pooling_2d_special()
{
    neurons::Shape in_sh{ 1, 6, 6, 3 };
    neurons::Shape k_sh{ 1, 2, 3, 3 };

    neurons::MaxPooling_2d pool{ in_sh, k_sh };

    neurons::TMatrix<> input{ in_sh };
    input.uniform_random(0, 10);

    neurons::TMatrix<> output = pool(input);
    neurons::TMatrix<> diff;


    std::cout << "Input:\n";
    std::cout << input << '\n';
    std::cout << "Output:\n";
    std::cout << output << '\n';
    //std::cout << "Diff output to input:\n";
    //std::cout << pool.get_diff() << '\n';

    neurons::TMatrix<> diff_E_to_y{ neurons::Shape{1, 3, 2, 3} };
    diff_E_to_y.gaussian_random(0, 2);

    neurons::TMatrix<> diff_E_to_x = pool.back_propagate(diff_E_to_y);

    std::cout << "Diff E to output:\n";
    std::cout << diff_E_to_y << '\n';

    std::cout << "Diff E to input:\n";
    std::cout << diff_E_to_x << '\n';
}

void test_EM_1d_single()
{
    neurons::TMatrix<> input{ neurons::Shape{1000} };
    input.gaussian_random(15, 20);

    std::vector<neurons::TMatrix<>> p_gaussians_in_x;
    std::vector<neurons::TMatrix<>> p_x_in_gaussians;

    neurons::EM_1d em1d{ 1 };
    std::vector<neurons::EM_Gaussian_1d> gaussians = em1d(p_x_in_gaussians, p_gaussians_in_x, input);

    for (size_t i = 0; i < gaussians.size(); ++i)
    {
        std::cout << "Gaussian " << i << ": mu = " << gaussians[i].m_mu
            << ", sigma = " << gaussians[i].m_sigma << ", probability = " << gaussians[i].m_probability << '\n';

        /*
        std::cout << p_x_in_gaussians[i] << '\n';
        std::cout << p_gaussians_in_x[i] << '\n';
        */
    }

    std::cout << '\n';
}


void test_EM_1d_mix()
{
    neurons::TMatrix<> input_1{ neurons::Shape{ 500 } };
    neurons::TMatrix<> input_2{ neurons::Shape{ 500 } };
    neurons::TMatrix<> input_3{ neurons::Shape{ 1000 } };

    std::vector<double> mix;
    input_1.gaussian_random(-15, 8);
    input_2.gaussian_random(10, 3);
    input_3.gaussian_random(20, 10);
    /*
    std::cout << input_1 << '\n';
    std::cout << input_2 << '\n';
    */

    for (lint i = 0; i < input_1.shape().size(); ++i)
    {
        mix.push_back(input_1[{i}]);
    }

    for (lint i = 0; i < input_2.shape().size(); ++i)
    {
        mix.push_back(input_2[{i}]);
    }

    for (lint i = 0; i < input_3.shape().size(); ++i)
    {
        mix.push_back(input_3[{i}]);
    }

    lint mix_size = mix.size();
    
    neurons::TMatrix<> mixed_input{ neurons::Shape{mix_size} };
    for (lint i = 0; i < mix_size; ++i)
    {
        mixed_input[{i}] = mix[i];
    }

    std::vector<neurons::TMatrix<>> p_gaussians_in_x;
    std::vector<neurons::TMatrix<>> p_x_in_gaussians;

    neurons::EM_1d em1d{ 3 };
    std::vector<neurons::EM_Gaussian_1d> gaussians = em1d(p_x_in_gaussians, p_gaussians_in_x, mixed_input);

    for (size_t i = 0; i < gaussians.size(); ++i)
    {
        std::cout << "Gaussian " << i << ": mu = " << gaussians[i].m_mu
            << ", sigma = " << gaussians[i].m_sigma << ", probability = " << gaussians[i].m_probability << '\n';
        /*
        std::cout << p_x_in_gaussians[i];
        std::cout << p_gaussians_in_x[i];
        std::cout << '\n';
        */
    }

    std::cout << '\n';
}

void test_rnn_unit()
{
    // neurons::RNN_unit rnn_unit;
}

void test_review_dataset()
{
    std::cout << "Hello world!\n";
    dataset::Review review{ "D:/develop/my_neurons/dataset/rnn_data_set/glove.6B.50d.txt",
        "D:/develop/my_neurons/dataset/rnn_data_set/reviews", 0.2, 40 };

    std::vector<neurons::TMatrix<>> inputs;
    std::vector<neurons::TMatrix<>> labels;
    review.get_training_set(inputs, labels, 5);

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        std::cout << inputs[i] << '\n';
        std::cout << labels[i] << '\n';
    }
}


void test_linear_regression_A()
{
    std::cout << "Start linear regression test ... \n";
    neurons::Vector x1{ 1, 0 };
    neurons::Vector x2{ 0, 1 };
    neurons::Vector x3{ 2, 0 };
    neurons::Vector x4{ 0, 2 };
    std::vector<neurons::Vector> X{ x1, x2, x3, x4 };
    neurons::Vector Y{ 0.1, 0.1, 0.2, 0.2 };

    neurons::Linear_Regression LR{ X, Y };
    LR.fit(10e-14);

    neurons::Vector tx1{ 2, 2 };
    neurons::Vector tx2{ 0.5, 0.5 };
    neurons::Vector tx3{ 3, 0 };
    std::vector<neurons::Vector> tX{ tx1, tx2, tx3 };
    neurons::Vector tY = LR.predict(tX);

    std::cout << tY << "\n";
    std::cout << LR.coef_and_intercept() << "\n";
}

void test_linear_regression_B()
{
    std::cout << "Start linear regression test ... \n";

    for (lint k = 0; k < 100; ++k)
    {
        std::cout << "================================================\n";
        lint x_dims = 1 + rand() % 10;
        lint samples = 2 + rand() % 10;
        std::cout << "Dimensions of x: " << x_dims << "\n";
        std::cout << "Number of samples: " << samples << "\n";
        std::cout << "--------------------------------------\n";
        std::vector<neurons::Vector> X;

        std::cout << "X:\n";
        for (lint i = 0; i < samples; ++i)
        {
            neurons::Vector x(x_dims);

            for (lint j = 0; j < x_dims; ++j)
            {
                x[j] = (static_cast<double>(rand() % 1000)) / 1000;
            }

            X.push_back(x);
            std::cout << x << "\n";
        }

        std::cout << "\n";

        neurons::Vector Y(samples);
        for (lint i = 0; i < samples; ++i)
        {
            Y[i] = (static_cast<double>(rand() % 1000));
        }
        std::cout << "Y:\n";
        std::cout << Y << "\n\n";

        for (lint i = 1; i < 10; ++i)
        {
            neurons::Linear_Regression LR{ X, Y };
            LR.fit(10e-5 / pow(10, i));
            std::cout << LR.coef_and_intercept() << "\n";
        }
    }
}


void test_of_PGM()
{
    std::cout << "=================== test_of_loading_PGM_dataset ==================" << "\n";
    std::vector<neurons::TMatrix<>> images;
    std::vector<neurons::TMatrix<>> labels;

    std::string dataset_dir = "D:/develop/my_neurons/dataset/cmu_facial/faces/an2i/";
    std::vector<std::string> train_files;
    std::vector<std::string> test_files;

    train_files.push_back(dataset_dir + "an2i_left_angry_open_4.pgm");
    train_files.push_back(dataset_dir + "an2i_left_angry_sunglasses_4.pgm");
    train_files.push_back(dataset_dir + "an2i_left_happy_open_4.pgm");

    test_files.push_back(dataset_dir + "an2i_straight_angry_open_4.pgm");
    test_files.push_back(dataset_dir + "an2i_straight_happy_open_4.pgm");

    dataset::PGM pgm{train_files, test_files, 2};

    std::vector<neurons::TMatrix<>> train_inputs;
    std::vector<neurons::TMatrix<>> train_labels;
    std::vector<neurons::TMatrix<>> test_inputs;
    std::vector<neurons::TMatrix<>> test_labels;

    pgm.get_test_set(test_inputs, test_labels, 0);
    pgm.get_training_set(train_inputs, train_labels, 0);

    for (size_t i = 0; i < train_inputs.size(); ++i)
    {
        train_inputs[i].reshape(neurons::Shape{ train_inputs[i].shape()[0], train_inputs[i].shape()[1] });
        std::cout << train_inputs[i] << "\n";
        std::cout << train_labels[i] << "\n";
    }

    for (size_t i = 0; i < test_inputs.size(); ++i)
    {
        test_inputs[i].reshape(neurons::Shape{ test_inputs[i].shape()[0], test_inputs[i].shape()[1] });
        std::cout << test_inputs[i] << "\n";
        std::cout << test_labels[i] << "\n";
    }
    // mnist.get_test_set(images, labels, 10);
}


void test_of_basic_operations()
{
    /*
    vector_cases();

    test_matrix_constructor();
    test_matrix_indexing();
    test_matrix_self_cal();
    test_matrix_mul();
    test_multiply_and_dot_product();

    test_matrix_dim_scale_up();
    test_matrix_other_cal();
    test_matrix_random_normal();
    test_matrix_transpose();

    test_matrix_collapse();
    test_matrix_fuse();
    test_of_basic_neuron_operations();

    test_conv_1d();
    test_conv_2d();

    test_pooling_2d();
    test_pooling_2d_special();
    */

    test_of_PGM();

    // test_EM_1d_single();
    
    // test_EM_1d_mix();

    // test_rnn_unit();

    // test_review_dataset();

    // test_linear_regression_A();
    // test_linear_regression_B();
}



