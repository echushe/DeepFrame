#include "NN.h"
#include <thread>

NN::NN(
    double l_rate,
    lint batch_size,
    lint threads,
    lint steps,
    lint epoch_size,
    lint secs_allowed,
    const dataset::Dataset &d_set)
    :
    m_l_rate{ l_rate },
    m_batch_size{ batch_size },
    m_threads{ threads },
    m_steps { steps },
    m_epoch_size { epoch_size },
    m_secs_allowed{ secs_allowed }
{
    // Load the training set
    d_set.get_training_set(this->m_train_set, this->m_train_labels);

    // Load the test set
    d_set.get_test_set(this->m_test_set, this->m_test_labels);

    if (!(
        this->m_train_set.size() > 0 &&
        this->m_train_labels.size() > 0 &&
        this->m_train_set.size() == this->m_train_labels.size() &&
        this->m_test_set.size() > 0 &&
        this->m_test_labels.size() > 0 &&
        this->m_test_set.size() == this->m_test_labels.size() &&
        this->m_train_set[0].shape() == this->m_test_set[0].shape() &&
        this->m_train_labels[0].shape() == this->m_test_labels[0].shape()
        ))
    {
        throw std::invalid_argument(std::string("The data set is wrong."));
    }

    this->m_train_rand = std::default_random_engine{ 100 };
    this->m_test_rand = std::default_random_engine{ 200 };
    
    this->m_train_distribution = std::uniform_int_distribution<size_t>{ 0, this->m_train_set.size() - 1 };
    this->m_test_distribution = std::uniform_int_distribution<size_t>{ 0, this->m_test_set.size() - 1 };
}


NN::~NN()
{}


void NN::print_train_set(std::ostream & os) const
{
    os << "There are " << this->m_train_set.size() << " items in the training set\n";
    if (this->m_train_set.size() > 0)
    {
        os << "The first item:\n";
        os << this->m_train_set[0] << '\n';
    }
}


void NN::print_train_label(std::ostream & os) const
{
    os << "There are " << this->m_train_labels.size() << " items in the training label\n";
    if (this->m_train_labels.size() > 0)
    {
        os << "The first item:\n";
        os << this->m_train_labels[0] << '\n';
    }
}


void NN::print_test_set(std::ostream & os) const
{
    os << "There are " << this->m_test_set.size() << " items in the test set\n";
    if (this->m_test_set.size() > 0)
    {
        os << "The first item:\n";
        os << this->m_test_set[0] << '\n';
    }
}


void NN::print_test_label(std::ostream & os) const
{
    os << "There are " << this->m_train_labels.size() << " items in the test labels\n";
    if (this->m_test_labels.size() > 0)
    {
        os << "The first item:\n";
        os << this->m_test_labels[0] << '\n';
    }
}


void NN::train()
{
    std::vector<std::vector<neurons::TMatrix<>>> inputs;
    std::vector<std::vector<neurons::TMatrix<>>> targets;
    std::vector<std::vector<neurons::TMatrix<>>> preds;

    lint start_time = neurons::now_in_seconds();

    double loss_sum = 0;
    double accuracy_sum = 0;

    for (lint i = 1; i <= this->m_steps; ++i)
    {
        this->get_batch(inputs, targets, this->m_train_set, this->m_train_labels, this->m_train_rand, this->m_train_distribution);
        loss_sum += this->train_step(inputs, targets, preds);
        accuracy_sum += this->get_accuracy(preds, targets);

        lint now = neurons::now_in_seconds();
        if (now - start_time > this->m_secs_allowed)
        {
            break;
        }

        if (0 == i % this->m_epoch_size)
        {

            std::cout << "Training step: " << i << '\n';
            std::cout << "The avg loss: " << loss_sum / this->m_epoch_size << '\n';
            std::cout << "The avg accuracy: " << accuracy_sum / this->m_epoch_size << "\n";
            std::cout << "Time: " << now - start_time << " seconds\n\n";

            loss_sum = 0;
            accuracy_sum = 0;
        }

        if (0 == i % (this->m_epoch_size * 5))
        {
            this->test();
        }

    }
}


void NN::test()
{
    std::vector<std::vector<neurons::TMatrix<>>> inputs;
    std::vector<std::vector<neurons::TMatrix<>>> targets;
    std::vector<std::vector<neurons::TMatrix<>>> preds;

    double loss_sum = 0;
    double accuracy_sum = 0;

    for (lint i = 0; i < this->m_epoch_size; ++i)
    {
        this->get_batch(inputs, targets, this->m_test_set, this->m_test_labels, this->m_test_rand, this->m_test_distribution);
        loss_sum += this->test_step(inputs, targets, preds);
        accuracy_sum += this->get_accuracy(preds, targets);
    }

    std::cout << "========Test:\n";
    std::cout << "========The avg loss: " << loss_sum / this->m_epoch_size << '\n';
    std::cout << "========The avg accuracy: " << accuracy_sum / this->m_epoch_size << "\n\n";
}


void NN::get_batch(
    std::vector<std::vector<neurons::TMatrix<>>> & data_batch,
    std::vector<std::vector<neurons::TMatrix<>>> & label_batch,
    const std::vector<neurons::TMatrix<>> & data,
    const std::vector<neurons::TMatrix<>> & label,
    std::default_random_engine & rand_generator,
    std::uniform_int_distribution<size_t> & distribution)
{
    data_batch.clear();
    label_batch.clear();

    size_t set_size = data.size();
    lint batch_size_of_each_thread = this->m_batch_size / this->m_threads;

    if (0 != this->m_batch_size % this->m_threads)
    {
        ++batch_size_of_each_thread;
    }

    std::vector<neurons::TMatrix<>> data_batch_of_each_thread;
    std::vector<neurons::TMatrix<>> label_batch_of_each_thread;

    for (lint i = 1; i <= this->m_batch_size; ++i)
    {
        // randomly select a training input
        size_t j = distribution(rand_generator);

        data_batch_of_each_thread.push_back(data[j]);
        label_batch_of_each_thread.push_back(label[j]);

        if (0 == i % batch_size_of_each_thread)
        {
            data_batch.push_back(data_batch_of_each_thread);
            label_batch.push_back(label_batch_of_each_thread);

            data_batch_of_each_thread.clear();
            label_batch_of_each_thread.clear();
        }
    }

    if (data_batch_of_each_thread.size() > 0)
    {
        data_batch.push_back(data_batch_of_each_thread);
        label_batch.push_back(label_batch_of_each_thread);
    }
}


double NN::train_step(
    const std::vector<std::vector<neurons::TMatrix<>>> & inputs,
    const std::vector<std::vector<neurons::TMatrix<>>> & targets,
    std::vector<std::vector<neurons::TMatrix<>>> & preds)
{
    preds.resize(inputs.size());

    size_t actual_threads = inputs.size();
    size_t new_threads = actual_threads - 1;
    std::vector<std::thread> train_threads{ new_threads };

    size_t thread_id = 0;
    // Create new threads if there are extra threads needed
    for (; thread_id < new_threads; ++thread_id)
    {
        train_threads[thread_id] = std::thread(
            [this, &inputs, &targets, thread_id, &preds]
        {
            preds[thread_id] = this->optimise(inputs[thread_id], targets[thread_id], thread_id);
        });
    }
    // Do the training within the main thread
    preds[thread_id] = this->optimise(inputs[thread_id], targets[thread_id], thread_id);

    for (size_t i = 0; i < new_threads; ++i)
    {
        train_threads[i].join();
    }

    double loss = 0;
    for (size_t i = 0; i < this->m_layers.size(); ++i)
    {
        loss += this->m_layers[i]->commit_training();
    }

    return loss / this->m_batch_size;
}


double NN::test_step(
    const std::vector<std::vector<neurons::TMatrix<>>> & inputs,
    const std::vector<std::vector<neurons::TMatrix<>>> & targets,
    std::vector<std::vector<neurons::TMatrix<>>> & preds)
{
    preds.resize(inputs.size());

    size_t actual_threads = inputs.size();
    size_t new_threads = actual_threads - 1;
    std::vector<std::thread> test_threads{ actual_threads };

    size_t thread_id = 0;
    // Create new threads if there are extra threads needed
    for (; thread_id < new_threads; ++thread_id)
    {
        test_threads[thread_id] = std::thread(
            [this, &inputs, &targets, thread_id, &preds]
        {
            preds[thread_id] = test(inputs[thread_id], targets[thread_id], thread_id);
        });
    }
    // Do the test within the main thread
    preds[thread_id] = test(inputs[thread_id], targets[thread_id], thread_id);
    
    for (size_t i = 0; i < new_threads; ++i)
    {
        test_threads[i].join();
    }

    double loss = 0;
    for (size_t i = 0; i < this->m_layers.size(); ++i)
    {
        loss += this->m_layers[i]->commit_testing();
    }

    return loss / this->m_batch_size;
}

double NN::get_accuracy(const neurons::TMatrix<> & pred, const neurons::TMatrix<> & target)
{
    double acc = 0;
    neurons::Coordinate pred_argmax = pred.argmax();
    neurons::Coordinate target_argmax = target.argmax();

    if (pred_argmax == target_argmax)
    {
        acc = 1.0;
    }
    else
    {
        acc = 0.0;
    }
    return acc;
}

double NN::get_accuracy(
    const std::vector<std::vector<neurons::TMatrix<>>> & preds,
    const std::vector<std::vector<neurons::TMatrix<>>> & targets)
{
    double sum = 0;

    for (size_t i = 0; i < preds.size(); ++i)
    {
        for (size_t j = 0; j < preds[i].size(); ++j)
        {
            sum += this->get_accuracy(preds[i][j], targets[i][j]);
        }
    }

    return sum / this->m_batch_size;
}

std::ostream & operator<<(std::ostream & os, const NN & nn)
{
    nn.print_layers(os);
    nn.print_train_set(os);
    nn.print_train_label(os);
    nn.print_test_set(os);
    nn.print_test_label(os);

    return os;
}
