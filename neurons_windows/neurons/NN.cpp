#include "NN.h"
#include <thread>

NN::NN(double l_rate, double mmt_rate, lint threads, const std::string & model_file, const dataset::Dataset &d_set)
    : 
    m_l_rate{ l_rate },
    m_mmt_rate{ mmt_rate <= 1 ? mmt_rate : 1 },
    m_threads{ threads },
    m_model_file{ model_file }
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
    
    this->m_train_distribution = std::uniform_int_distribution<size_t>{ 0, this->m_train_set.size() - 1 };
    this->m_test_distribution = std::uniform_int_distribution<size_t>{ 0, this->m_test_set.size() - 1 };
}


NN::~NN()
{}


lint NN::n_layers() const
{
    return this->m_layers.size();
}

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


void NN::train_network(
    lint batch_size,
    lint epoch_size,
    lint epochs,
    lint epochs_between_saves,
    lint secs_allowed)
{
    std::vector<std::vector<neurons::TMatrix<>>> inputs;
    std::vector<std::vector<neurons::TMatrix<>>> targets;
    std::vector<std::vector<neurons::TMatrix<>>> preds;

    lint start_time = neurons::now_in_seconds();

    double loss_sum = 0;
    double accuracy_sum = 0;

    lint steps = epoch_size * epochs;

    for (lint i = 1; i <= steps; ++i)
    {
        this->get_batch
        (batch_size, inputs, targets, this->m_train_set, this->m_train_labels, this->m_train_distribution);
        loss_sum += this->train_step(batch_size, inputs, targets, preds);
        accuracy_sum += this->get_accuracy(batch_size, preds, targets);

        lint now = neurons::now_in_seconds();
        if (now - start_time > secs_allowed)
        {
            break;
        }

        if (0 == i % epoch_size)
        {

            std::cout << "Training epoch: " << i / epoch_size << '\n';
            std::cout << "Training batch size: " << batch_size << '\n';
            std::cout << "Training epoch size: " << epoch_size << '\n';
            std::cout << "The avg loss: " << loss_sum / epoch_size << '\n';
            std::cout << "The avg accuracy: " << accuracy_sum / epoch_size << "\n";
            std::cout << "Time: " << now - start_time << " seconds\n\n";

            loss_sum = 0;
            accuracy_sum = 0;
        }

        if (0 == i % (epoch_size * epochs_between_saves))
        {
            this->test_network(batch_size, epoch_size);
            this->save(this->m_model_file);
        }

    }
}


void NN::test_network(lint batch_size, lint epoch_size)
{
    std::vector<std::vector<neurons::TMatrix<>>> inputs;
    std::vector<std::vector<neurons::TMatrix<>>> targets;
    std::vector<std::vector<neurons::TMatrix<>>> preds;

    double loss_sum = 0;
    double accuracy_sum = 0;

    for (lint i = 0; i < epoch_size; ++i)
    {
        this->get_batch
        (batch_size, inputs, targets, this->m_test_set, this->m_test_labels, this->m_test_distribution);
        loss_sum += this->test_step(batch_size, inputs, targets, preds);
        accuracy_sum += this->get_accuracy(batch_size, preds, targets);
    }

    std::cout << "========Test batch size: " << batch_size << '\n';
    std::cout << "========Test epoch size: " << epoch_size <<'\n';
    std::cout << "========The avg loss: " << loss_sum / epoch_size << '\n';
    std::cout << "========The avg accuracy: " << accuracy_sum / epoch_size << "\n\n";
}


std::vector<neurons::TMatrix<>> NN::network_predict(lint batch_size, const std::vector<neurons::TMatrix<>>& inputs) const
{
    std::vector<std::vector<neurons::TMatrix<>>> data_batch;
    std::vector<neurons::TMatrix<>> all_preds;

    lint batch_size_of_each_thread = batch_size / this->m_threads;

    if (0 != batch_size % this->m_threads)
    {
        ++batch_size_of_each_thread;
    }

    std::vector<neurons::TMatrix<>> data_batch_of_each_thread;

    for (size_t i = 0; i < inputs.size(); i += batch_size)
    {
        data_batch.clear();

        for (size_t j = 0; j < batch_size; ++j)
        {
            if (i + j < inputs.size())
            {
                data_batch_of_each_thread.push_back(inputs[i + j]);
            }
            else
            {
                break;
            }

            //std::cout << data[j];
            //std::cout << label[j] << "\n";

            if (0 == (j + 1) % batch_size_of_each_thread)
            {
                data_batch.push_back(data_batch_of_each_thread);
                data_batch_of_each_thread.clear();
            }
        }

        if (data_batch_of_each_thread.size() > 0)
        {
            data_batch.push_back(data_batch_of_each_thread);
        }

        auto preds = this->predict_step(batch_size, data_batch);
        for (const std::vector<neurons::TMatrix<>> & preds_each_thread : preds)
        {
            for (const neurons::TMatrix<> & pred : preds_each_thread)
            {
                all_preds.push_back(pred);
            }
        }
    }

    return all_preds;
}


void NN::get_batch(
    lint batch_size,
    std::vector<std::vector<neurons::TMatrix<>>> & data_batch,
    std::vector<std::vector<neurons::TMatrix<>>> & label_batch,
    const std::vector<neurons::TMatrix<>> & data,
    const std::vector<neurons::TMatrix<>> & label,
    std::uniform_int_distribution<size_t> & distribution)
{
    data_batch.clear();
    label_batch.clear();

    lint batch_size_of_each_thread = batch_size / this->m_threads;

    if (0 != batch_size % this->m_threads)
    {
        ++batch_size_of_each_thread;
    }

    std::vector<neurons::TMatrix<>> data_batch_of_each_thread;
    std::vector<neurons::TMatrix<>> label_batch_of_each_thread;

    for (lint i = 1; i <= batch_size; ++i)
    {
        // randomly select a training input
        size_t j = distribution(neurons::global::global_rand_engine);

        data_batch_of_each_thread.push_back(data[j]);
        label_batch_of_each_thread.push_back(label[j]);

        //std::cout << data[j];
        //std::cout << label[j] << "\n";

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
    lint batch_size,
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

    return loss / batch_size;
}


double NN::test_step(
    lint batch_size,
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

    return loss / batch_size;
}

std::vector<std::vector<neurons::TMatrix<>>> NN::predict_step
(lint batch_size, const std::vector<std::vector<neurons::TMatrix<>>>& inputs) const
{
    std::vector<std::vector<neurons::TMatrix<>>> preds;
    preds.resize(inputs.size());

    size_t actual_threads = inputs.size();
    size_t new_threads = actual_threads - 1;
    std::vector<std::thread> test_threads{ actual_threads };

    size_t thread_id = 0;
    // Create new threads if there are extra threads needed
    for (; thread_id < new_threads; ++thread_id)
    {
        test_threads[thread_id] = std::thread(
            [this, &inputs, thread_id, &preds]
        {
            preds[thread_id] = predict(inputs[thread_id], thread_id);
        });
    }
    // Do the test within the main thread
    preds[thread_id] = predict(inputs[thread_id], thread_id);

    for (size_t i = 0; i < new_threads; ++i)
    {
        test_threads[i].join();
    }

    return preds;
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
    lint batch_size,
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

    return sum / batch_size;
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
