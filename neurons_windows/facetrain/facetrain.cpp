
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include "network.h"

/*
  Code based on original's facetrain.c <http://www-2.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/faces.html>
*/

int GL_IS_DEBUG; // 0: No, 1: Yes
std::string relative_set_dir{ "D:/develop/my_neurons/dataset/cmu_facial/sets/" };
std::string relative_face_dir{ "D:/develop/my_neurons/dataset/cmu_facial/faces/" };

void printusage(std::string prog)
{
    std::cout << "USAGE: " << prog << std::endl;
    std::cout << "       -n <network file>" << std::endl;
	std::cout << "       [-f <network borrow from>]" << std::endl;
    std::cout << "       [-N <network type>]" << std::endl;
    std::cout << "       [-S <number of epochs between saves of network>]" << std::endl;
    std::cout << "       [-T]" << std::endl;
    std::cout << "       [-H]" << std::endl;
    std::cout << "       [-a <number of threads>]" << std::endl;
    std::cout << "       [-b <batch size>]" << std::endl;
    std::cout << "       [-e <number of epochs>]" << std::endl;
    std::cout << "       [-E <epoch size>]" << std::endl;
    std::cout << "       [-l <learning-rate>]" << std::endl;
    std::cout << "       [-m <momentum>]" << std::endl;
    std::cout << "       [-s <random number generator seed>]" << std::endl;
    std::cout << "       [-t <training set list>]" << std::endl;
    std::cout << "       [-1 <testing set 1 list>]" << std::endl;
    std::cout << "       [-2 <testing set 2 list>]" << std::endl;
}


void print_fname_list(std::vector<std::string> fname_vect, std::string vect_name)
{
    // if (vect_name!=NULL){
    std::cout << vect_name << " (" << fname_vect.size() << " items):" << std::endl;
    // }

    size_t index;
    for (index = 0; index < fname_vect.size(); index++)
    {
        std::cout << fname_vect[index] << std::endl;
    }
}

int read_list_file(std::vector<std::string> & fname_vector, std::string fname)
{
    int count = -1;
    if (fname.length() == 0)
    {
        std::cout << "read_list_file(): invalid filename: it is empty string";
        return count;
    }

    std::ifstream infile(fname);

    if (!infile)
    {
        std::cout << "file not found, file name: " << fname << '\n';
        return count;
    }

    std::string line;
    while (std::getline(infile, line))
    {
        fname_vector.push_back(relative_face_dir + line);
        count++;
        // std::cout << "Adding " << line << std::endl;
    }
    // std::cout << "Total item: " << count << std::endl;
    return count;
}

std::vector<std::string> nw_name_vect;
int convert_nw_name_to_int(std::string network_name)
{
    if (nw_name_vect.empty())
    {
        nw_name_vect.push_back("face"); // user_id, index 0
        nw_name_vect.push_back("pose"); // pose, index 1
        nw_name_vect.push_back("express"); // expression, index 2
        nw_name_vect.push_back("shades"); // => sunglasses, index 3
    }
    std::string tmp_str;
    for (size_t idx = 0; idx < nw_name_vect.size(); idx++)
    {
        tmp_str = nw_name_vect[idx];
        if (tmp_str.compare(0, network_name.length(), network_name) == 0)
        {
            return idx; // match found, return index
        }
    }

    // no match found:
    return -1;
}


int main(int argc, const char* argv[])
{
    std::string nw_infile, nw_from_file, fname_train, fname_test1, fname_test2;

    int retval = -1;
    int GL_IS_DEBUG = 0;

    /* FIXME: tweak the following according to specs 2.1.xx */
    lint argv_epochs = 100;
    lint list_errors = 0;
    bool test_only = false;
    lint argv_seed = 1;
    bool argv_hidtopgm = false;
    lint n_epochs_between_save = 100;
    int argv_list_errors = 0;
    double argv_momentum = 0.3;
    double argv_learning_rate = 0.001;
    lint argv_batch_size = 8;
    lint argv_nthreads = 4;
    lint argv_epoch_size = 10;
    std::string argv_network = "dnn";

    if (argc < 2)
    {
        printusage(argv[0]);

        return -1;
    }

    std::string network_type = std::string{ argv[1] };
    std::vector<std::string> new_argvs;
    int new_argc = argc - 1;
    new_argvs.push_back(argv[0]);
    for (lint i = 2; i < argc; ++i)
    {
        new_argvs.push_back(std::string(argv[i]));
    }

    if ("cnn" == network_type)
    {
        run_cnn(new_argc, new_argvs);
        return 0;
    }
    else if ("dnn" == network_type)
    {
        run_dnn(new_argc, new_argvs);
        return 0;
    }
    /* Do not exit the program
       Run fracetrain if it is neither dnn nor cnn
    else
    {
        std::cout << "Unrecognized network type: " << network_type << " !" << std::endl;
        return -1;
    }
    */


    /*** Scan command line ***/
    int ind;
    for (ind = 1; ind < argc; ind++)
    {
        // std::cout << argv[ind] << '\n';
        /*** Parse switches ***/
        if (argv[ind][0] == '-')
        {
            switch (argv[ind][1])
            {
            case 'n': nw_infile.assign(argv[++ind]);
                break;
            case 'f': nw_from_file.assign(argv[++ind]);
                break;
            case 'e': argv_epochs = atoi(argv[++ind]);
                break;
            case 'E': argv_epoch_size = atoi(argv[++ind]);
                break;
            case 's': argv_seed = atoi(argv[++ind]);
                break;
            case 'S': n_epochs_between_save = atoi(argv[++ind]);
                break;
            case 't': fname_train.assign(argv[++ind]);
                break;
            case '1': fname_test1.assign(argv[++ind]);
                break;
            case '2': fname_test2.assign(argv[++ind]);
                break;
            case 'T': list_errors = 1;
                test_only = true;
                argv_epochs = 0;
                break;
            case 'D': GL_IS_DEBUG = 1;
                break;
            case 'b': argv_batch_size = std::stoi(argv[++ind]);
                break;
            case 'a': argv_nthreads = std::stoi(argv[++ind]);
                break;
            case 'l': argv_learning_rate = std::stoi(argv[++ind]);
                break;
            case 'N': argv_network.assign(argv[++ind]);
                break;
            case 'm': argv_momentum = std::stoi(argv[++ind]);
                break;
            case 'H': 
                argv_hidtopgm = true;
                break;
     default: std::cout << "Unknown switch '" << argv[ind][1] << "'" << std::endl;
                break;
            }
        }
    }


    /*** Reading file_list ***/
    /*
    if ((fname_train.length() == 0) && (fname_test1.length() == 0) && (fname_test2.length() == 0))
    {
        std::cout << "Syntax error: all train-filename, test1-filename, and test2-filename are NULL!" << std::endl;
        return -1;
    }
    */

    /*** If any train, test1, or test2 sets have been specified, then
         load them in. ***/
    std::vector<std::string> train_list;
    std::vector<std::string> test1_list;
    std::vector<std::string> test2_list;

    if (fname_train.length() > 0)
    {
        retval = read_list_file(train_list, relative_set_dir + fname_train);
    }
    if (fname_test1.length() > 0)
    {
        retval = read_list_file(test1_list, relative_set_dir + fname_test1);
    }
    if (fname_test2.length() > 0)
    {
        retval = read_list_file(test2_list, relative_set_dir + fname_test2);
    }

    std::cout << "after read: " << train_list.size() << std::endl;

    if (GL_IS_DEBUG == 1)
    {
        print_fname_list(train_list, "train_list");
        print_fname_list(test1_list, "test1_list");
        print_fname_list(test2_list, "test2_list");
    }


    /*** If we haven't specified a network save file, we should... ***/
    if (nw_infile.length() == 0)
    {
        std::cout << "facetrain: You might want to specify an output file, i.e., -n <network file>" << std::endl;
        // exit(-1);
    }

    /*** Don't try to train if there's no training data ***/
    if (fname_train.length() == 0)
    {
        argv_epochs = 0;
    }

    int label_id = convert_nw_name_to_int(nw_infile);
    if (label_id < 0)
    {
        std::cout << "Network file name should be one of the follows: face, pose, express, shades.\n";
        return -1;
    }

    run_facial_network(
        argv_hidtopgm, // false
        test_only,
        argv_seed,
        argv_batch_size,    //8,
        argv_nthreads,      //4,
        argv_epoch_size,    // 100,
        argv_epochs,
        n_epochs_between_save,
        label_id,
        argv_learning_rate, // 0.001,
        argv_momentum,      // 0.3,
        argv_network,
        nw_infile,
        nw_from_file,
        train_list,
        test1_list,
        test2_list);

    return 0;

}
