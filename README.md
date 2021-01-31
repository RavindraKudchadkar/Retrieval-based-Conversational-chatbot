
Dataset
=================================================================================================
The data in the folder /datasets contain:
trainfile.txt 
validationfile.txt 
testfile.txt 
glovevectors.txt

The dataset can be found at https: www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip?dl=0&file_subpath=%2Fubuntu_data

Glove vector file can be found at: https://drive.google.com/drive/folders/1X7q34EelbkAfPC7iG97WdTjXqudClvws

===============================================================================================

Directory /Folder contains:

All the files generated while training and testing.

Installations
================================================================================================
Create a conda environment and activate it,

Installation of all the packages:

pip3 install tensorflow

conda install pytorch -c pytorch 

pip3 install numpy

pip3 install keras 

Making the necessary files
================================================================================================
Making All the necessary files which will be saved inside the directory /Folder

python3 making_files.py --trainpath trainfile_path --testpath testfile_path --glovepath glovefile_path

Here an integer will be printed to the screen which is the size of the vocabulary formed. Use this integer to  change the value 
of vocab_size in the Config class of every file containing the neural Model. 

A total of  14 pickled files will be formed inside /Folder, they are:

word_embeddings
context
context_lengths
response
response_lengths
label
single_turn_context
vocab
test_context
test_context_lengths
test_response
test_response_lengths
test_label
test_single_turn_context

Training
=======================================================================================================================================================
Training the models:

For single turn response selection models (biLSTM)

python3 LSTM_train.py --batch_size 50 --learning_rate 0.001 --l2_decay 0.0001 --epochs num_epochs --model_name model_name(biLSTM or two_layered_LSTM)

For multi turn response selection models (SMN, DAM, IOI, MSN)

python train.py --batch_size 50 --learning_rate 0.001 --l2_decay 0.0001 --epochs num_epochs --model_name model_name(SMN/DAM/IOI/MSN)

After the training a graph will appear(accuracy and loss v/s epochs), please save it.

The trained models will be saved in the directory /TIDUA as : saved_modelname_model_totalpairs_examples.pt (e.g saved_DAM_model_1000000_examples)

Testing
========================================================================================================================================================
Testing the models:

python test.py --model_name model_name --model_path model_path

A file containing all the scores for a particular model will saved in /TIDUA as : scores_of_modelname (e.g scores_of_SMN)

Evaluation against metrics
=====================================================================================================================================
Evaluating the models:

python evaluate.py --scores_path  path_to_scores_for_a_particular_model

The following values will be printed to the screen, please save these:
Recall@1
Recall@2 
Recall@5
Precision@1
Precision@2
Precision@5
MRR
MAP

Command line Demo
================================================================================================================================
Demo:

Make a file containing the an utternance (with the name demo_utt.txt), and another one containing a collection of responses with the name (demo_response.txt).
run as follows:
python demo.py 




