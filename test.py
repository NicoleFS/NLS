import torch.nn as nn
import torch
from dataloader import SentenceData1
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from LSTM import ModelLSTM
from BiLSTM import ModelBiLSTM
from MaxBiLSTM import ModelMaxBiLSTM
import argparse

def test(config):
    
    if config.add_tensorboard_writer:
        writer = SummaryWriter('runs/' + config.model_type + '-1')
        
    device = "cpu"
   
    pickle_embed = open("./embeddings.pickle","rb")
    embedding_matrix = pickle.load(pickle_embed)
    pickle_vocab = open("./vocab_snli.pickle", "rb")
    vocabulary = pickle.load(pickle_vocab)
    pickle_w2i = open("./word2index.pickle", "rb")
    word2index = pickle.load(pickle_w2i)
    
    output_dim = 3
    
    # Return embeddings
    dataset = SentenceData1("./dataset-sts/data/rte/snli/snli_1.0/snli_1.0_test.txt", embedding_matrix, word2index)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)    
    
    # Initialize all models with standard parameters
    if config.model_type == "unidirectional":
        model = ModelLSTM(300, config.num_hidden, config.num_layers, output_dim, device=device, pad_packed = True)
        model.load_state_dict(torch.load("./LSTM_64_ALL_epoch_49.pt"))
        
    elif config.model_type == "bidirectional":
        model = ModelBiLSTM(300, config.num_hidden, config.num_layers, output_dim, device=device, pad_packed = True)
        model.load_state_dict(torch.load("./BiLSTM_64_ALL_epoch_40.pt"))
        
    elif config.model_type == "maxpooling":
        model = ModelMaxBiLSTM(300, config.num_hidden, config.num_layers, output_dim, device=device, pad_packed = True)
        model.load_state_dict(torch.load("./MaxBiLSTM_64_ALL_epoch_49.pt"))
        
    model.eval()
    
    with torch.no_grad():
            
        accuracies = []
        for step, sentence_info in enumerate(dataloader):
            # Retrieve important information from dataloader output
            max_len = sentence_info[3][0]            
            sentence1_matrix = sentence_info[0].view(max_len, config.batch_size, 300)
            sentence2_matrix = sentence_info[1].view(max_len, config.batch_size, 300)
            sentences = sentence_info[2]
            
            sent1_len = sentences[1]
            sent2_len = sentences[3]
            targets = sentences[4]
            
            # Cast input to device
            batch_sent1 = sentence1_matrix.to(device)
            batch_sent1 = batch_sent1.float()
            batch_sent2 = sentence2_matrix.to(device)
            batch_sent2 = batch_sent2.float()
            
            # Make the targets ready for the model
            batch_targets = targets.to(device)
            
            # Initialize the hidden and cell state with zeros
            ho_1 = torch.zeros(config.num_layers, config.batch_size, config.num_hidden)
            ho_1 = ho_1.to(device)
            co_1 = torch.zeros(config.num_layers, config.batch_size, config.num_hidden)
            co_1 = co_1.to(device)
            
            ho_2 = torch.zeros(config.num_layers, config.batch_size, config.num_hidden)
            ho_2 = ho_2.to(device)
            co_2 = torch.zeros(config.num_layers, config.batch_size, config.num_hidden)
            co_2 = co_2.to(device)
            
            # Run the model
            predictions, hidden_layer1, hidden_layer2 = model(batch_sent1, batch_sent2, sent1_len.to(device), sent2_len.to(device), (ho_1, co_1), (ho_2, ho_2))
            predictions = predictions.view(config.batch_size, 3)
            accuracy = torch.sum(torch.eq(torch.argmax(predictions,1), batch_targets)).type(torch.FloatTensor) / config.batch_size
            accuracies.append(accuracy)
            
            if config.save_output:
                if config.model_type == "unidirectional":
                    with open('accuracies_lstm_test.pkl', 'wb') as g:
                        pickle.dump(accuracies, g)
                
                elif config.model_type == "bidirectional":
                    with open('accuracies_bilstm_test.pkl', 'wb') as g:
                        pickle.dump(accuracies, g)
                
                elif config.model_type == "maxpooling":
                    with open('accuracies_maxpool_test.pkl', 'wb') as g:
                        pickle.dump(accuracies, g)
        
        print("Model type: ", config.model_type)
        print("Average accuracy: ", np.sum(accuracies)/len(accuracies)) 
    
    
                            
if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params    
    parser.add_argument('--model_type', type=str, default="maxpooling", help='Which model do you want to run, choose between unidirectional, bidirectional or maxpooling  (default)')
    parser.add_argument('--num_hidden', type=int, default=512, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers in the model')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')

    # Misc params
    parser.add_argument('--add_tensorboard_writer', type=bool, default=True, help='Whether to use tensorboard, default is True')
    parser.add_argument('--save_output', type=bool, default=False, help='Whether to save the test accuracies, default is False')

    config = parser.parse_args()

    # Train the model
    test(config)
