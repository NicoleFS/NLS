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
from benchmark import BenchMark
import argparse

def train(config):
    
    if config.add_tensorboard_writer:
        writer = SummaryWriter('runs/' + config.model_type + '-1')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    pickle_embed = open("./embeddings.pickle","rb")
    embedding_matrix = pickle.load(pickle_embed)
    pickle_vocab = open("./vocab_snli.pickle", "rb")
    vocabulary = pickle.load(pickle_vocab)
    pickle_w2i = open("./word2index.pickle", "rb")
    word2index = pickle.load(pickle_w2i)
    
    output_dim = 3
    
    # Return embeddings
    dataset = SentenceData1("./dataset-sts/data/rte/snli/snli_1.0/snli_1.0_train.txt", embedding_matrix, word2index)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)    
    
    # Initialize all models with standard parameters
    
    if config.model_type == "unidirectional":
        model = ModelLSTM(300, config.num_hidden, config.num_layers, output_dim, device=device, pad_packed = True)
    elif config.model_type == "bidirectional":
        model = ModelBiLSTM(300, config.num_hidden, config.num_layers, output_dim, device=device, pad_packed = True)
    elif config.model_type == "maxpooling":
        model = ModelMaxBiLSTM(300, config.num_hidden, config.num_layers, output_dim, device=device, pad_packed = True)
    elif config.model_type == "benchmark":
        model = BenchMark(300, output_dim, device=device)
        
    print("Model: ", config.model_type)
    model = model.to(device)
    
    losses = []
    accuracies = []
    
    iters = 0
    count = 0

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = config.learning_rate)
    
    for epoch in range(config.num_epochs):
        
        print("Epoch: ", epoch)
        
        # For each batch
        for step, sentence_info in enumerate(dataloader):
            
            iters += 1
            
            # Retrieve important information from dataloader output
            max_len = sentence_info[3][0]            
            sentence1_matrix = sentence_info[0].view(max_len, config.batch_size, 300)
            sentence2_matrix = sentence_info[1].view(max_len, config.batch_size, 300)
            sentences = sentence_info[2]
            
            sent1_len = sentences[1]
            sent2_len = sentences[3]
            targets = sentences[4]
            
            optimizer.zero_grad()
            
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
            if config.model_type == "benchmark":
                predictions = model(batch_sent1, sent1_len.to(device), batch_sent2, sent2_len.to(device))
            else:
                predictions, hidden_layer1, hidden_layer2 = model(batch_sent1, batch_sent2, sent1_len.to(device), sent2_len.to(device), (ho_1, co_1), (ho_2, ho_2))
            
            # Calculate the loss using Cross Entropy Error
            predictions = predictions.view(config.batch_size,3)            
            loss = loss_function(predictions, batch_targets)
            accuracy = torch.sum(torch.eq(torch.argmax(predictions,1), batch_targets)).type(torch.FloatTensor) / config.batch_size
            
            # Do backprop
            loss.backward()            
            optimizer.step()
            
            
            # Every once in a while, show loss and accuracy
            if iters % 10 == 0:
                print("Loss: ",loss)
                print("Accuracy: ", accuracy)
                
                if config.add_tensorboard_writer:
                    writer.add_scalar('loss', loss.item(), count)
                    writer.add_scalar("accuracy", accuracy.item(), count)
                losses.append(loss)
                accuracies.append(accuracy)
                count += 1
            
            # Save the model every 10 epochs or when the model is finished
            if config.save_model:
                if epoch % 1000 == 0 or epoch == 49:
                
                    if config.model_type == "unidirectional":
                        torch.save(model.state_dict(), "LSTM_64_ALL_epoch_" + str(epoch) + ".pt")
                        with open('losses_lstm.pkl', 'wb') as f:
                            pickle.dump(losses, f)
                        with open('accuracies_lstm.pkl', 'wb') as g:
                            pickle.dump(accuracies, g)    
                            
                    elif config.model_type == "bidirectional":
                        torch.save(model.state_dict(), "BiLSTM_64_ALL_epoch_" + str(epoch) + ".pt")
                        with open('losses_bilstm.pkl', 'wb') as f:
                            pickle.dump(losses, f)
                        with open('accuracies_bilstm.pkl', 'wb') as g:
                            pickle.dump(accuracies, g)
                            
                    elif config.model_type == "maxpooling":
                        torch.save(model.state_dict(), "MaxBiLSTM_64_ALL_epoch_" + str(epoch) + ".pt")
                        with open('losses_maxpool.pkl', 'wb') as f:
                            pickle.dump(losses, f)
                        with open('accuracies_maxpool.pkl', 'wb') as g:
                            pickle.dump(accuracies, g)
                    
                    elif config.model_type == "benchmark":
                        torch.save(model.state_dict(), "benchmark_64_6400_epoch_" + str(epoch) + ".pt")
                        with open('losses_benchmark.pkl', 'wb') as f:
                            pickle.dump(losses, f)
                        with open('accuracies_benchmark.pkl', 'wb') as g:
                            pickle.dump(accuracies, g)


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params    
    parser.add_argument('--model_type', type=str, default="maxpooling", help='Which model do you want to run, choose between benchmark, unidirectional, bidirectional or maxpooling  (default)')
    parser.add_argument('--num_hidden', type=int, default=512, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')

    # Misc params
    parser.add_argument('--num_epochs', type=int, default=50, help='How many epochs for training')
    parser.add_argument('--add_tensorboard_writer', type=bool, default=True, help='Whether to use tensorboard, default is True')
    parser.add_argument('--save_model', type=bool, default=False, help='Whether to save the model every 10 epochs, default is False')

    config = parser.parse_args()

    # Train the model
    train(config)
    
 
