import torch.nn as nn
import torch
from test_dataloader import SentenceData1
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm




class ModelLSTM(nn.Module):

    #def __init__(self, batch_size, seq_length, vocabulary_size,
                 #lstm_num_hidden=256, lstm_num_layers=1, device='cuda:0'):
        
    def __init__(self, input_dim, lstm_num_hidden, lstm_num_layers, output_dim):

        super(ModelLSTM, self).__init__()
        
        # Initialize the LSTM using the PyTorch LSTM module
        self.lstm = nn.LSTM(input_dim, lstm_num_hidden, lstm_num_layers)
        
        # Initialize a linear layer
        #self.classification = nn.Sequential(
        #    nn.Linear(2*lstm_num_hidden, lstm_num_hidden),
        #    nn.Linear(lstm_num_hidden, 512),
        #    nn.Linear(512, output_dim)
        #    )
        
        self.classification = nn.Sequential(
            nn.Linear(2*lstm_num_hidden, 512),
            nn.Linear(512, 512),
            nn.Linear(512, output_dim)
            )
        #self.linear = nn.Linear(2*lstm_num_hidden, output_dim)
    
    def forward(self, x1, x2, x1_lens, x2_lens, hidden_state1, hidden_state2):
        
        hidden_state1[0].requires_grad_()
        hidden_state1[1].requires_grad_()
        hidden_state2[0].requires_grad_()
        hidden_state2[1].requires_grad_()
        
        # Sort the lengths of first sentences in decreasing order and keep the switch in indices
        #sort_x1, ind_1 = torch.sort(x1_lens, 0, descending = True)
        
        # Sort the embedding matrix of the first sentences according to the new indices
        #x1 = x1[:,ind_1,:]
        
        # Sort the lengths of second sentences in decreasing order and keep the switch in indices
        #sort_x2, ind_2 = torch.sort(x2_lens, 0, descending = True)
        
        # Sort the embedding matrix of the first sentences according to the new indices
        #x2 = x2[:,ind_2,:]
        
        #x1_packed = nn.utils.rnn.pack_padded_sequence(x1, sort_x1)
        #x2_packed = nn.utils.rnn.pack_padded_sequence(x2, sort_x2)
        
        # Calculate the output of the LSTM and the hidden and cell state
        out_tmp1, (h1,c1) = self.lstm(x1, (hidden_state1[0], hidden_state1[1]))
        out_tmp2, (h2,c2) = self.lstm(x2, (hidden_state2[0], hidden_state2[1]))
        
        #out_tmp1, (h1,c1) = self.lstm(x1_packed, (hidden_state1[0], hidden_state1[1]))
        #out_tmp2, (h2,c2) = self.lstm(x2_packed, (hidden_state2[0], hidden_state2[1]))
        
        #x1_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(x1_packed)
        #x1_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(x1_packed)
        
        #h1_unsorted = h1.new(*h1.size())
        #h1_sorted = h1_unsorted.scatter_(1, ind_1.unsqueeze(0).unsqueeze(2), h1)

        #h2_unsorted = h2.new(*h2.size())
        #h2_sorted = h2_unsorted.scatter_(1, ind_2.unsqueeze(0).unsqueeze(2), h2)
        
        out_tmp = torch.cat((h1, h2), 2)
        #out_tmp = torch.cat((h1_sorted, h2_sorted),2)
        
        # Calculate the actual output by feeding the LSTM output to the final linear layer.
        #out = self.linear(out_tmp)
        out = self.classification(out_tmp)
        
        #x1 = nn.utils.rnn.pad_packed_sequence(x1_packed, padding_value=0)
        #x2 = nn.utils.rnn.pad_packed_sequence(x1_packed, padding_value=0)
        
        return out, (h1,c1), (h2,c2)


def train():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    pickle_embed = open("./embeddings_train.pickle","rb")
    embedding_matrix = pickle.load(pickle_embed)
    pickle_vocab = open("./vocab_snli_train.pickle", "rb")
    vocabulary = pickle.load(pickle_vocab)
    pickle_w2i = open("./word2index.pickle", "rb")
    word2index = pickle.load(pickle_w2i)
    
    batchsize = 64
    hidden_nodes = 2048
    lstm_layers = 1
    output_dim = 3
    
    # Return embeddings
    dataset = SentenceData1("./dataset-sts/data/rte/snli/snli_1.0/snli_1.0_train.txt", embedding_matrix, word2index)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, drop_last=True)    
    
    model = ModelLSTM(300, hidden_nodes, lstm_layers, output_dim)
    model = model.to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    
    iters = 0
    
    for epoch in range(100):
        
        for step, sentence_info in enumerate(dataloader):
            
            iters += 1
            
            max_len = sentence_info[3][0]            
            sentence1_matrix = sentence_info[0].view(max_len, batchsize, 300)
            sentence2_matrix = sentence_info[1].view(max_len, batchsize, 300)
            sentences = sentence_info[2]
            
            sent1_len = sentences[1]
            sent2_len = sentences[3]
            targets = sentences[4]
            
            optimizer.zero_grad()
            
            batch_sent1 = sentence1_matrix.to(device)
            batch_sent1 = batch_sent1.float()
            batch_sent2 = sentence2_matrix.to(device)
            batch_sent2 = batch_sent2.float()
            
            # Make the targets ready for the model
            #batch_targets = torch.stack(targets)
            batch_targets = targets.to(device)
            
            # Initialize the hidden and cell state with zeros
            ho_1 = torch.zeros(lstm_layers, batchsize, hidden_nodes)
            ho_1 = ho_1.to(device)
            co_1 = torch.zeros(lstm_layers, batchsize, hidden_nodes)
            co_1 = co_1.to(device)
            
            ho_2 = torch.zeros(lstm_layers, batchsize, hidden_nodes)
            ho_2 = ho_2.to(device)
            co_2 = torch.zeros(lstm_layers, batchsize, hidden_nodes)
            co_2 = co_2.to(device)
            
            predictions, hidden_layer1, hidden_layer2 = model(batch_sent1, batch_sent2, sent1_len.to(device), sent2_len.to(device), (ho_1, co_1), (ho_2, ho_2))
            
            # Transpose the predictions and calculate the loss using Cross Entropy Error
            predictions = predictions.view(batchsize,3)
            
            loss = loss_function(predictions, batch_targets)
            
            loss.backward()
            
            optimizer.step()
            
            
            if iters % 20 == 0:
                print(loss)
        
        #print(loss)
            
    
    
    
    
train()
    
