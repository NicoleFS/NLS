import torch.nn as nn
import torch
import numpy as np

class ModelLSTM(nn.Module):
        
    def __init__(self, input_dim, lstm_num_hidden, lstm_num_layers, output_dim, device, pad_packed):

        super(ModelLSTM, self).__init__()
        
        # Initialize the LSTM using the PyTorch LSTM module
        self.lstm = nn.LSTM(input_dim, lstm_num_hidden, lstm_num_layers)
        self.pad_packed = pad_packed
        self.device = device
        
        self.classification = nn.Sequential(
            nn.Linear(2*lstm_num_hidden, 512),
            nn.Linear(512, 512),
            nn.Linear(512, output_dim)
            )
    
    def forward(self, x1, x2, x1_lens, x2_lens, hidden_state1, hidden_state2):
        
        hidden_state1[0].requires_grad_()
        hidden_state1[1].requires_grad_()
        hidden_state2[0].requires_grad_()
        hidden_state2[1].requires_grad_()
        
        if self.pad_packed: 
            
            # Sort the embedding matrix of the first sentences according to the new indices
            sort_x1, ind_1 = np.sort(x1_lens)[::-1], np.argsort(-x1_lens)
            ind_1 = torch.LongTensor(ind_1).to(self.device)
            x1 = x1.index_select(1, ind_1)#torch.cuda.LongTensor(idx_sort))
            
            # Sort the lengths of second sentences in decreasing order and keep the switch in indices
            sort_x2, ind_2 = np.sort(x2_lens)[::-1], np.argsort(-x2_lens)
            ind_2 = torch.LongTensor(ind_2).to(self.device)
            
            # Sort the embedding matrix of the first sentences according to the new indices
            x2 = x2.index_select(1, ind_2)
            
            x1_packed = nn.utils.rnn.pack_padded_sequence(x1, sort_x1)
            x2_packed = nn.utils.rnn.pack_padded_sequence(x2, sort_x2)
            
            # Calculate the output of the LSTM and the hidden and cell state
            out_tmp1, (h1,c1) = self.lstm(x1_packed, (hidden_state1[0], hidden_state1[1]))
            out_tmp2, (h2,c2) = self.lstm(x2_packed, (hidden_state2[0], hidden_state2[1]))
            
            # Unsort the hidden states
            ind1_unsort = np.argsort(ind_1)        
            ind1_unsort = torch.LongTensor(ind1_unsort).to(self.device)
            h1 = h1.index_select(1, ind1_unsort)
            
            ind2_unsort = np.argsort(ind_2)        
            ind2_unsort = torch.LongTensor(ind2_unsort).to(self.device)        
            h2 = h2.index_select(1, ind2_unsort)
        
        else:
            out_tmp1, (h1,c1) = self.lstm(x1, (hidden_state1[0], hidden_state1[1]))
            out_tmp2, (h2,c2) = self.lstm(x2, (hidden_state2[0], hidden_state2[1]))
        
        out_tmp = torch.cat((h1, h2), 2)
        out = self.classification(out_tmp)
        
        return out, (h1,c1), (h2,c2)
