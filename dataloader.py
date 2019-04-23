from torch.utils.data import Dataset, DataLoader
import re
import pickle
import numpy as np
import string

class SentenceData1(Dataset):
    '''
    
    Tryout for dataset NLS project
    
    '''  
    
    
    def __init__(self, snli_dir, embed_matrix, w2i, transform=None):
        
        self.snli_dir = snli_dir
        self.transform = transform
        #self.vocab = vocab
        self.w2i = w2i
        self.embed_matrix = embed_matrix
        self.sentences, self.max_length = self.get_sentences(self.snli_dir)
        
        
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        
        sentence1 = self.sentences[idx][0]
        sentence2 = self.sentences[idx][2]
        
        embed1, embed2 = self.make_embeddings(self.embed_matrix, self.sentences[idx])
        
        return embed1, embed2, self.sentences[idx], self.max_length
    
    def get_sentences(self, directory):
        
        all_sentences = []
        snli_data = open(directory, 'rb')
        max_length = 0        
        read = 0
        
        amount_sentences = 0
        
        for line in snli_data:
            
            #if read > 0:
            if read > 0 and amount_sentences <= 6400:
                
                data = []
                
                # Split the line on tabs
                tmp = re.split(r'\t+', line.decode('utf-8'))
                
                # Get separate sentences
                sent1 = re.sub(r'[^\w\s]','',tmp[5].lower())
                sent2 = re.sub(r'[^\w\s]','',tmp[6].lower())
                
                # Get target
                if tmp[0] == 'entailment':
                    target = 0
                elif tmp[0] == 'neutral':
                    target = 1
                elif tmp[0] == 'contradiction':
                    target = 2
                
                # Retrieve separate words
                words1 = sent1.split(" ")
                words2 = sent2.split(" ")
                
                # Throw away words that are not in the vocabulary
                for word in words1:
                    if word not in self.w2i:
                        words1.remove(word)
                
                for word in words2:
                    if word not in self.w2i:
                        if word == "garagetype":
                            words2.remove(word)
                        words2.remove(word)
                
                # Retrieve max sentence length
                #lengths = [max_length]
                
                #if len(words1) > 10 and len(words1) < 15:
                    
                lengths = [max_length, len(words1), len(words2)]
                max_length = max(lengths)
                
                if len(words1) != 0 and len(words2) != 0:
                
                    # Put data together
                    data.append(words1)
                    data.append(len(words1))
                    data.append(words2)
                    data.append(len(words2))
                    data.append(target)
                    all_sentences.append(data)
            
            amount_sentences += 1
            
            read = 1
        
        for sentence in all_sentences:
            
            added_padding1 = max_length - len(sentence[0])
            added_padding2 = max_length - len(sentence[2])
            
            for i in range(added_padding1):
                sentence[0].append("PAD")
            for i in range(added_padding2):
                sentence[2].append("PAD")
        
        print("Max length: ", max_length)
        print("Amount of sentences: ", len(all_sentences))
        return all_sentences, max_length
    
    #def get_sentences(self, directory):
        
        #all_sentences = []
        #snli_data = open(directory, 'rb')
        
        #lines = snli_data.readlines()
        
        #max_length = 0
        
        #for line in lines[1:]:
            
            #data = []
            
            ## Decode into string
            #line = line.decode('utf-8')
            
            ## Split the line on tabs
            #tmp = re.split(r'\t+', line)
            
            ## Get separate sentences
            ##sent1 = tmp[5]
            #sent1 = re.sub(r'[^\w\s]','',tmp[5])
            #sent1 = sent1.lower()
            ##sent2 = tmp[6]
            #sent2 = re.sub(r'[^\w\s]','',tmp[6])
            #sent2 = sent2.lower()
            
            ## Get target
            #target = tmp[0]
            
            ## Retrieve separate words
            #words1 = sent1.split(" ")
            #words2 = sent2.split(" ")
            
            ## Throw away words that are not in the vocabulary
            #for word in words1:
                #if word not in self.w2i:
                    #words1.remove(word)
                    
            
            #for word in words2:
                #if word not in self.w2i:
                    #words2.remove(word)
            
            ## Retrieve max sentence length
            #lengths = [max_length, len(words1), len(words2)]
            #max_length = max(lengths)
            
            ## Put data together
            #data.append(words1)
            #data.append(words2)
            #data.append(target)
            #all_sentences.append(data)
            
        #return all_sentences, max_length
    
    def make_embeddings(self, embeddings, sentences):
        
        # Empty array for sentence embedding
        sentence1_embed = np.zeros(shape=(300, self.max_length))
        sentence2_embed = np.zeros(shape=(300, self.max_length))
        
        # Get first and second sentence
        for sentence in sentences:
            first = sentences[0]
            second = sentences[2]
        
        # Loop through all words in both sentences and append them 
        for i, word in enumerate(first):
            word_embedding = []
            if word in self.w2i:
                ind = self.w2i[word]
                word_embedding = embeddings[ind]
            elif word == 'PAD':
                word_embedding = np.zeros(shape=(300))
                
            # WHAT TO DO WHAT TO DO
            else:
                word_embedding = np.zeros(shape=(300))
                
            if len(word_embedding) == 0:
                print(first)
                print(word)
            sentence1_embed[:,i] = word_embedding
            #embeds = np.expand_dims(word_embedding, axis=1)
            #sentence1_embed[:,i] = embed
        
        for i, word in enumerate(second):
            word_embedding = []
            if word in self.w2i:
                ind = self.w2i[word]
                word_embedding = embeddings[ind]
            elif word == 'PAD':
                word_embedding = np.zeros(shape=(300))
                
            # WHAT TO DO WHAT TO DO
            else:
                word_embedding = np.zeros(shape=(300))
            sentence2_embed[:,i] = word_embedding
            #embeds = np.expand_dims(word_embedding, axis=1)
            #sentence2_embed[:,i] = embeds
            
            
        return sentence1_embed, sentence2_embed
    

class SentenceData2(Dataset):
    '''
    
    Tryout for dataset NLS project
    
    '''  
    
    
    def __init__(self, snli_dir, vocab, transform=None):
        self.snli_dir = snli_dir
        self.transform = transform
        self.vocab = vocab
        self.sentences, self.max_length = self.get_sentences(self.snli_dir)
        
        
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx][0], self.sentences[idx][1], self.sentences[idx][2]
    
    def get_sentences(self, directory):
        
        all_sentences = []
        snli_data = open(directory, 'rb')
        max_length = 0        
        read = 0
        
        for line in snli_data:
            
            if read > 0:
            
                data = []
                
                # Split the line on tabs
                tmp = re.split(r'\t+', line.decode('utf-8'))
                
                # Get separate sentences
                sent1 = re.sub(r'[^\w\s]','',tmp[5].lower())
                sent2 = re.sub(r'[^\w\s]','',tmp[6].lower())
                
                # Get target
                if tmp[0] == 'entailment':
                    target = 0
                elif tmp[0] == 'neutral':
                    target = 1
                elif tmp[0] == 'contradiction':
                    target = 2
                    
                # Retrieve separate words
                words1 = sent1.split(" ")
                words2 = sent2.split(" ")
                
                # Throw away words that are not in the vocabulary
                for word in words1:
                    if word not in self.vocab:
                        words1.remove(word)
                
                for word in words2:
                    if word not in self.vocab:
                        words2.remove(word)
                
                # Retrieve max sentence length
                lengths = [max_length, len(words1), len(words2)]
                max_length = max(lengths)
                
                # Put data together
                data.append(words1)
                data.append(len(words1))
                data.append(words2)
                data.append(len(words2))
                data.append(target)
                all_sentences.append(data)
            
            read = 1
        
        for sentence in all_sentences:
            
            added_padding1 = max_length - len(sentence[0])
            added_padding2 = max_length - len(sentence[2])
            
            for i in range(added_padding1):
                sentence[0].append("PAD")
            for i in range(added_padding2):
                sentence[2].append("PAD")
                
        return all_sentences, max_length
        
        
if __name__ == '__main__':
    
    pickle_embed = open("./embeddings_train.pickle","rb")
    embedding_matrix = pickle.load(pickle_embed)
    pickle_vocab = open("./vocab_snli_train.pickle", "rb")
    vocabulary = pickle.load(pickle_vocab)
    pickle_w2i = open("./word2index.pickle", "rb")
    word2index = pickle.load(pickle_w2i)
    
    # Returns embeddings
    dataset1 = SentenceData1("./dataset-sts/data/rte/snli/snli_1.0/snli_1.0_train.txt", embedding_matrix, word2index)
    
    # Return sentences
    dataset2 = SentenceData2("./dataset-sts/data/rte/snli/snli_1.0/snli_1.0_train.txt", word2index)
    dataloader = DataLoader(dataset1, batch_size=10, shuffle=False)
    
    
    for sentences in dataloader:
        # Print all output of dataloader batch
        #print(sentences)
        
        # Print embedding first sentences
        #print(sentences[0])
        
        # Print embedding second sentences
        #print(sentences[1])
        
        # Print sentence data
        #print(sentences[2])
        
        # Print first sentences
        #print(sentences[2][0])
        
        # Print first sentence lengths
        #print(sentences[2][1])
        
        # Print second sentences
        #print(sentences[2][2])
        
        # Print second sentence lengths
        #print(sentences[2][3])
        
        # Print targets
        #print(sentences[2][4])
        
        # Print 
        #print(sentences[2][1])
        #print(sentences[2][3])
        quit()
