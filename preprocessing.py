import numpy as np
import re
import pickle

def get_words(vocabulary, dataset):
    
    dataset = dataset.readlines()
    
    for line in dataset[1:]:
        line = line.decode('utf-8')
        tmp = re.split(r'\t+', line)
        sent = tmp[5] + " " + tmp[6]
        sent = re.sub(r'[^\w\s]','',sent)        
        
        words = sent.split(" ")
        
        for word in words:
            word = word.lower()
            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1
    return vocabulary
        
def make_embed_matrix(vocabulary, dataset):
    
    #embedding_matrix = np.empty(shape=(0,0))
    embedding_matrix = []
    word_index = {}
    ind = 0
    glove_vocab = []
    
    for line in dataset:
        
        # Make line into string to be able to split
        line = line.decode('utf-8')
        
        # Split the line on spaces
        split_line = line.split(" ")
        
        word = split_line[0]
        glove_vocab.append(word)
        
        # If the word is in the vocabulary, retrieve the embedding
        if word in vocabulary:
            embedding_matrix.append(split_line[1:])
            word_index[word] = ind
            ind += 1
            
        
    embedding_matrix = np.asarray(embedding_matrix, dtype=np.float32)
    print("Words with embedding: ", len(word_index))
    print("Size of embedding_matrix: ", embedding_matrix.shape)
    
    return embedding_matrix, word_index, glove_vocab
        
        

glove = open("../glove.840B.300d/glove.840B.300d.txt", 'rb')
snli_sentences_train = open("../dataset-sts/data/rte/snli/snli_1.0/snli_1.0_train.txt", 'rb')
snli_sentences_test = open("../dataset-sts/data/rte/snli/snli_1.0/snli_1.0_test.txt", "rb")
snli_sentences_val = open("../dataset-sts/data/rte/snli/snli_1.0/snli_1.0_dev.txt", "rb")

vocab = {}
snli_vocab = get_words(vocab, snli_sentences_train)
print("making train vocabulary done")
print("vocabulary length: ", len(snli_vocab))
snli_vocabulary = get_words(snli_vocab, snli_sentences_test)
print("making train + test vocabulary done")
print("vocabulary length: ", len(snli_vocabulary))
all_snli_vocab = get_words(snli_vocabulary, snli_sentences_val)
print("making train + test + val vocabulary done")
print("vocabulary length: ", len(all_snli_vocab))

embeddings, word2index, glove_vocab = make_embed_matrix(all_snli_vocab, glove)

remove = [word for word in all_snli_vocab if word not in glove_vocab]
for item in remove:
    del all_snli_vocab[item]
    
print("length of word2index: ", len(word2index))
print("Shape of embeddings: ", embeddings.shape)
print("vocabulary length after comparing with glove: ", len(all_snli_vocab.keys()))
#with open("word2index.pickle", "wb") as word2index_file:
    #pickle.dump(word2index, word2index_file)
#with open("vocab_snli_train.pickle", 'wb') as vocab_file:
    #pickle.dump(all_snli_vocab, vocab_file)
    
#with open("embeddings_train.pickle", 'wb') as embed_file:
    #pickle.dump(embeddings, embed_file)
