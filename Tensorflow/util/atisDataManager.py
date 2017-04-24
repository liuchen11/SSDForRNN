import sys
import numpy as np

'''
>>> data management of atis dataset
'''
class atisDataManager(object):

    '''
    >>> Constructor
    '''
    def __init__(self,hyper_params):
        self.max_sequence_length=hyper_params['max_sequence_length']
        self.window_size=hyper_params['window_size']
        self.embedding_dim=hyper_params['embedding_dim']
        self.embedding_file=hyper_params['embedding_file']
        self.idx_global2local={}
        self.embedding_matrix=None
        self.set_files={}
        self.set_pt={}
        for key in hyper_params['set_files']:
            self.set_files[key]=hyper_params['set_files'][key]      # a list of length 2: [content_file, label_file]
            self.set_pt[key]=0
        self.set_contents={}

    '''
    >>> load embedding matrix
    >>> load the content of different data subset
    '''
    def load(self):
        # Load the dictionary
        embedding_matrix=[]
        with open(self.embedding_file,'r') as fopen:
            for idx,line in enumerate(fopen):
                index_str,vector_str=line.split(':')
                index=int(index_str)
                vector=map(float,vector_str.split(','))
                embedding_matrix.append(vector)
                self.idx_global2local[index]=idx
            print('%d word embeddings are loaded'%len(embedding_matrix))
        embedding_matrix.append(np.random.randn(self.embedding_dim))                # for unknown words
        embedding_matrix.append(np.zeros([self.embedding_dim],dtype=np.float32))    # for padding
        self.embedding_matrix=np.array(embedding_matrix,dtype=np.float32)
        assert(self.embedding_matrix.shape[1]==self.embedding_dim)

        # Load the content
        for key in self.set_files:
            content_file,label_file=self.set_files[key]
            contents=[]                 # list of list of int, the content of a data subset
            with open(content_file,'r') as fopen:
                for line in fopen:
                    word_list=map(int,line.split(','))
                    word_list=map(lambda x:self.idx_global2local[x] if x in self.idx_global2local \
                     else self.embedding_matrix.shape[0]-2,word_list)
                    contents.append(word_list)
            labels=[]                   # list of list of int, the labels of a data subset
            with open(label_file,'r') as fopen:
                for line in fopen:
                    label_list=map(int,line.split(','))
                    labels.append(label_list)
            content_label_pairs=zip(contents,labels)
            print('There are %d sentences in data subset %s'%(len(content_label_pairs),key))
            self.set_contents[key]=content_label_pairs

    '''
    >>> subset, initialization
    '''
    def set_initialization(self,set_label,permutation):
        if not set_label in self.set_contents:
            raise ValueError('Invalid or not initialized set label: %s'%set_label)

        self.set_pt[set_label]=0
        if permutation==True:
            self.set_contents[set_label]=np.random.permutation(self.set_contents[set_label])

    '''
    >>> batch generation
    '''
    def batch_gen(self,set_label,batch_size):
        if not set_label in self.set_contents:
            raise ValueError('Invalid or not initialized set label: %s'%set_label)

        left_pad=int(self.window_size/2)
        right_pad=int((self.window_size-1)/2)

        inputs=np.zeros([batch_size,self.max_sequence_length,self.window_size],dtype=np.int)
        inputs.fill(self.embedding_matrix.shape[0]-1)                   # fill with padding index
        masks=np.zeros([batch_size,self.max_sequence_length],dtype=np.int)
        labels=np.zeros([batch_size,self.max_sequence_length],dtype=np.int)

        end_of_epoch=False
        for batch_idx in xrange(batch_size):
            contents,ground_truth=self.set_contents[set_label][self.set_pt[set_label]]
            content_label_pair=zip(contents,ground_truth)
            if len(content_label_pair)>self.max_sequence_length:
                content_label_pair=content_label_pair[:self.max_sequence_length]
            for idx,(content,label) in enumerate(content_label_pair):
                left_idx=max(0,idx-left_pad)
                right_idx=min(len(content_label_pair),idx+right_pad)
                input_begin_idx=left_idx-(idx-left_pad)
                input_end=right_idx-(idx-left_pad)
                inputs[batch_idx,idx,input_begin_idx:input_end]=contents[left_idx:right_idx]
                masks[batch_idx,idx]=1
                labels[batch_idx,idx]=label

            self.set_pt[set_label]+=1
            if len(self.set_contents[set_label])==self.set_pt[set_label]:
                end_of_epoch=True
                self.set_initialization(set_label=set_label,permutation=True)

        return inputs,masks,labels,end_of_epoch



