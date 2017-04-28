import sys
sys.path.insert(0,'./util')
import numpy as np
import tensorflow as tf

import sharp

'''
>>> multi-layer Recurrent Neural Network
>>> many2many mode
'''
class N2NRNNs(object):

    '''
    >>> Constructor
    >>> neurons: list<int>, number of neurons in each layer
    >>> nonlinearity: list<str>, activation function in each layer
    '''
    def __init__(self,hyper_params):
        self.name=hyper_params['name']
        self.neurons=hyper_params['neurons']
        self.nonlinearity=hyper_params['nonlinearity']
        self.batch_size=hyper_params['batch_size']
        self.max_sequence_length=hyper_params['max_sequence_length']
        self.embedding_dim=hyper_params['embedding_dim']
        self.window_size=hyper_params['window_size']
        self.vocab_size=hyper_params['vocab_size'] if 'vocab_size' in hyper_params else None
        self.update_policy=hyper_params['update_policy']
        self.grad_clip_norm=hyper_params['grad_clip_norm'] if 'grad_clip_norm' in hyper_params else 1.0
        self.sess=None

        # Load structure information
        self.all_layers=len(self.neurons)
        self.hidden_layers=len(self.neurons)-2
        self.input_size=self.neurons[0]
        self.hidden_size=self.neurons[1:-1]
        self.output_size=self.neurons[-1]
        if type(self.nonlinearity)==str:
            self.nonlinearity=[self.nonlinearity,]*self.hidden_layers
        assert(self.window_size*self.embedding_dim==self.input_size)

        # Load step size
        self.step_size={'input_matrix':[1.,]*self.hidden_layers,'recurrent_matrix':[1.,]*self.hidden_layers,'hidden_bias':[1.,]*self.hidden_layers,
            'init_state':[1.,]*self.hidden_layers,'embedding_matrix':1.,'output_matrix':1.,'output_bias':1.}
        if 'step_size' in hyper_params:
            self.__load_step_size__(hyper_params['step_size'])

        self.inputs=tf.placeholder(tf.int32,shape=[self.batch_size,self.max_sequence_length,self.window_size])
        self.masks=tf.placeholder(tf.int32,shape=[self.batch_size,self.max_sequence_length])
        self.labels=tf.placeholder(tf.int32,shape=[self.batch_size,self.max_sequence_length])

        if 'pretrain_embedding' in hyper_params and hyper_params['pretrain_embedding']==True and 'embedding_matrix' in hyper_params:
            print('pretrained word embeddings are imported')
            self.embedding_matrix=tf.Variable(hyper_params['embedding_matrix'],dtype=tf.float32)
            if self.vocab_size!=None:
                assert(self.vocab_size==hyper_params['embedding_matrix'].shape[0])
            else:
                self.vocab_size=hyper_params['embedding_matrix'].shape[0]
            assert(self.embedding_dim==hyper_params['embedding_matrix'].shape[1])
        else:
            print('word embeddings are initialized from scratch')
            if self.vocab_size==None:
                raise ValueError('vocab_size must be specified when initializing embedding_matrix from scratch')
            self.embedding_matrix=tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_dim],-1.0,1.0),dtype=tf.float32)

        self.embedding_output=tf.nn.embedding_lookup(self.embedding_matrix,self.inputs)   # of shape [self.batch_size,self.max_sequence_length,self.window_size,self.embedding_dim]
        self.embedding_output=tf.reshape(self.embedding_output,shape=[self.batch_size,self.max_sequence_length,self.window_size*self.embedding_dim])

        input_slices=tf.split(self.embedding_output,self.max_sequence_length,1)           # a list of tensors of shape [self.batch_size,1,self.window_size*self.embedding_dim] * self.max_sequence_length
        for idx in xrange(self.max_sequence_length):                                      # a list of tensors of shape [self.batch_size,self.window_size*self.embedding_dim] * self.max_sequence_length
            input_slices[idx]=tf.reshape(input_slices[idx],shape=[self.batch_size,self.window_size*self.embedding_dim])
        output_slices=[]                                                                  # a list of tensors of shape [self.batch_size,self.output_size] * self.max_sequence_length

        # Constructing the neural network by feeding the first token
        sys.stdout.write('Constructing the network\r')
        self.rnn_cell_list=[]            # a list of rnn cells of input dim of I and hidden dim of H
        self.rnn_var_init_state=[]       # a list of init states of all rnn cells
        self.rnn_var_weight=[]           # a list of weights of (of size (H+I)*H, U=X[:I].T, W=X[:H].T) all rnn cells
        self.rnn_var_bias=[]             # a list of bias of the rnn cells (of size H)
        rnn_state_list=[]                # a list of current states of all rnn
        current_slice=input_slices[0]    # of shape [self.batch_size, self.window_size*self.embedding_dim]
        with tf.variable_scope('RNN') as scope:
            for idx,neuron_num in enumerate(self.hidden_size):
                if idx<len(self.nonlinearity):
                    activation_func={'relu':tf.nn.relu,'tanh':tf.tanh,'sigmoid':tf.sigmoid,'sigd':tf.sigmoid}[self.nonlinearity[idx].lower()]
                else:
                    print('For hidden layer %d, use tanh as default activation_func'%(idx+1))
                    activation_func='tanh'
                rnn_cell_this_layer=tf.contrib.rnn.BasicRNNCell(neuron_num,activation=activation_func)
                init_state=tf.get_variable(name='init_%d'%(idx+1),
                    initializer=tf.constant(np.zeros([neuron_num,],dtype=np.float32)),dtype=tf.float32)
                state=tf.matmul(tf.ones([self.batch_size,1],dtype=tf.float32),tf.reshape(init_state,shape=[1,neuron_num]))
                current_slice,state=rnn_cell_this_layer(current_slice,state)
                self.rnn_cell_list.append(rnn_cell_this_layer)
                rnn_state_list.append(state)
                self.rnn_var_init_state.append(init_state)
                for variable in tf.global_variables():
                    if not variable in self.rnn_var_init_state+self.rnn_var_weight+self.rnn_var_bias+[self.embedding_matrix,]:
                        dimension=len(variable.value().shape)
                        if dimension==1:
                            self.rnn_var_bias.append(variable)
                        elif dimension==2:
                            self.rnn_var_weight.append(variable)
                        else:
                            raise ValueError('There should not be a tensor whose dimension is neither 1d nor 2d')
                assert(len(self.rnn_cell_list)==idx+1)
                assert(len(self.rnn_var_init_state)==idx+1)
                assert(len(self.rnn_var_weight)==idx+1)
                assert(len(self.rnn_var_bias)==idx+1)
        with tf.variable_scope('Classifier') as scope:
            self.output_matrix=tf.get_variable(name='output_matrix',shape=[self.hidden_size[-1],self.output_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.output_bias=tf.get_variable(name='output_bias',shape=[self.output_size,],
                initializer=tf.truncated_normal_initializer(stddev=0.01))
            output_slices.append(tf.add(tf.matmul(current_slice,self.output_matrix),self.output_bias))

        for current_slice in input_slices[1:]:
            with tf.variable_scope('RNN',reuse=True) as scope:
                for idx,neuron_num in enumerate(self.hidden_size):
                    rnn_cell_this_layer=self.rnn_cell_list[idx]
                    current_slice,state=rnn_cell_this_layer(current_slice,rnn_state_list[idx])
                    rnn_state_list[idx]=state
            with tf.variable_scope('Classifier',reuse=True) as scope:
                output_slices.append(tf.add(tf.matmul(current_slice,self.output_matrix),self.output_bias))

        for idx,output_slice in enumerate(output_slices):
            output_slices[idx]=tf.reshape(output_slice,shape=[self.batch_size,1,self.output_size])
        self.output=tf.concat(output_slices,axis=1)            # of shape [self.batch_size, self.max_sequence_length, self.output_size]
        self.output=tf.nn.softmax(self.output,dim=-1)           # softmax normalization
        self.prediction=tf.argmax(self.output,axis=2)          # of shape [self.batch_size, self.max_sequence_length]

        embedded_label=tf.nn.embedding_lookup(tf.eye(self.output_size),self.labels)         # of shape [self.batch_size, self.max_sequence_length, self.output_size]
        unnomaralized_loss=-tf.multiply(embedded_label,tf.log(self.output))                 # of shape [self.batch_size, self.max_sequence_length, self.output_size]
        unnomaralized_loss=tf.reduce_sum(unnomaralized_loss,axis=2)                         # of shape [self.batch_size, self.max_sequence_length]
        normalized_loss=tf.multiply(unnomaralized_loss,tf.cast(self.masks,tf.float32))
        self.loss=tf.reduce_sum(normalized_loss)/tf.reduce_sum(tf.cast(self.masks,tf.float32))
        print('Network structure constructed')

        sys.stdout.write('Constructing the optimizer\r')
        self.__optimizer_construction__(policy=self.update_policy)
        print('Optimizer construction completed!')

    '''
    >>> learning_rate decay
    '''
    def learning_rate_decay(self,ratio):
        current_learning_rate=self.update_policy['learning_rate']
        self.update_policy['learning_rate']*=ratio
        modified_learning_rate=self.update_policy['learning_rate']
        self.__optimizer_construction__(policy=self.update_policy)
        print('Learning rate decay %.4f -> %.4f'%(current_learning_rate,modified_learning_rate))

    '''
    >>> construct the optimizer
    '''
    def __optimizer_construction__(self,policy):
        update_policy_name=policy['name']
        if update_policy_name.lower() in ['sgd','gradient_descent']:
            learning_rate=policy['learning_rate']
            print('We use a sgd optimizer with learning rate %.4f'%learning_rate)
            optimizer=tf.train.GradientDescentOptimizer(learning_rate)
            gradients=optimizer.compute_gradients(self.loss)
            clipped_gradients=[(tf.clip_by_value(grad,-self.grad_clip_norm,self.grad_clip_norm),var) for grad,var in gradients]
            clipped_gradients=[self.__query_step_size__(grad,var) for grad,var in clipped_gradients]
            self.update=optimizer.apply_gradients(clipped_gradients)
        elif update_policy_name.lower() in ['ssd','spectral_descent']:
            learning_rate=policy['learning_rate']
            print('We use a ssd optimizer with learning rate %.4f'%learning_rate)
            optimizer=tf.train.GradientDescentOptimizer(learning_rate)
            gradients=optimizer.compute_gradients(self.loss)
            clipped_gradients=[]
            for grad,var in gradients:
                if grad==None:
                    raise Exception('Error: there exists a disconnected node (%s) of the loss function'%var.name)
                elif var in self.rnn_var_init_state+self.rnn_var_bias+[self.output_matrix,self.output_bias,self.embedding_matrix]:         # Use traditional gradient descent
                    clipped_grad=tf.clip_by_value(grad,-self.grad_clip_norm,self.grad_clip_norm)
                    clipped_gradients.append((clipped_grad,var))
                elif var in self.rnn_var_weight:
                    layer_index=self.rnn_var_weight.index(var)
                    input_dim=self.neurons[layer_index]
                    hidden_dim=self.neurons[layer_index+1]
                    assert(var.value().shape==[input_dim+hidden_dim,hidden_dim])
                    input_matrix_sharp,input_matrix_sigular_value=sharp.sharp(grad[:input_dim])          # of shape [input_dim, hidden_dim]
                    recur_matrix_sharp,recur_matrix_sigular_value=sharp.sharp(grad[input_dim:])          # of shape [hidden_dim, hidden_dim]
                    input_matrix_sharp=tf.clip_by_value(input_matrix_sharp,-self.grad_clip_norm,self.grad_clip_norm)
                    recur_matrix_sharp=tf.clip_by_value(recur_matrix_sharp,-self.grad_clip_norm,self.grad_clip_norm)
                    # input_matrix_sharp/=tf.reduce_sum(input_matrix_sigular_value)
                    # recur_matrix_sharp/=tf.reduce_sum(recur_matrix_sigular_value)
                    grad_sharp=tf.concat([input_matrix_sharp,recur_matrix_sharp],axis=0)
                    clipped_gradients.append((grad_sharp,var))
                else:
                    raise Exception('Untracked variable: (%s)'%var.name)
            clipped_gradients=[self.__query_step_size__(grad,var) for grad,var in clipped_gradients]
            self.update=optimizer.apply_gradients(clipped_gradients)
        else:
            raise ValueError('Unrecognized update policy name: %s'%update_policy_name)

    '''
    >>> load step_size information
    '''
    def __load_step_size__(self,step_size_info):
        if 'input_matrix' in step_size_info:
            input_matrix_lr=step_size_info['input_matrix']
            self.step_size['input_matrix']=input_matrix_lr if type(input_matrix_lr)==list else [input_matrix_lr,]*self.hidden_layers
        assert(len(self.step_size['input_matrix'])==self.hidden_layers)
        if 'recurrent_matrix' in step_size_info:
            recurrent_matrix_lr=step_size_info['recurrent_matrix']
            self.step_size['recurrent_matrix']=recurrent_matrix_lr if type(recurrent_matrix_lr)==list else [recurrent_matrix_lr,]*self.hidden_layers
        assert(len(self.step_size['recurrent_matrix'])==self.hidden_layers)
        if 'hidden_bias' in step_size_info:
            hidden_bias_lr=step_size_info['hidden_bias']
            self.step_size['hidden_bias']=hidden_bias_lr if type(hidden_bias_lr)==list else [hidden_bias_lr,]*self.hidden_layers
        assert(len(self.step_size['hidden_bias'])==self.hidden_layers)
        if 'init_state' in step_size_info:
            init_state_lr=step_size_info['init_state']
            self.step_size['init_state']=init_state_lr if type(init_state_lr)==list else [init_state_lr,]*self.hidden_layers
        assert(len(self.step_size['init_state'])==self.hidden_layers)
        if 'embedding_matrix' in step_size_info:
            self.step_size['embedding_matrix']=step_size_info['embedding_matrix']
        if 'output_matrix' in step_size_info:
            self.step_size['output_matrix']=step_size_info['output_matrix']
        if 'output_bias' in step_size_info:
            self.step_size['output_bias']=step_size_info['output_bias']

    '''
    >>> query the step_size of a variable
    '''
    def __query_step_size__(self,grad,var):
        if grad==None:
            return grad,var

        if var in self.rnn_var_init_state:
            layer_index=self.rnn_var_init_state.index(var)
            step_size_this_var=self.step_size['init_state'][layer_index]
            grad=grad*step_size_this_var
        elif var in self.rnn_var_weight:                # contains two parts
            layer_index=self.rnn_var_weight.index(var)
            input_dim=self.neurons[layer_index]
            step_size_input_part=self.step_size['input_matrix'][layer_index]
            step_size_recurrent_part=self.step_size['recurrent_matrix'][layer_index]
            input_part=grad[:input_dim]*step_size_input_part
            recurrent_part=grad[input_dim:]*step_size_recurrent_part
            grad=tf.concat([input_part,recurrent_part],axis=0)
        elif var in self.rnn_var_bias:
            layer_index=self.rnn_var_bias.index(var)
            step_size_this_var=self.step_size['hidden_bias'][layer_index]
            grad=grad*step_size_this_var
        elif var in [self.output_matrix,]:
            step_size_this_var=self.step_size['output_matrix']
            grad=grad*step_size_this_var
        elif var in [self.output_bias,]:
            step_size_this_var=self.step_size['output_bias']
            grad=grad*step_size_this_var
        elif var in [self.embedding_matrix,]:
            step_size_this_var=self.step_size['embedding_matrix']
            grad=grad*step_size_this_var
        else:
            raise Exception('Untracked variable: (%s)'%var.name)
        return grad,var


    '''
    >>> initialize the network configuration to start training, validation and testing
    '''
    def train_validate_test_init(self):
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

    '''
    >>> for debug
    '''
    def debug(self,inputs,masks,labels):
        debug_dict={self.inputs:inputs,self.masks:masks,self.labels:labels}
        result,=self.sess.run([self.output],feed_dict=debug_dict)
        return result,

    '''
    >>> training phrase
    >>> inputs,masks,labels: np.array, inputs of the network
    >>> inputs, np.array of shape [None,self.max_sequence_length,self.window_size]
    >>> masks, np.array of shape [None,self.max_sequence_length]
    >>> labels, np.array of shape [None,self.max_sequence_length]
    '''
    def train(self,inputs,masks,labels):
        instance_num=inputs.shape[0]
        assert(instance_num==masks.shape[0])
        assert(instance_num==labels.shape[0])
        if instance_num<self.batch_size:
            _inputs=np.zeros([self.batch_size,self.max_sequence_length,self.window_size],dtype=np.int)
            _inputs[:instance_num]=inputs
            inputs=_inputs
            _masks=np.zeros([self.batch_size,self.max_sequence_length],dtype=np.int)
            _masks[:instance_num]=masks
            masks=_masks
            _labels=np.zeros([self.batch_size,self.max_sequence_length],dtype=np.int)
            _labels[:instance_num]=labels
            labels=_labels

        train_dict={self.inputs:inputs,self.masks:masks,self.labels:labels}
        self.sess.run(self.update,feed_dict=train_dict)
        prediction_this_batch,loss_this_batch=self.sess.run([self.prediction,self.loss],feed_dict=train_dict)
        return prediction_this_batch,loss_this_batch

    '''
    >>> validation phrase
    >>> inputs,masks,labels: np.array, inputs of the network
    '''
    def validate(self,inputs,masks,labels):
        instance_num=inputs.shape[0]
        assert(instance_num==masks.shape[0])
        assert(instance_num==labels.shape[0])
        if instance_num<self.batch_size:
            _inputs=np.zeros([self.batch_size,self.max_sequence_length,self.window_size],dtype=np.int)
            _inputs[:instance_num]=inputs
            inputs=_inputs
            _masks=np.zeros([self.batch_size,self.max_sequence_length],dtype=np.int)
            _masks[:instance_num]=masks
            masks=_masks
            _labels=np.zeros([self.batch_size,self.max_sequence_length],dtype=np.int)
            _labels[:instance_num]=labels
            labels=_labels

        validate_dict={self.inputs:inputs,self.masks:masks,self.labels:labels}
        prediction_this_batch,loss_this_batch=self.sess.run([self.prediction,self.loss],feed_dict=validate_dict)
        return prediction_this_batch,loss_this_batch

    '''
    >>> test phrase
    >>> inputs,masks: np.array, inputs of the network
    '''
    def test(self,inputs,masks):
        instance_num=inputs.shape[0]
        assert(instance_num==masks.shape[0])
        if instance_num<self.batch_size:
            _inputs=np.zeros([self.batch_size,self.max_sequence_length,self.window_size],dtype=np.int)
            _inputs[:instance_num]=inputs
            inputs=_inputs
            _masks=np.zeros([self.batch_size,self.max_sequence_length],dtype=np.int)
            _masks[:instance_num]=masks
            masks=_masks

        test_dict={self.inputs:inputs,self.masks:masks}
        prediction_this_batch,=self.sess.run([self.prediction],feed_dict=test_dict)
        return prediction_this_batch,

    '''
    >>> save the parameters
    >>> file2load: str, file containing the parameters to load into the workspace
    '''
    def load_params(self,file2load):
        saver=tf.train.Saver()
        saver.restore(self.sess,file2load)
        print('Parameters are imported from file %s'%file2load)

    '''
    >>> dump the parameters
    >>> file2dump: str, file to dump the parameters from the workspace
    '''
    def dump_params(self,file2dump):
        saver=tf.train.Saver()
        saved_path=saver.save(self.sess, file2dump)
        print('Parameters are saved in the file %s'%file2dump)

    '''
    >>> terminate the training, validation and test phrase
    '''
    def train_validate_test_end(self):
        self.sess.close()
        self.sess=None
