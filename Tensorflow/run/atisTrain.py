import os
import sys
sys.path.insert(0,'./util')
sys.path.insert(0,'./models')
if sys.version_info.major==2:
    import cPickle as pickle
else:
    import pickle
import numpy as np

import RNNs
import xmlParser
import atisDataManager

if __name__=='__main__':

    if len(sys.argv)!=2:
        print('Usage: python atisTrain.py <config>')
        exit(0)

    hyper_params=xmlParser.parse(sys.argv[1],flat=False)

    # Construct data manager
    data_manager_params=hyper_params['data_manager']
    my_data_manager=atisDataManager.atisDataManager(data_manager_params)
    my_data_manager.load()

    # Construct RNN network model
    rnn_network_params=hyper_params['rnn_network']
    if rnn_network_params['pretrain_embedding']==True:
        rnn_network_params['embedding_matrix']=my_data_manager.embedding_matrix
    my_rnn_model=RNNs.N2NRNNs(rnn_network_params)
    batch_size=my_rnn_model.batch_size

    # Configure tha training parameters and start training
    train_params=hyper_params['train']
    batches=train_params['batches']
    check_err_frequency=train_params['check_err_frequency']
    do_test_frequency=train_params['do_test_frequency']
    model_saved_folder=train_params['model_saved_folder']

    train_err_ckpt={}
    test_err_ckpt={}

    if not os.path.exists(model_saved_folder):
        os.makedirs(model_saved_folder)

    my_rnn_model.train_validate_test_init()
    my_data_manager.set_initialization(set_label='train',permutation=True)
    train_loss_list=[]
    for batch_idx in xrange(batches):
        inputs,masks,labels,_=my_data_manager.batch_gen(set_label='train',batch_size=batch_size)
        _,loss_this_batch=my_rnn_model.train(inputs,masks,labels)
        result,=my_rnn_model.debug(inputs,masks,labels)
        train_loss_list.append(loss_this_batch)
        sys.stdout.write('Batch_idx = %d/%d, loss = %.4f\r'%(batch_idx+1,batches,loss_this_batch))

        if (batch_idx+1)%check_err_frequency==0:
            print('Average loss in [%d,%d) = %.4f'%(batch_idx+1-check_err_frequency,batch_idx+1,np.mean(train_loss_list[-check_err_frequency:])))
            train_err_ckpt[batch_idx+1]=np.mean(train_loss_list[-check_err_frequency:])

        if (batch_idx+1)%do_test_frequency==0:
            my_rnn_model.dump_params(file2dump=model_saved_folder+os.sep+'%s_%d.ckpt'%(my_rnn_model.name,batch_idx+1))

            test_loss_list=[]
            my_data_manager.set_initialization(set_label='test',permutation=True)
            end_of_epoch=False
            while not end_of_epoch:
                inputs,masks,labels,end_of_epoch=my_data_manager.batch_gen(set_label='test',batch_size=batch_size)
                _,loss_this_batch=my_rnn_model.validate(inputs,masks,labels)
                test_loss_list.append(loss_this_batch)
                sys.stdout.write('Test phrase .. current batch loss = %.4f, average loss = %.4f\r'%(loss_this_batch,np.mean(test_loss_list)))
            print('Test phrase completed! Average loss = %.4f               '%np.mean(test_loss_list))
            test_err_ckpt[batch_idx+1]=np.mean(test_loss_list)

    my_rnn_model.train_validate_test_end()
    pickle.dump({'train':train_err_ckpt,'test':test_err_ckpt},open(model_saved_folder+os.sep+'%s.pkl'%my_rnn_model.name,'w'))

