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
    do_analyze_var=train_params['do_analyze_var'] if 'do_analyze_var' in train_params else True
    lr_decay_frequency=train_params['lr_decay_frequency'] if 'lr_decay_frequency' in train_params else None
    lr_decay_ratio=train_params['lr_decay_ratio'] if 'lr_decay_ratio' in train_params else None
    if lr_decay_frequency<=0:
        lr_decay_frequency=None
    if lr_decay_ratio<=0:
        lr_decay_ratio=None

    train_err_ckpt={}
    test_err_ckpt={}
    var_analysis_ckpt={}

    if not os.path.exists(model_saved_folder):
        os.makedirs(model_saved_folder)
    if do_analyze_var==True:
        print('Variable analysis is ENABLED')
    else:
        print('Variable analysis is DISABLED')

    my_rnn_model.train_validate_test_init()
    my_data_manager.set_initialization(set_label='train',permutation=True)
    train_loss_list=[]
    for batch_idx in xrange(batches):
        inputs,masks,labels,_=my_data_manager.batch_gen(set_label='train',batch_size=batch_size)
        _,loss_this_batch=my_rnn_model.train(inputs,masks,labels)
        train_loss_list.append(loss_this_batch)
        sys.stdout.write('Batch_idx = %d/%d, loss = %.4f\r'%(batch_idx+1,batches,loss_this_batch))

        if (batch_idx+1)%check_err_frequency==0:
            print('Average loss in [%d,%d) = %.4f'%(batch_idx+1-check_err_frequency,batch_idx+1,np.mean(train_loss_list[-check_err_frequency:])))
            train_err_ckpt[batch_idx+1]=np.mean(train_loss_list[-check_err_frequency:])
            if do_analyze_var==True:
                var_analysis_report=my_rnn_model.analyze_var(inputs,masks,labels)
                var_analysis_ckpt[batch_idx+1]=var_analysis_report

        if lr_decay_frequency!=None and (batch_idx+1)%lr_decay_frequency==0:
            my_rnn_model.learning_rate_decay(ratio=lr_decay_ratio)

        if (batch_idx+1)%do_test_frequency==0:
            my_rnn_model.dump_params(file2dump=model_saved_folder+os.sep+'%s_%d.ckpt'%(my_rnn_model.name,batch_idx+1))

            test_loss_list=[]
            my_data_manager.set_initialization(set_label='test',permutation=True)
            end_of_epoch=False
            while not end_of_epoch:
                inputs,masks,labels,end_of_epoch=my_data_manager.batch_gen(set_label='test',batch_size=1)
                _,loss_this_batch=my_rnn_model.validate(inputs,masks,labels)
                test_loss_list.append(loss_this_batch)
                sys.stdout.write('Test phrase .. current batch loss = %.4f, average loss = %.4f\r'%(loss_this_batch,np.mean(test_loss_list)))
            print('Test phrase completed! Average loss = %.4f               '%np.mean(test_loss_list))
            test_err_ckpt[batch_idx+1]=np.mean(test_loss_list)

    my_rnn_model.train_validate_test_end()
    saved_dict={'train':train_err_ckpt, 'test':test_err_ckpt, 'config':hyper_params,'var_analysis':var_analysis_ckpt}
    saved_dict_name=model_saved_folder+os.sep+'%s.pkl'%my_rnn_model.name
    if os.path.exists(saved_dict_name):
        print('WARNING: File %s already exists'%saved_dict_name)
        sub_idx=1
        while os.path.exists(model_saved_folder+os.sep+'%s(%d).pkl'%(my_rnn_model.name,sub_idx)):
            sub_idx+=1
        saved_dict_name=model_saved_folder+os.sep+'%s(%d).pkl'%(my_rnn_model.name,sub_idx)
        print('Saved in file %s instead!'%saved_dict_name)
    pickle.dump(saved_dict, open(saved_dict_name,'w'))

