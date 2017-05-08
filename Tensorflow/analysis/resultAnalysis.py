import os
import sys
import traceback
import cPickle as pickle
import numpy as np

'''
>>> folder: str, the folder to scan
>>> recursive: bool, whether or not to scan the folder recursively
>>> fout: file in 'w' mode, output channel
'''
def print_results(folder,recursive,fout=sys.stdout):
    file_list=[]
    if recursive==False:
        for item in os.listdir(folder):
            if os.path.isfile(folder+os.sep+item) and item .split('.')[-1] in ['pkl',]:
                file_list.append(folder+os.sep+item)
    else:
        for subdir,dirs,files in os.walk(folder):
            for item in files:
                if item.split('.')[-1] in ['pkl',]:
                    file_list.append(subdir+os.sep+item)
    print('%d files detected in total'%len(file_list))

    results=[]
    for idx,item in enumerate(file_list):
        sys.stdout.write('Loading %d/%d ...\r'%(idx+1,len(file_list)))
        try:
            data=pickle.load(open(item,'rb'))
            train_err_list=data['train']
            test_err_list=data['test']
            optimizer_name=data['config']['rnn_network']['update_policy']['name']
            learning_rate=data['config']['rnn_network']['update_policy']['learning_rate']
            step_size=data['config']['rnn_network']['step_size']
            batch_size=data['config']['rnn_network']['batch_size']

            train_min_err=np.inf
            train_min_idx=-1
            test_min_err=np.inf
            test_min_idx=-1
            for key in train_err_list.keys():
                train_err_value=train_err_list[key]
                if train_err_value is np.nan:
                    train_min_err=np.nan
                    train_min_idx=np.nan
                    break
                elif train_err_value<train_min_err:
                    train_min_err=train_err_value
                    train_min_idx=key
            for key in test_err_list.keys():
                test_err_value=test_err_list[key]
                if test_err_value is np.nan:
                    test_min_err=np.nan
                    test_min_idx=np.nan
                    break
                elif test_err_value<test_min_err:
                    test_min_err=test_err_value
                    test_min_idx=key
            results.append({'train_min_err':train_min_err,'train_min_idx':train_min_idx,'test_min_err':test_min_err,'test_min_idx':test_min_idx,
                'file':item,'optimizer_name':optimizer_name,'batch_size':batch_size,'learning_rate':learning_rate,'step_size':step_size})
        except:
            print('Error while loading file: %s'%item)
            traceback.print_exc()
    results=sorted(results,lambda x,y: -1 if x['train_min_err']<y['train_min_err'] else 1)
    for result in results:
        fout.write('file: %s, train_min_err:%.3f, test_min_err:%.3f, %s %d\n'%(result['file'],result['train_min_err'],result['test_min_err'],
            result['optimizer_name'],result['batch_size']))

if __name__=='__main__':

    if len(sys.argv)!=2:
        print('Usage: python resultAnalysis.py <folder>')
        exit(0)

    print_results(folder=sys.argv[1],recursive=False,fout=sys.stdout)
