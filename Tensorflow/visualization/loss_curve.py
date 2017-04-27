import os
import sys
import random
import cPickle
import matplotlib.pyplot as plt

if __name__=='__main__':
    if len(sys.argv)<2:
        print('Usage: python loss_curve.py <pkl file ...>')
        exit(0)

def get_color(color_idx):
    base_color=['r','g','b','y','c','m','k']
    if color_idx<7:
        return base_color[color_idx]
    else:
        dex=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
        ret_color='#'
        for _ in xrange(6):
            token_idx=random.randint(0,15)
            ret_color+=dex[token_idx]
        return ret_color

handles=[]
for idx,pkl_file in enumerate(sys.argv[1:]):
    label=pkl_file.split(os.sep)[-1]
    info=cPickle.load(open(pkl_file,'rb'))
    train_err_ckpt=info['train']
    test_err_ckpt=info['test']
    train_ckpt_list=train_err_ckpt.keys()
    test_ckpt_list=test_err_ckpt.keys()
    if sys.version_info.major==2:
        train_ckpt_list=sorted(train_ckpt_list,lambda x,y:-1 if x<y else 1)
        test_ckpt_list=sorted(test_ckpt_list,lambda x,y:-1 if x<y else 1)
    else:
        train_ckpt_list=sorted(train_ckpt_list,key=lambda x:x,reverse=False)
        test_ckpt_list=sorted(test_ckpt_list,key=lambda x:x,reverse=False)
    train_err_list=[train_err_ckpt[pos] for pos in train_ckpt_list]
    test_err_list=[test_err_ckpt[pos] for pos in test_ckpt_list]
    color_this_file=get_color(color_idx=idx)
    handle,=plt.plot(train_ckpt_list,train_err_list,color=color_this_file,
        label=label,linestyle='-')
    _,=plt.plot(test_ckpt_list,test_err_list,color=color_this_file,
        label=label,linestyle='--')
    handles.append(handle)

plt.xlabel('batches')
plt.ylabel('loss')
plt.legend()
plt.show()
