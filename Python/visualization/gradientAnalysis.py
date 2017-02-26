import os
import sys
import cPickle
import matplotlib.pyplot as plt

'''
>>> pkl_file, str, input *.pkl file
>>> grad_file, str, output *.grad file
'''
def pkl2grad(pkl_file,grad_file):
    gradient_info_list=cPickle.load(open(pkl_file,'rb'))
    if os.path.exists(grad_file):
        print 'warning: file %s already exists'%grad_file
        print 'information follows original texts.'
    fopen=open(grad_file,'a')

    for gradient_info in gradient_info_list:
        fopen.write('==================================\n')
        for key in gradient_info:
            if not key in ['gV','gU','gs','gW']:
                fopen.write('%s:%s\n'%(key,gradient_info[key]))

        layer_num=len(gradient_info['gU'])
        fopen.write('This is a recurrent neural network of %d hidden layer(s)\n'%layer_num)
        for idx in xrange(layer_num):
            fopen.write('Average abs value of matrix gU in layer %d: %.5f\n'%(idx+1, gradient_info['gU'][idx]))
            fopen.write('Average abs value of matrix gW in layer %d: %.5f\n'%(idx+1, gradient_info['gW'][idx]))
            fopen.write('Average abs value of matrix gs in layer %d: %.5f\n'%(idx+1, gradient_info['gs'][idx]))
        fopen.write('Average abs value of matrix gV: %.5f\n'%(gradient_info['gV']))
    fopen.flush()

'''
>>> grad_file: str, input *.grad file
>>> pkl_file: str, output *.pkl file
>>> layer_num: int, number of layers
'''
def grad2pkl(grad_file,pkl_file,layer_num):
    lines=open(grad_file,'r').readlines()
    lines=map(lambda x:x if x[-1]!='\n' else x[:-1],lines)

    ret=[]
    line_num_per_note=layer_num*3+5
    print_num=len(lines)/line_num_per_note
    for print_idx in xrange(print_num):
        line_begin=print_idx*line_num_per_note
        index=int(lines[line_begin+1].split(':')[1])+1
        epoch=int(lines[line_begin+2].split(':')[1])
        gU=[];gW=[];gs=[]
        for layer_idx in xrange(layer_num):
            gUi=float(lines[line_begin+4+layer_idx*3+0].split(' ')[-1])
            gWi=float(lines[line_begin+4+layer_idx*3+1].split(' ')[-1])
            gsi=float(lines[line_begin+4+layer_idx*3+2].split(' ')[-1])
            gU.append(gUi)
            gW.append(gWi)
            gs.append(gsi)
        gV=float(lines[line_begin+4+layer_num*3].split(' ')[-1])
        ret.append({'epoch':epoch, 'index':index, 'gU':gU, 'gW':gW, 'gs':gs, 'gV':gV})
    cPickle.dump(ret,open(pkl_file,'wb'))

'''
>>> produce pkl file in a batch manner
'''
def grad2pklBatch(folder,layer_num):
    for subdir,dirs,files in os.walk(folder):
        for file in files:
            if file.split('.')[-1] in ['grad',]:
                print 'detect file %s'%(subdir+os.sep+file)
                pkl_file='.'.join(file.split('.')[:-1])+'.pkl'
                if os.path.exists(subdir+os.sep+pkl_file):
                    print 'file %s already exists!'%(subdir+os.sep+pkl_file)
                    continue
                grad2pkl(subdir+os.sep+file,subdir+os.sep+pkl_file,layer_num)

'''
>>> plot gradient trend
'''
def plotGradient(pkl_file,output_file,batch_per_epoch=-1):
    gradient_info_list=cPickle.load(open(pkl_file,'rb'))
    layer_num=len(gradient_info_list[0]['gU'])
    x_axis_idx=range(len(gradient_info_list)) if batch_per_epoch<=0 else map(lambda x:batch_per_epoch*x['epoch']+x['index'],gradient_info_list)

    for layer_idx in xrange(layer_num):
        plt.plot(x_axis_idx,map(lambda x:x['gU'][layer_idx],gradient_info_list),label='gU%d'%(layer_idx+1))
        plt.plot(x_axis_idx,map(lambda x:x['gW'][layer_idx],gradient_info_list),label='gW%d'%(layer_idx+1))
        plt.plot(x_axis_idx,map(lambda x:x['gs'][layer_idx],gradient_info_list),label='gs%d'%(layer_idx+1))
    plt.plot(x_axis_idx,map(lambda x:x['gV'],gradient_info_list),label='gV')
    
    plt.legend()
    plt.title('gradient information loaded from %s'%pkl_file)
    plt.xlabel('Epoch/Batches')
    plt.ylabel('Gradient Avg Abs Value')

    if output_file==sys.stdout:
        plt.show()
    else:
        if not os.path.exists(os.dirname(output_file)):
            os.makedirs(os.dirname(output_file))
        fig=plt.gcf()
        fig.set_size_inches(18,13)
        fig.savefig(output_file)


