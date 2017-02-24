import os
import sys
import random
import matplotlib.pyplot as plt

color_list=['r','g','b','c','m','y','k']

def random_color():
    color_index=random.randint(0,256**3-1)
    color_str=str(hex(color_index))
    return color_str.replace('0x','#')

'''
>>> entries: list of dict, each element has key 'train', 'test', 'title'
>>> output_file: str, the file saved
'''
def plotLearningCurve(entries,output_file):
    color_id=0
    for entry in entries:
        train_values=map(float,entry['train'].split('|'))
        test_values=map(float,entry['test'].split('|')) if entry.has_key('test') else None
        color=color_list[color_id] if color_id<len(color_list) else random_color()
        color_id+=1

        handler1,=plt.plot(range(1,len(train_values)+1),train_values,color=color,linewidth=3,label=entry['title'])
        if test_values!=None:
            handler2,=plt.plot(range(1,len(test_values)+1),test_values,color=color,linewidth=1)

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if output_file==sys.stdout:
        plt.show()
    else:
        if not os.path.exists(os.dirname(output_file)):
            os.makedirs(os.dirname(output_file))
        fig=plt.gcf()
        fig.set_size_inches(18,13)
        fig.savefig(output_file)

