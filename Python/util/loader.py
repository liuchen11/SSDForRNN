def loadData(file_name):
    ret=[]
    with open(file_name,'r') as fopen:
        for line in fopen:
            sentence=line.split(',')
            ret.append(map(int,sentence))
    return ret

def loadDict(file_name):
    ret={}
    with open(file_name,'r') as fopen:
        for line in fopen:
            parts=line.split(':')
            index=int(parts[0])
            vector=parts[1].split(',')
            ret[index]=map(float,vector)
    return ret

def loadCudaFile(file_name):
    return ''.join(open(file_name,'r').readlines())

def loadCudaFunc(file_name,func_list=None,global_mask=None):
    lines=open(file_name,'r').readlines()
    functions=[]
    for idx,line in enumerate(lines):
        content=line.replace(' ','').replace('\n','')
        if content[:11]=='//function:':
            functions.append([idx,content[11:]])
    function_dict={}
    for idx,item in enumerate(functions):
        startline=item[0]
        endline=functions[idx+1][0] if idx!=len(functions)-1 else len(lines)
        func_name=item[1]
        function_dict[func_name]=''.join(lines[startline:endline])
    header=''.join(lines[:functions[0][0]])
    if func_list==None:
        function_dict['__header__']=header
        return function_dict
    elif global_mask==None:
        ret_str=''
        for func in func_list:
            try:
                ret_str+=function_dict[func]
            except:
                print 'No function called %s in file %s'%(func,file_name)
        return header+ret_str
    else:
        ret_str=''
        assert(len(func_list)==len(global_mask))
        for func,mask in zip(func_list,global_mask):
            try:
                ret_str+=function_dict[func].replace('__global__','') if mask==False else function_dict[func]
            except:
                print 'No function called %s in file %s'%(func,file_name)
        return header+ret_str