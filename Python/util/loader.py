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
