import xml.etree.ElementTree as et

def cast(value,tag):
    if tag=='str':
        return value
    if tag=='int':
        return int(value)
    if tag=='float':
        return float(value)
    if tag=='bool':
        return True if value in ['true','True'] else False
    if tag=='list_str':
        return value.split(',')
    if tag=='list_int':
        return map(int,value.split(','))
    if tag=='list_float':
        return map(float,value.split(','))
    if tag=='list_bool':
        return map(lambda x: True if x in ['true','True'] else False,
                value.split(','))
    print 'unrecognized type: %s'%tag
    return value

def element2dict(element):
    ret={}
    for child in element:
        if child.getchildren()==[]:
            attr=child.attrib['name']
            category=child.attrib['type']
            value=cast(child.text,category)
            ret[attr]=value
        else:
            attr=child.tag
            ret[attr]=element2dict(child)
    return ret

def flatten(dictTree):
    ret={}
    for key in dictTree:
        if type(dictTree[key])==dict:
            subdict=flatten(dictTree[key])
            for subkey in subdict:
                if ret.has_key(subkey):
                    print 'Conflict of the key value %s when flattening.'%subkey
                else:
                    ret[subkey]=subdict[subkey]
        else:
            if ret.has_key(key):
                print 'Conflict of the key value %s when flattening.'%key
            ret[key]=dictTree[key]
    return ret

'''
>>> parse a xml file and return a dict
>>> file: str. xml file.
>>> flat: bool. use flat or hierarchical dictionary
'''
def parse(file,flat):
    root=et.parse(file).getroot()
    dictTree=element2dict(root)
    if flat:
        dictTree=flatten(dictTree)
    return dictTree