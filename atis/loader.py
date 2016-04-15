import cPickle
import csv
import sys
import numpy as np

def loadBinVec(fileName,vocab):
	'''
	>>>fileName: *.bin containing wordVec information
	>>>vocab: vocabulary dictionary
	'''
	wordVec={}
	with open(fileName,'rb') as fopen:
		header=fopen.readline()
		vocabSize,dimensions=map(int,header.split())
		binaryLen=np.dtype('float32').itemsize*dimensions
		for line in xrange(vocabSize):
			word=[]
			while True:
				ch=fopen.read(1)
				if ch==' ':
					word=''.join(word)
					break
				else:
					word.append(ch)
			if word in vocab:
				wordVec[word]=np.fromstring(fopen.read(binaryLen),dtype='float32')
			else:
				fopen.read(binaryLen)
	return wordVec

if __name__=='__main__':
	if len(sys.argv)!=2:
		print 'Usage: python loader.py <wordvec>'
		exit(0)
	vec_path=sys.argv[1]

	data=cPickle.load(open('atis.pkl'))

	train_word,train_tbl,train_lbl=data[0]
	test_word,test_tbl,test_lbl=data[1]
	word2idx,tbl2idx,lbl2idx=data[2]['words2idx'],data[2]['tables2idx'],data[2]['labels2idx']

	#create train/test indexes and labels
	train_word_file=csv.writer(open('train_word.csv','w'))
	train_label_file=csv.writer(open('train_label.csv','w'))
	test_word_file=csv.writer(open('test_word.csv','w'))
	test_label_file=csv.writer(open('test_label.csv','w'))

	train_word_file.writerows(train_word)
	train_label_file.writerows(train_lbl)
	test_word_file.writerows(test_word)
	test_label_file.writerows(test_lbl)

	#create index2word
	idx2word={}
	for key in word2idx:
		idx2word[word2idx[key]]=key

	#create the plain text
	with open('atis.txt','w') as fopen:
		for i in xrange(len(train_word)):
			for j in xrange(len(train_word[i])):
				fopen.write(idx2word[train_word[i][j]]+' ')
			fopen.write('\n')
		for i in xrange(len(test_word)):
			for j in xrange(len(test_word[i])):
				fopen.write(idx2word[test_word[i][j]]+' ')
			fopen.write('\n')

	#create word embedding information
	wordVec=loadBinVec(vec_path,word2idx)
	with open('dict.csv','w') as fopen:
		for key in wordVec:
			fopen.write(str(word2idx[key])+":")
			for i in xrange(len(wordVec[key])):
				if i!=len(wordVec[key])-1:
					fopen.write(str(wordVec[key][i])+',')
				else:
					fopen.write(str(wordVec[key][i])+'\n')

