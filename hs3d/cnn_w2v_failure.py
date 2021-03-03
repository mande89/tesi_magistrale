import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.utils import shuffle
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from matplotlib import pyplot
from keras.models import model_from_json
from gensim.models import Word2Vec

vocabFinal = set()
vocabList = []

def create_kmers(seq, k, sw):
	kmers = []
	q,r = divmod(len(seq), sw)
	max_kmers = q - k + sw + 1
	if(sw == 1):
		max_kmers = max_kmers - 1
#	print('max_kmers: ')
#	print(max_kmers)
	for i in range(max_kmers):
		single_k_mer = seq[i*sw: i*sw + k]
		if(not (single_k_mer in vocabFinal)):
			vocabFinal.add(single_k_mer)
			vocabList.append([single_k_mer])
		kmers.append(single_k_mer)
	return(kmers)


def loadData(fileName):
	Lines = open(fileName)
	List = []
	Matrix = []
	count = 0
	Y = []
	for line in Lines:
		line = line[line.find(':') + 2: -1]
		if(len(line) == 140):
			count = count + 1
			List.append(line)
			kmer = create_kmers(line, 5, 1)
			Matrix.append(kmer)
	print('values: ')
	print(count)
	Matrix = np.array(Matrix)
	return List, Matrix

def calcXY(array):
	array = np.array(array)
#	X = shuffle(X, random_state = 0)
	X = array[:, 0:array.shape[1]-1,:]
	Y = array[:, array.shape[1] - 1 : array.shape[1], 0]
	return X,Y

EI_training_true_seq, EI_training_true_list = loadData("../datasets/EI_true.seq")
EI_training_false_seq, EI_training_false_list = loadData("../datasets/EI_false.seq")
#print(EI_training_true_list)

#IE_training_true_seq, IE_training_true_list = loadData("../datasets/IE_true.seq")
#IE_training_false_seq, IE_training_false_list = loadData("../datasets/IE_false.seq")

def processData(vocabulary, matrix, output):
	X = []
	for i in range(len(matrix)):
		sv = vocabulary[matrix[i]]
	#	print(sv.shape)
		sv = np.insert(sv, sv.shape[0], output, axis = 0)
		X.append(sv)
	return X

def processSingleValue(vocabulary, word, output):
	try:
		sv = vocabulary[word]
		sv = np.insert(sv, sv.shape[1], output, axis = 1)
		return sv
	except:
		pass

#EI_training_true_list = np.concatenate((EI_training_true_list, IE_training_true_list), axis = 0)
#EI_training_false_list = np.concatenate((EI_training_false_list, IE_training_false_list), axis = 0)

EI_training_true_list = shuffle(EI_training_true_list, random_state = 0)
EI_training_false_list = shuffle(EI_training_false_list, random_state = 0)

EI_train_true = EI_training_true_list[0:1000]
EI_test_true = EI_training_true_list[1000:]

EI_train_false = EI_training_false_list[0:10000]
EI_test_false = EI_training_false_list[10000:]

#EI = EI_test_false
#for j in range(len(EI_test_true)):
#	EI.append(EI_test_true[j])
#X_test = EI

vocab = ['AAA', 'AAC', 'AAT', 'AAG', 'ACA', 'ACC', 'ACT', 'ACG', 'ATA', 'ATC', 'ATT', 'ATG', 'AGA', 'AGC', 'AGT', 'AGG', 'CAA', 'CAC', 'CAT', 'CAG', 'CCA', 'CCC', 'CCT', 'CCG', 'CTA', 'CTC', 'CTT', 'CTG', 'CGA', 'CGC', 'CGT', 'CGG', 'TAA', 'TAC', 'TAT', 'TAG', 'TCA', 'TCC', 'TCT', 'TCG', 'TTA', 'TTC', 'TTT', 'TTG', 'TGA', 'TGC', 'TGT', 'TGG', 'GAA', 'GAC', 'GAT', 'GAG', 'GCA', 'GCC', 'GCT', 'GCG', 'GTA', 'GTC', 'GTT', 'GTG', 'GGA', 'GGC', 'GGT', 'GGG']


modelUnique = Word2Vec(vocabList, min_count=1, size=32, workers=3, window=3, sg=0)
mT = Word2Vec(vocabList, min_count=1, size=32, workers=3, window=3, sg=0)
mF = Word2Vec(vocabList, min_count=1, size=32, workers=3, window=3, sg=0)

#model = Word2Vec(EI_training_true_list, min_count=1, size=32, workers=3, window=3, sg=0)
#model2 = Word2Vec(EI_training_false_list, min_count=1, size=32, workers=3, window=3, sg=0)
#array = (model['GGAG', 'GAGT', 'AGTT', 'GTTT', 'TTTG', 'TTGC', 'TGCT', 'GCTG', 'CTGT', 'TGTG', 'GTGC', 'TGCT', 'GCTC', 'CTCA', 'TCAA', 'CAAC', 'AACT', 'ACTT', 'CTTC', 'TTCC', 'TCCT', 'CCTG', 'CTGA', 'TGAT', 'GATC', 'ATCT', 'TCTA', 'CTAC', 'TACA', 'ACAA', 'CAAC', 'AACC', 'ACCA', 'CCAG', 'CAGA', 'AGAC', 'GACA', 'ACAA', 'CAAA', 'AAAA', 'AAAG', 'AAGC', 'AGCC', 'GCCC', 'CCCA', 'CCAT', 'CATG', 'ATGC', 'TGCT', 'GCTT', 'CTTC', 'TTCT', 'TCTC', 'CTCC', 'TCCT', 'CCTA', 'CTAA', 'TAAA', 'AAAC', 'AACT', 'ACTC', 'CTCC', 'TCCG', 'CCGC', 'CGCC', 'GCCA', 'CCAT', 'CATG', 'ATGT', 'TGTA', 'GTAT', 'TATG', 'ATGA', 'TGAG', 'GAGC', 'AGCT', 'GCTG', 'CTGG', 'TGGG', 'GGGT', 'GGTA', 'GTAT', 'TATG', 'ATGG', 'TGGG', 'GGGA', 'GGAG', 'GAGT', 'AGTG', 'GTGG', 'TGGT', 'GGTG', 'GTGG', 'TGGC', 'GGCA', 'GCAA', 'CAAG', 'AAGG', 'AGGC', 'GGCT', 'GCTT', 'CTTT', 'TTTG', 'TTGG', 'TGGA', 'GGAG', 'GAGT', 'AGTG', 'GTGT', 'TGTA', 'GTAG', 'TAGA', 'AGAG', 'GAGA', 'AGAC', 'GACA', 'ACAT', 'CATG', 'ATGC', 'TGCT', 'GCTA', 'CTAG', 'TAGC', 'AGCA', 'GCAA', 'CAAG', 'AAGG', 'AGGG', 'GGGT', 'GGTA', 'GTAC', 'TACT', 'ACTG', 'CTGG', 'TGGG', 'GGGG', 'GGGT'])
#print(array.shape)
#array = np.insert(array, array.shape[0], 1, axis = 0)
#print(array)
#print(array.shape)

X_train_true = processData(mT, EI_train_true, 1)
X_train_false = processData(mF, EI_train_false, 0)

X_train = np.concatenate((X_train_true, X_train_false), axis = 0)

X_train, Y_train = calcXY(X_train)
#X_test, Y_test = calcXY(X_test)

print(X_train)
print(Y_train)

print(X_train.shape)
print(Y_train.shape)



model = Sequential()
model.add(Conv1D(15, kernel_size=20, activation='relu', input_shape=(136, 32)))
model.add(Conv1D(1, kernel_size=40, activation='relu'))
model.add(Flatten())
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.001, momentum=0.95)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, epochs=1000)

model_json = model.to_json()
with open("modelcnn_30p.json", 'w') as json_file:
	json_file.write(model_json)
model.save_weights("modelcnn_30p.h5")
print('model saved')

X_test_T = processData(modelUnique, EI_test_true, 1)
X_test_F = processData(modelUnique, EI_test_false, 0)

X_test = np.concatenate((np.array(X_test_T), np.array(X_test_F)), axis = 0)
X_test, Y_test = calcXY(X_test)

eval = model.evaluate(X_test, Y_test)
print(eval)

Y_pred = model.predict(X_test, verbose = 1)

positive = 0
tp = 0
fp = 0
tn = 0
fn = 0

#vedere meglio questa parte relativa alle probabilità che appartengano ad una classe invece che ad un'altra
def classify(X_test, output, threshold):
	positive = 0
	tp = 0
	tn = 0
	fn = 0
	fp = 0
	ytrumagg = 0
	X_test = np.array(X_test)
	print(X_test.shape[0])
	for i in range(X_test.shape[0]):
		svTrue = processSingleValue(mT, X_test[i], output[i])
		svFalse = processSingleValue(mF, X_test[i], output[i])
		if((svTrue is not None)and(svFalse is not None)):
			svTrue = np.array(svTrue)
			svTY = svTrue[0,svTrue.shape[1] - 1: svTrue.shape[1]]
			svTrue = svTrue[:,0:svTrue.shape[1] -1]
			
			svFalse = np.array(svFalse)
			svFY = svFalse[0, svFalse.shape[1] - 1 : svFalse.shape[1]]
			svFalse = svFalse[:, 0:svFalse.shape[1] -1]

		#	singleXTrue = svTrue.flatten()
		#	singleXFalse = svFalse.flatten()
			
		#	sXt = np.empty((1, 4416))
		#	sXt = np.empty((1, 4416))
		#	sXt[0] = singleXTrue
		#	sXf = np.empty((1, 4416))
		#	sXf = np.empty((1, 4416))
		#	sXf[0] = singleXFalse
			sXt = np.reshape(svTrue, (1, 136, 32))
			sXf = np.reshape(svFalse, (1, 136, 32))
			YpredictedTrue = model.predict(sXt, verbose = 1)
			YpredictedFalse = model.predict(sXf, verbose = 1)
			print('Y:')
			print(YpredictedTrue)
			print(YpredictedFalse)
			print(output[i])
			if(YpredictedTrue > YpredictedFalse):
				#allora il sistema lo classifica come appartenente alla classe addestrata con gli esempi positivi
				ytrumagg = ytrumagg + 1
				if(output[i] == 1):
					positive = positive + 1
					if(YpredictedTrue > threshold):
						#vuol dire che allora è stato classificato bene
						tp = tp + 1
					else:
						#allora è un falso negativo
						fn = fn + 1
				else:
					if(YpredictedTrue > threshold):
						#abbiamo un falso positivo
						fp = fp + 1
					else:
						#abbiamo un vero negativo
						tn = tn + 1
			else:
			#in questo caso il sistema lo classifica come appartenente alla classe addestrata con gli esempi negativi
				if(output[i] == 1):
					positive = positive + 1
					if(YpredictedFalse > threshold):
						#vuol dire che allora è stato classificato bene
						tp = tp + 1
					else:
						#allora è un falso negativo
						fn = fn + 1
				else:
					if(YpredictedFalse > threshold):
						#abbiamo un falso positivo
						fp = fp + 1
					else:
						#abbiamo un vero negativo
						tn = tn + 1
	return positive, tp, fp, tn, fn, ytrumagg

#pt, tpt, fpt, tnt, fnt, yt = classify(EI_test_true, np.ones(np.array(EI_test_true).shape[0]), 0.5)
#pf, tpf, fpf, tnf, fnf, yf = classify(EI_test_false, np.zeros(np.array(EI_test_false).shape[0]), 0.5)

def classifySingle(X_test, output, threshold):
	positive = 0
	tp = 0
	tn = 0
	fn = 0
	fp = 0
	ytrumagg = 0
	X_test = np.array(X_test)
	print(X_test.shape[0])
	for i in range(X_test.shape[0]):
		sv = processSingleValue(modelUnique, X_test[i], output[i])
		if(sv is not None):
			sv = np.array(sv)
			svY = sv[0,sv.shape[1] - 1: sv.shape[1]]
			svX = sv[:,0:sv.shape[1] -1]

		#	singleXTrue = svTrue.flatten()
		#	singleXFalse = svFalse.flatten()
			
		#	sXt = np.empty((1, 4416))
		#	sXt = np.empty((1, 4416))
		#	sXt[0] = singleXTrue
		#	sXf = np.empty((1, 4416))
		#	sXf = np.empty((1, 4416))
		#	sXf[0] = singleXFalse
			sXt= np.reshape(svX, (1, 136, 32))
			Ypredicted = model.predict(sXt, verbose = 1)
			print('Y:')
			print(Ypredicted)
			print(output[i])
			#allora il sistema lo classifica come appartenente alla classe addestrata con gli esempi positivi
			if(output[i] == 1):
				positive = positive + 1
				if(Ypredicted > threshold):
					#vuol dire che allora è stato classificato bene
					tp = tp + 1
				else:
					#allora è un falso negativo
					fn = fn + 1
			else:
				if(Ypredicted > threshold):
					#abbiamo un falso positivo
					fp = fp + 1
				else:
					#abbiamo un vero negativo
					tn = tn + 1
	return positive, tp, fp, tn, fn


X_class = np.array(EI_train_true)
X_class = np.concatenate((X_class, np.array(EI_train_false)), axis = 0)

#pttest, tpttest, fpttest, tnttest, fnttest = classifySingle(EI_test_true, np.ones(np.array(EI_test_true).shape[0]), 0.5)
#pftest, tpftest, fpftest, tnftest, fnftest = classifySingle(EI_test_false, np.zeros(np.array(EI_test_false).shape[0]), 0.5)

#ptrain, tptrain, fptrain, tntrain, fntrain, ytrain = classify(X_class, Y_train, 0.5)

def mtcs(pred, threshold, output):
	positive = 0
	for h in pred:
		if(h > threshold):
			positive = positive + 1
	i = 0
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	while i < pred.shape[0]:
		if (pred[i] > threshold):
			if(output[i] == 1):
				tp = tp + 1
			else:
				fp = fp + 1
		else:
			if(output[i] == 1):
				fn = fn + 1
			else:
				tn = tn + 1
		i = i +1
	return (positive, tp, fp, tn, fn)

pttest, tpttest, fpttest, tnttest, fnttest = mtcs(Y_pred, 0.5, Y_test)

print('positive positive class: ' )
print(pttest)

print('true positive: ' )
print(tpttest)
print('false positive: ')
print(fpttest)
print('true negative: ' )
print(tnttest)
print('false negative: ')
print(fnttest)

#print('positive positive class: ' )
#print(pftest)

#print('true positive: ' )
#print(tpftest)
#print('false positive: ')
#print(fpftest)
#print('true negative: ' )
#print(tnftest)
#print('false negative: ')
#print(fnftest)

#print('positive positive class: ' )
#print(pt)

#print('true positive: ' )
#print(tpt)
#print('false positive: ')
#print(fpt)
#print('true negative: ' )
#print(tnt)
#print('false negative: ')
#print(fnt)
#print('yTrue maggiore: ')
#print(yt)

#print('positive false class: ' )
#print(pf)

#print('true positive: ' )
#print(tpf)
#print('false positive: ')
#print(fpf)
#print('true negative: ' )
#print(tnf)
#print('false negative: ')
#print(fnf)
#print('yTrue maggiore: ')
#print(yf)


#print('positive positive class: ' )
#print(ptrain)

#print('true positive: ' )
#print(tptrain)
#print('false positive: ')
#print(fptrain)
#print('true negative: ' )
#print(tntrain)
#print('false negative: ')
#print(fntrain)
#print('yTrue maggiore: ')
#print(ytrain)

#X_true_EI = np.expand_dims(X_true_EI, axis=3)
#X_true_EI = np.insert(X_true_EI, 1, ones, axis = 3)
#print(X_true_EI.shape)