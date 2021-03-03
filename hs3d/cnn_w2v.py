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
		kmers.append(single_k_mer)
	return(kmers)


def loadData(fileName, output):
	Lines = open(fileName)
	List = []
	Matrix = []
	count = 0
	Y = []
	for line in Lines:
		proximity = line[line.find(')') -1 : line.find(')')]
		line = line[line.find(':') + 2: -1]
		if((output != 0)or(proximity != 1)):
			if(len(line) == 140):
				count = count + 1
				List.append(line)
				kmer = create_kmers(line, 3, 1)
				Matrix.append(kmer)
	print('values: ')
	print(count)
	return List, Matrix


EI_training_true_seq, EI_training_true_list = loadData("../datasets/EI_true.seq", 1)
EI_training_false_seq, EI_training_false_list = loadData("../datasets/EI_false.seq", 0)

IE_training_true_seq, IE_training_true_list = loadData("../datasets/IE_true.seq", 1)
IE_training_false_seq, IE_training_false_list = loadData("../datasets/IE_false.seq", 0)
#print(EI_training_true_list)

def processData(vocabulary, matrix, output):
	X = []
	for i in range(len(matrix)):
		sv = vocabulary[matrix[i]]
	#	print(sv.shape)
		sv = np.insert(sv, sv.shape[0], output, axis = 0)
		X.append(sv)
	return X

EI = EI_training_false_list
for j in range(len(EI_training_true_list)):
	EI.append(EI_training_true_list[j])

mu = Word2Vec(EI, min_count=1, size=32, workers=3, window=3, sg=0)

#model = Word2Vec(EI_training_true_list, min_count=1, size=32, workers=3, window=3, sg=0)
#model2 = Word2Vec(EI_training_false_list, min_count=1, size=32, workers=3, window=3, sg=0)
#array = (model['GGAG', 'GAGT', 'AGTT', 'GTTT', 'TTTG', 'TTGC', 'TGCT', 'GCTG', 'CTGT', 'TGTG', 'GTGC', 'TGCT', 'GCTC', 'CTCA', 'TCAA', 'CAAC', 'AACT', 'ACTT', 'CTTC', 'TTCC', 'TCCT', 'CCTG', 'CTGA', 'TGAT', 'GATC', 'ATCT', 'TCTA', 'CTAC', 'TACA', 'ACAA', 'CAAC', 'AACC', 'ACCA', 'CCAG', 'CAGA', 'AGAC', 'GACA', 'ACAA', 'CAAA', 'AAAA', 'AAAG', 'AAGC', 'AGCC', 'GCCC', 'CCCA', 'CCAT', 'CATG', 'ATGC', 'TGCT', 'GCTT', 'CTTC', 'TTCT', 'TCTC', 'CTCC', 'TCCT', 'CCTA', 'CTAA', 'TAAA', 'AAAC', 'AACT', 'ACTC', 'CTCC', 'TCCG', 'CCGC', 'CGCC', 'GCCA', 'CCAT', 'CATG', 'ATGT', 'TGTA', 'GTAT', 'TATG', 'ATGA', 'TGAG', 'GAGC', 'AGCT', 'GCTG', 'CTGG', 'TGGG', 'GGGT', 'GGTA', 'GTAT', 'TATG', 'ATGG', 'TGGG', 'GGGA', 'GGAG', 'GAGT', 'AGTG', 'GTGG', 'TGGT', 'GGTG', 'GTGG', 'TGGC', 'GGCA', 'GCAA', 'CAAG', 'AAGG', 'AGGC', 'GGCT', 'GCTT', 'CTTT', 'TTTG', 'TTGG', 'TGGA', 'GGAG', 'GAGT', 'AGTG', 'GTGT', 'TGTA', 'GTAG', 'TAGA', 'AGAG', 'GAGA', 'AGAC', 'GACA', 'ACAT', 'CATG', 'ATGC', 'TGCT', 'GCTA', 'CTAG', 'TAGC', 'AGCA', 'GCAA', 'CAAG', 'AAGG', 'AGGG', 'GGGT', 'GGTA', 'GTAC', 'TACT', 'ACTG', 'CTGG', 'TGGG', 'GGGG', 'GGGT'])
#print(array.shape)
#array = np.insert(array, array.shape[0], 1, axis = 0)
#print(array)
#print(array.shape)

X_true_EI = processData(mu, EI_training_false_list, 1)
X_true_IE = processData(mu, IE_training_false_list, 1)
X_true_EI = np.array(X_true_EI)
X_true_IE = np.array(X_true_IE)
X_true_EI = np.concatenate((X_true_EI, X_true_IE), axis = 0)
print(X_true_EI.shape)

X_true_EI = shuffle(X_true_EI, random_state = 0)

X_true = X_true_EI[:,0 : X_true_EI.shape[1] - 1, :]
Y_true = X_true_EI[:,X_true_EI.shape[1] - 1: X_true_EI.shape[1], 0]
#Y_true = np.reshape(Y_true, (1,999))
print(Y_true[0])
print(X_true.shape)
print(Y_true.shape)
#print(Y_true)
#Y_true = np.squeeze(Y_true, axis = 0)
#print(Y_true.shape)
X_false_EI = processData(mu, EI_training_true_list, 0)
X_false_IE = processData(mu, IE_training_true_list, 0)
X_false_IE = np.array(X_false_IE)
X_false_EI = np.array(X_false_EI)
X_false_EI = np.concatenate((X_false_EI, X_false_IE), axis = 0)

X_false_EI = shuffle(X_false_EI, random_state=0)
X_false = X_false_EI[:, 0: X_false_EI.shape[1] - 1, :]
Y_false = X_false_EI[:, X_false_EI.shape[1] - 1: X_false_EI.shape[1], 0]

X_train_true = X_true_EI[0 : 2000, :, :]
X_test_true = X_true_EI[2000 : X_true_EI.shape[0], :, :]

X_train_false = X_false_EI[0 : 2000, :, :]
X_test_false = X_false_EI[2000 : X_false_EI.shape[0], :, :]

EI_X = X_train_true
EI_X = np.concatenate((EI_X, X_train_false), axis = 0)
EI_X = shuffle(EI_X, random_state=0)

EI_X_test = X_test_true
EI_X_test = np.concatenate((EI_X_test, X_test_false), axis = 0)
EI_X_test = shuffle(EI_X_test, random_state=0)

X_train = EI_X[:, 0: EI_X.shape[1] - 1, :]
Y_train = EI_X[:, EI_X.shape[1] - 1 : EI_X.shape[1], 0]

X_test = EI_X_test[:, 0:EI_X_test.shape[1] - 1, :]
Y_test = EI_X_test[:, EI_X_test.shape[1] - 1 : EI_X_test.shape[1], 0]

print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)

model = Sequential()
model.add(Conv1D(140, kernel_size=2, activation='relu', input_shape=(138, 32)))
model.add(Conv1D(70, kernel_size=2, activation='relu'))
#model.add(Conv1D(35, kernel_size=2, activation='relu'))
#model.add(Conv1D(35, kernel_size=2, activation='relu'))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.001, momentum=0.1)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=60)
#history = model.fit(X_train, Y_train, epochs=30)

model_json = model.to_json()
with open("modelcnn.json", 'w') as json_file:
	json_file.write(model_json)
model.save_weights("modelcnn.h5")
print('model saved')

#eval = model.evaluate(X_test, Y_test)
#print(eval)

EI_Y_pred = model.predict(X_test, verbose = 1)
print(EI_Y_pred)
EI_Y_bool = np.argmax(EI_Y_pred, axis = 1)
print(EI_Y_bool)
print(classification_report(Y_test, EI_Y_bool))
c = 0
for v in EI_Y_bool:
	if (v == 1):
		c = c + 1
print('number of 1: ')
print(c)

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

positive, tp, fp, tn, fn = mtcs(EI_Y_pred, 0.8, Y_test)

print('positive: ' )
print(positive)

print('true positive: ' )
print(tp)
print('false positive: ')
print(fp)
print('true negative: ' )
print(tn)
print('false negative: ')
print(fn)

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()

#plot dell'accuracy
pyplot.subplot(212)
pyplot.title('accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='val')
pyplot.legend()
pyplot.show()