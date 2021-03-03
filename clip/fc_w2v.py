import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.utils import shuffle
from keras_radam import RAdam
from keras.optimizers import RMSprop
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

def loadData(fileName):
	Lines = open(fileName)
	List = []
	Matrix = []
	ListNegative = []
	ListPositive = []
	count = 0
	Y = []
	i = 0
	seq = ""
	out = 0
	for line in Lines:
		if(i == 3):
			i = 0
		if(i == 1):
			seq = out + line[: -1]
		if(i == 2):
			seq = seq + line[:-1]
			List.append(seq)
		if(i == 0):
			out = line[line.find(':') + 1: line.find(':') + 2]
			Matrix.append(out)
			#print(out)
		i = i+1
#	print(List)
	print(len(List[0]))
	print(Matrix.count('0'))
	print(Matrix.count('1'))
	i = 0
	for single_seq in List:
		single = [[],[],[],[]]
		positive = True
		h = True
		for char in single_seq:
			if((char != 'H')and(char != 'N')):
				if(char == '0'):
					positive = False
			else:
				h = False
		if((len(single_seq) == 102))and(h):
			app = create_kmers(single_seq, 3, 1)	
			
		#	app = np.array(kmer)
		#	print('shape single seq: ', app.shape)
		#	app = app.flatten()
			#print(app)
			if(positive):
				ListPositive.append(app)
			else:
				ListNegative.append(app)
	return ListPositive, ListNegative
		
X_train_positive, X_train_negative = loadData("../newDatasets/10_PARCLIP_ELAVL1A_hg19/seq_train.fa")

#print(X_train_positive)
X_test_positive, X_test_negative = loadData("../newDatasets/10_PARCLIP_ELAVL1A_hg19/seq_test.fa")

#X_train_positive2, X_train_negative2 = loadData("../newDatasets/11_CLIPSEQ_ELAVL1_hg19/seq_train.fa")
#X_test_positive2, X_test_negative2 = loadData("../newDatasets/11_CLIPSEQ_ELAVL1_hg19/seq_test.fa")
#X_train_positive = X_train_positive + X_train_positive2
#X_train_negative = X_train_negative + X_train_negative2

#X_test_positive = X_test_positive + X_test_positive2
#X_test_negative = X_test_negative + X_test_negative2

print(len(X_train_positive))
EI = X_train_negative
for j in range(len(X_train_positive)):
	EI.append(X_train_positive[j])

def processData(vocabulary, matrix, output):
	X = []
	for i in range(len(matrix)):
		found = True
		try:
			sv = vocabulary[matrix[i]]
	#	print(sv.shape)
			
		except:
			found = false
			print('kmer non presente')
		if(found):
			sv = np.insert(sv, sv.shape[0], output, axis = 0)
			X.append(sv)
	return X
print('len EI: ', len(EI))
#print(EI)
modelUnique = Word2Vec(EI, min_count=1, size=32, workers=3, window=3, sg=0)
print('model created')


X_train_positive = processData(modelUnique, X_train_positive, 1)
X_train_positive = np.array(X_train_positive)
print('shape after func called:', X_train_positive.shape)

#X_train_positive = np.insert(X_train_positive, X_train_positive.shape[1], 1, axis = 1)

X_train_negative = processData(modelUnique, X_train_negative, 0)
X_train_negative = np.array(X_train_negative)
#X_train_negative = np.insert(X_train_negative, X_train_negative.shape[1], 0, axis = 1)
#print(X_train_negative)
print('shape train negative: ', X_train_negative.shape)

X_train = X_train_positive
X_train = np.concatenate((X_train, X_train_negative), axis = 0)
X_train = shuffle(X_train, random_state = 0)
X_input_train = X_train[:, 0 : X_train.shape[1] - 1]
X_output_train = X_train[:, X_train.shape[1] - 1 : X_train.shape[1], 0]

X_input_train_flattened = []
for i in range (X_input_train.shape[0]):
	X_input_train_flattened.append(X_input_train[i].flatten())
X_input_train_flattened = np.array(X_input_train_flattened) 

X_test_positive = processData(modelUnique, X_test_positive, 1)
X_test_positive = np.array(X_test_positive)
print(X_test_positive.shape)
#X_test_positive = np.insert(X_test_positive, X_test_positive.shape[1], 1, axis = 1)

X_test_negative = processData(modelUnique, X_test_negative, 0)
X_test_negative = np.array(X_test_negative)
print(X_test_negative.shape)
#X_test_negative = np.insert(X_test_negative, X_test_negative.shape[1], 0, axis = 1)

X_test = X_test_positive
X_test = np.concatenate((X_test, X_test_negative), axis = 0)
X_test = shuffle(X_test, random_state = 0)

X_input_test = X_test[:, 0 : X_test.shape[1] - 1]
X_output_test = X_test[:, X_test.shape[1] - 1 : X_test.shape[1], 0]

X_input_test_flattened = []
for i in range (X_input_test.shape[0]):
	X_input_test_flattened.append(X_input_test[i].flatten())
X_input_test_flattened = np.array(X_input_test_flattened)

EI_classifier = Sequential()
print('EI_classifierlo creato')
EI_classifier.add(Dense(4, input_dim = (3200), activation = 'sigmoid'))
for i in range(0, 1):
	EI_classifier.add(Dense(138, activation = 'sigmoid'))

EI_classifier.add(Dense(1, activation='sigmoid'))
print('layer di output creato')
opt = SGD(lr=0.01, momentum=0.01)
opt2 = RMSprop(learning_rate=0.01)

EI_classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#EI_classifier.compile(RAdam(), loss='binary_crossentropy', metrics=['accuracy'])
EI_classifier.summary()

print('XTrain shape: ', X_input_train_flattened.shape)
print('YTrain shape: ', X_output_train.shape)
print('XTest shape: ', X_input_test_flattened.shape)
print('YTest shape: ', X_output_test.shape)

history = EI_classifier.fit(X_input_train_flattened, X_output_train, validation_data = (X_input_test_flattened, X_output_test), epochs= 100)
#history = EI_classifier.fit(X_input_train_flattened, X_output_train, epochs= 100)
EI_Y_pred = EI_classifier.predict(X_input_test_flattened, verbose = 1)
print(EI_Y_pred)
EI_Y_bool = np.argmax(EI_Y_pred, axis = 1)
print(EI_Y_bool)
print(classification_report(X_output_test, EI_Y_bool))
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

positive, tp, fp, tn, fn = mtcs(EI_Y_pred, 0.5, X_output_test)

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

#plot della funzione di costo
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