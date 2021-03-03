import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras_radam import RAdam
from sklearn.metrics import classification_report
from matplotlib import pyplot

def singleNt(arg):
	switcher = {
		'A': [1,0,0,0], 
		'C': [0,1,0,0],
		'G': [0,0,1,0],
		'T': [0,0,0,1],
		'H': [0,0,0,0]
	}
	return switcher.get(arg)

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
		h = True
		positive = True
		if(len(single_seq) == 102):			
			for char in single_seq:
				if((char != 'H')and(char != 'N')):
					if(char == '0'):
						positive = False
					else:
						if(char == '1'):
							positive = True
						else:
						#	print(char)
							x = singleNt(char)
							#print(x)
							i = i + 1
						#	print(i)
							single[0].append(x[0])
							single[1].append(x[1])
							single[2].append(x[2])
							single[3].append(x[3])
				else:
					h = False
			if(h):
				app = np.array(single)
			#	print('shape single seq: ', app.shape)
				app = app.flatten()
				#print(app)
				if(positive):
					ListPositive.append(app)
				else:
					ListNegative.append(app)
	return ListPositive, ListNegative
		
X_train_positive, X_train_negative = loadData("../newDatasets/10_PARCLIP_ELAVL1A_hg19/seq_train.fa")
#X_train_positive2, X_train_negative2 = loadData("../newDatasets/11_CLIPSEQ_ELAVL1_hg19/seq_train.fa")

#print(X_train_positive)
X_test_positive, X_test_negative = loadData("../newDatasets/10_PARCLIP_ELAVL1A_hg19/seq_test.fa")
#X_test_positive2, X_test_negative2 = loadData("../newDatasets/11_CLIPSEQ_ELAVL1_hg19/seq_test.fa")

#X_train_positive = X_train_positive + X_train_positive2
#X_train_negative = X_train_negative + X_train_negative2

#X_test_positive = X_test_positive + X_test_positive2
#X_test_negative = X_test_negative + X_test_negative2

X_train_positive = np.array(X_train_positive)
print('shape after func called:', X_train_positive.shape)

X_train_positive = np.insert(X_train_positive, X_train_positive.shape[1], 1, axis = 1)

print(len(X_train_negative))
print(X_train_negative[0])
print(X_train_negative[0].shape)
X_train_negative = np.array(X_train_negative)
X_train_negative = np.insert(X_train_negative, X_train_negative.shape[1], 0, axis = 1)
#print(X_train_negative)
print('shape train negative: ', X_train_negative.shape)

X_train = X_train_positive
X_train = np.concatenate((X_train, X_train_negative), axis = 0)
X_train = shuffle(X_train, random_state = 0)

X_input_train = X_train[:, 0 : X_train.shape[1] - 1]
X_output_train = X_train[:, X_train.shape[1] - 1 : X_train.shape[1]]

#print(X_test_positive)
X_test_positive = np.array(X_test_positive)
print(X_test_positive.shape)
X_test_positive = np.insert(X_test_positive, X_test_positive.shape[1], 1, axis = 1)

print(len(X_test_negative))
print(X_test_negative[0])
print(X_test_negative[0].shape)
X_test_negative = np.array(X_test_negative)
X_test_negative = np.array(X_test_negative)
print(X_test_negative.shape)
X_test_negative = np.insert(X_test_negative, X_test_negative.shape[1], 0, axis = 1)

X_test = X_test_positive
X_test = np.concatenate((X_test, X_test_negative), axis = 0)
X_test = shuffle(X_test, random_state = 0)

X_input_test = X_test[:, 0 : X_test.shape[1] - 1]
X_output_test = X_test[:, X_test.shape[1] - 1 : X_test.shape[1]]

EI_classifier = Sequential()
print('EI_classifierlo creato')
EI_classifier.add(Dense(4, input_dim = (404), activation = 'relu'))
for i in range(0, 1):
	EI_classifier.add(Dense(80, activation = 'relu'))

EI_classifier.add(Dense(1, activation='sigmoid'))
print('layer di output creato')
opt = SGD(lr=0.01, momentum=0.1)
opt2 = RMSprop(learning_rate=0.01)
EI_classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#EI_classifier.compile(RAdam(), loss='binary_crossentropy', metrics=['accuracy'])
EI_classifier.summary()

history = EI_classifier.fit(X_input_train, X_output_train, validation_data=(X_input_test, X_output_test), epochs= 100)
EI_Y_pred = EI_classifier.predict(X_input_test, verbose = 1)
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