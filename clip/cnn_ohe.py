import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
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
			#	app = app.flatten()
				#print(app)
				if(positive):
					app = np.append(app, 1)
					ListPositive.append(app)
				else:
					app = np.append(app, 0)
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

X_train_positive = np.array(X_train_positive)
print('shape after func called:', X_train_positive.shape)

#X_train_positive = np.insert(X_train_positive, X_train_positive.shape[1], 1, axis = 1)

print(len(X_train_negative))
print(X_train_negative[0])
print(X_train_negative[0].shape)
X_train_negative = np.array(X_train_negative)
#X_train_negative = np.insert(X_train_negative, X_train_negative.shape[1], 0, axis = 1)
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
#X_test_positive = np.insert(X_test_positive, X_test_positive.shape[1], 1, axis = 1)

print(len(X_test_negative))
print(X_test_negative[0])
print(X_test_negative[0].shape)
X_test_negative = np.array(X_test_negative)
X_test_negative = np.array(X_test_negative)
print(X_test_negative.shape)
#X_test_negative = np.insert(X_test_negative, X_test_negative.shape[1], 0, axis = 1)

X_test = X_test_positive
X_test = np.concatenate((X_test, X_test_negative), axis = 0)
X_test = shuffle(X_test, random_state = 0)

X_input_test = X_test[:, 0 : X_test.shape[1] - 1]
X_output_test = X_test[:, X_test.shape[1] - 1 : X_test.shape[1]]

X_input_train = np.reshape(X_input_train, (X_input_train.shape[0], 4, 101))
X_input_test = np.reshape(X_input_test, (X_input_test.shape[0], 4, 101))
print('shape: ', X_input_train.shape)
print('shapeY: ', X_output_train.shape)

model = Sequential()
model.add(Conv1D(140, kernel_size=2, activation='relu', input_shape=(4, 101)))
model.add(Conv1D(70, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.1, momentum=0.1)
opt2 = RMSprop(learning_rate=0.01)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(RAdam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
#model.fit(X_input_train, X_output_train, validation_data=(X_input_test, X_output_test), epochs=100)
history = model.fit(X_input_train, X_output_train, validation_data=(X_input_test, X_output_test), epochs=100)

EI_Y_pred = model.predict(X_input_test, verbose = 1)
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