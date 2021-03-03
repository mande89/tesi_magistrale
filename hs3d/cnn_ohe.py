import numpy as np
#import tensorboard
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.utils import shuffle
from keras.optimizers import SGD
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
			#	print(line)
				singleSeq = [[],[],[],[]]
				h = True
				for char in line:
					if (char != 'H'):
						x = singleNt(char)
				#	print('ok')
			#		print(char)
						singleSeq[0].append(x[0])
						singleSeq[1].append(x[1])
						singleSeq[2].append(x[2])
						singleSeq[3].append(x[3])
					else:
						h = False
				#	singleSeq = np.append(singleSeq, x, axis = 1)
				#	singleSeq[1].append(output)
				#Matrix = np.append(Matrix, singleSeq, axis = 0)
				if(h):
					app = np.array(singleSeq)
			#	print(app.shape)
			#	app = app.flatten()
					app = np.append(app, output)
			#	print(app.shape)
					Matrix.append(app)

	print('values: ')
	print(count)
	return List, Matrix

#loading our training data
EI_training_true_seq, EI_training_true_matrix = loadData("../datasets/training_EI_true.seq", 0)
EI_training_false_seq, EI_training_false_seq_matrix = loadData("../datasets/training_EI_false.seq", 1)
IE_training_true_seq, IE_training_true_matrix = loadData("../datasets/training_IE_true.seq", 0)
IE_training_false_seq, IE_training_false_seq_matrix = loadData("../datasets/training_IE_false.seq", 1)

#loading our test datasets
EI_test_true_seq, EI_test_true_matrix = loadData("../datasets/test_EI_true.seq", 0)
EI_test_false_seq, EI_test_false_seq_matrix = loadData("../datasets/test_EI_false.seq", 1)
IE_test_true_seq, IE_test_true_matrix = loadData("../datasets/test_IE_true.seq", 0)
IE_test_false_seq, IE_test_false_seq_matrix = loadData("../datasets/test_IE_false.seq", 1)

X_false_EI = np.array(EI_training_true_matrix)
X_false_IE = np.array(IE_training_true_matrix)
X_false_test_IE = np.array(IE_test_true_matrix)
X_false_EI = np.concatenate((X_false_EI, X_false_IE), axis = 0)
X_false_EI = np.concatenate((X_false_EI, X_false_test_IE), axis = 0)
X_false_EI = np.concatenate(( X_false_EI, EI_test_true_matrix), axis=0)
X_false_EI = shuffle(X_false_EI, random_state=0)
X_train_false = X_false_EI[0 : 2000, :]
X_test_false = X_false_EI[2000 : X_false_EI.shape[0], :]

print(np.array(EI_training_false_seq_matrix).shape)
EI_X = np.array(EI_training_false_seq_matrix)
IE_X = np.array(IE_training_false_seq_matrix)
EI_X = np.concatenate((EI_X, IE_X), axis=0)

EI_X = np.concatenate((EI_X, X_train_false), axis = 0)
EI_X = shuffle(EI_X, random_state = 0)

EI_training_X = EI_X[:, 0 : EI_X.shape[1] - 1]
EI_training_Y = EI_X[:, EI_X.shape[1] - 1 : EI_X.shape[1]]
print(EI_training_Y.shape)
print((np.reshape(EI_training_X, (EI_training_X.shape[0], 4, 140))).shape)
EI_training_X = np.reshape(EI_training_X, (EI_training_X.shape[0], 4, 140))


print(np.array(EI_test_false_seq_matrix).shape)
EI_testX = np.array(EI_test_false_seq_matrix)
IE_testX = np.array(IE_test_false_seq_matrix)
EI_testX = np.concatenate((EI_testX, IE_testX), axis=0)

EI_testX = np.concatenate((EI_testX, X_test_false), axis = 0)
EI_testX = shuffle(EI_testX, random_state = 0)

EI_X_test = EI_testX[:, 0:EI_testX.shape[1] - 1]
EI_Y_test = EI_testX[:, EI_testX.shape[1] - 1 : EI_testX.shape[1]]
EI_X_test = np.reshape(EI_X_test, (EI_X_test.shape[0], 4, 140))


model = Sequential()
model.add(Conv1D(140, kernel_size=2, activation='relu', input_shape=(4, 140)))
model.add(Conv1D(70, kernel_size=2, activation='relu'))
#model.add(Conv1D(35, kernel_size=2, activation='relu'))
#model.add(Conv1D(35, kernel_size=1, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.1, momentum=0.1)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(EI_training_X, EI_training_Y, validation_data=(EI_X_test, EI_Y_test), epochs=4)
#history = model.fit(EI_training_X, EI_training_Y, epochs=4)
#eval = model.evaluate(EI_X_test, EI_Y_test)
#print(eval)
EI_Y_pred = model.predict(EI_X_test, verbose = 1)
print(EI_Y_pred)
EI_Y_bool = np.argmax(EI_Y_pred, axis = 1)
print(EI_Y_bool)
print(classification_report(EI_Y_test, EI_Y_bool))

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

positive, tp, fp, tn, fn = mtcs(EI_Y_pred, 0.5, EI_Y_test)

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