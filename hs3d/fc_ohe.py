import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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
					app = app.flatten()
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

#classificatore to recognize EI splice sites:
EI_X1 = np.array(EI_training_false_seq_matrix)
EI_Xfalse = np.array(IE_training_false_seq_matrix)
EI_X1 = np.concatenate((EI_X1, EI_Xfalse), axis = 0)
print(EI_X1.shape)
#EI_X1 = np.append(EI_X1, np.ones((EI_X1.shape[0]), dtype = int), axis = 1) 
EI_X1 = np.insert(EI_X1, EI_X1.shape[1], 1, axis = 1)

#concateniamo gli array
EI_X2 = np.array(EI_training_true_matrix)
EI_Xtrue = np.array(IE_training_true_matrix)
EI_X2 = np.concatenate((EI_X2, EI_Xtrue), axis=0)
EI_X2 = np.insert(EI_X2, EI_X2.shape[1], 0, axis = 1)

#EI_X3 = np.array(IE_training_true_matrix)
#EI_X3 = np.insert(EI_X3, EI_X3.shape[1], 0, axis = 1)

#EI_X4 = np.array(IE_training_false_seq_matrix)
#EI_X4 = np.insert(EI_X4, EI_X4.shape[1], 0, axis = 1)



#test per EI splice sites
EI_test_X1 = np.array(EI_test_false_seq_matrix)
EI_test_Xfalse = np.array(IE_test_false_seq_matrix)
EI_test_X1 = np.concatenate((EI_test_X1, EI_test_Xfalse), axis=0)

EI_test_X1 = np.insert(EI_test_X1, EI_test_X1.shape[1], 1, axis = 1)


EI_test_X2 = np.array(EI_test_true_matrix)
EI_test_Xtrue = np.array(IE_test_false_seq_matrix)
EI_test_X2 = np.concatenate((EI_test_X2, EI_test_Xtrue), axis=0)

EI_test_X2 = np.insert(EI_test_X2, EI_test_X2.shape[1], 0, axis = 1)
EI_app_training = EI_test_X2[0:2000,:]
EI_test_X2 = EI_test_X2[2000:EI_test_X2.shape[0],:]


EI_X = EI_X1
EI_X = np.concatenate((EI_X, EI_X2), axis = 0)
#EI_X = np.concatenate((EI_X, EI_X3), axis = 0)
#EI_X = np.concatenate((EI_X, EI_X4), axis = 0)
EI_X = np.concatenate((EI_X, EI_app_training), axis = 0)
EI_X = shuffle(EI_X, random_state = 0)
#sparpagliamo i valori positivi e negativi

EI_training_X = EI_X[:, 0 : EI_X.shape[1] - 1]
EI_training_Y = EI_X[:, EI_X.shape[1] - 1 : EI_X.shape[1]]

print('EI shape training: ')
print(EI_training_X.shape)
print(EI_training_Y.shape)

#EI_test_X3 = np.array(IE_test_true_matrix)
#EI_test_X3 = np.insert(EI_test_X3, EI_test_X3.shape[1], 0, axis = 1)

#EI_test_X4 = np.array(IE_test_false_seq_matrix)
#EI_test_X4 = np.insert(EI_test_X4, EI_test_X4.shape[1], 0, axis = 1)

EI_testX = EI_test_X1
EI_testX = np.concatenate((EI_testX, EI_test_X2), axis = 0)
#EI_testX = np.concatenate((EI_testX, EI_test_X3), axis = 0)
#EI_testX = np.concatenate((EI_testX, EI_test_X4), axis = 0)

#EI_testX = np.random.shuffle(EI_testX)
EI_testX = shuffle(EI_testX, random_state = 0)
EI_testX_ev = EI_testX[:, 0 : EI_testX.shape[1] - 1]
EI_testY_ev = EI_testX[:, EI_testX.shape[1] - 1 : EI_testX.shape[1]]

print('EI shape testing: ')
print(EI_testX_ev.shape)
print(EI_testY_ev.shape)

#proviamo a definire un EI_classifier keras
EI_classifier = Sequential()
print('EI_classifierlo creato')
EI_classifier.add(Dense(4, input_dim = (560), activation = 'relu'))
for i in range(0, 1):
	EI_classifier.add(Dense(80, activation = 'relu'))

EI_classifier.add(Dense(1, activation='sigmoid'))
print('layer di output creato')
opt = SGD(lr=0.1, momentum=0)
EI_classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#EI_classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
EI_classifier.summary()

print('funzione di perdita aggiunta')
print('training input: ')
print(EI_training_X)
print('training output: ')
print(EI_training_Y)

history = EI_classifier.fit(EI_training_X, EI_training_Y, validation_data=(EI_testX_ev, EI_testY_ev), epochs=10)
#history = EI_classifier.fit(EI_training_X, EI_training_Y, epochs=10)
print('addestramento')
_, accuracyTraining = EI_classifier.evaluate(EI_training_X, EI_training_Y)
_, accuracy = EI_classifier.evaluate(EI_testX_ev, EI_testY_ev)
print('Accuracy: %.2f' % (accuracy*100))
#vedere molto meglio le varie metriche disponibili
EI_Y_pred = EI_classifier.predict(EI_testX_ev, verbose = 1)
print(EI_Y_pred)
EI_Y_bool = np.argmax(EI_Y_pred, axis = 1)
print(EI_Y_bool)
print(classification_report(EI_testY_ev, EI_Y_bool))
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

positive, tp, fp, tn, fn = mtcs(EI_Y_pred, 0.5, EI_testY_ev)

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


Predicted_training = EI_classifier.predict(EI_training_X, verbose = 1)
positive2, tp2, fp2, tn2, fn2 = mtcs(Predicted_training, 0.8, EI_training_Y)

print('positive2: ' )
print(positive2)

print('true positive2: ' )
print(tp2)
print('false positive2: ')
print(fp2)
print('true negative2: ' )
print(tn2)
print('false negative2: ')
print(fn2)

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

