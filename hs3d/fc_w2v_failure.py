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


vocab = ['AAA', 'AAC', 'AAT', 'AAG', 'ACA', 'ACC', 'ACT', 'ACG', 'ATA', 'ATC', 'ATT', 'ATG', 'AGA', 'AGC', 'AGT', 'AGG', 'CAA', 'CAC', 'CAT', 'CAG', 'CCA', 'CCC', 'CCT', 'CCG', 'CTA', 'CTC', 'CTT', 'CTG', 'CGA', 'CGC', 'CGT', 'CGG', 'TAA', 'TAC', 'TAT', 'TAG', 'TCA', 'TCC', 'TCT', 'TCG', 'TTA', 'TTC', 'TTT', 'TTG', 'TGA', 'TGC', 'TGT', 'TGG', 'GAA', 'GAC', 'GAT', 'GAG', 'GCA', 'GCC', 'GCT', 'GCG', 'GTA', 'GTC', 'GTT', 'GTG', 'GGA', 'GGC', 'GGT', 'GGG']

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
			kmer = create_kmers(line, 3, 1)
			Matrix.append(kmer)
	print('values: ')
	print(count)
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


EI_train_true = EI_training_true_list[0:1000]
EI_test_true = EI_training_true_list[1000:]

EI_train_false = EI_training_false_list[0:10000]
EI_test_false = EI_training_false_list[10000:]

EI = EI_test_false
for j in range(len(EI_test_true)):
	EI.append(EI_test_true[j])
X_test = EI
print(vocabList)
modelUnique = Word2Vec(vocabList, min_count=1, size=4, workers=3, window=3, sg=0)
mT = Word2Vec(vocabList, min_count=1, size=4, workers=3, window=3, sg=0)
mF = Word2Vec(vocabList, min_count=1, size=4, workers=3, window=3, sg=0)

#modelUnique = Word2Vec(EI, min_count=1, size=32, workers=3, window=3, sg=0)
#mT = Word2Vec(EI_train_true, min_count=1, size=32, workers=3, window=3, sg=0)
#mF = Word2Vec(EI_train_false, min_count=1, size=32, workers=3, window=3, sg=0)

X_train_true = processData(mT, EI_train_true, 1)
X_train_false = processData(mF, EI_train_false, 0)

X_train = np.concatenate((X_train_true, X_train_false), axis = 0)

X_train, Y_train = calcXY(X_train)
#X_test, Y_test = calcXY(X_test)

print(X_train)
print(Y_train)

print(X_train.shape)
print(Y_train.shape)

X_train_flattened = []
for i in range (X_train.shape[0]):
	X_train_flattened.append(X_train[i].flatten())
X_train_flattened = np.array(X_train_flattened)
print(X_train_flattened.shape)



#print(X_test.shape)
#print(Y_test.shape)

#X_test_flattened = []
#for i in range (X_test.shape[0]):
#	X_test_flattened.append(X_test[i].flatten())
#X_test_flattened = np.array(X_test_flattened)
#print(X_test_flattened.shape)


model = Sequential()
#model.add(Dense(32, input_dim = (4416), activation = 'relu'))
model.add(Dense(4, input_dim = (552), activation = 'relu'))
for i in range(1, 20):
	model.add(Dense(138, activation = 'relu'))

model.add(Dense(1, activation='sigmoid'))
print('layer di output creato')

opt = SGD(lr=0.001, momentum=0.95)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train_flattened, Y_train, epochs=60)
model_json = model.to_json()
with open("nn_fitted_w2v.json", 'w') as json_file:
	json_file.write(model_json)
model.save_weights("nn_fitted_w2v.h5")
print('model saved')

#eval = model.evaluate(X_test, Y_test)
#print(eval)

#EI_Y_pred = model.predict(X_test_flattened, verbose = 1)
#print(EI_Y_pred)


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

			singleXTrue = svTrue.flatten()
			singleXFalse = svFalse.flatten()
			
		#	sXt = np.empty((1, 4416))
			sXt = np.empty((1, 552))
			sXt[0] = singleXTrue
		#	sXf = np.empty((1, 4416))
			sXf = np.empty((1, 552))
			sXf[0] = singleXFalse
			#da una parte è giusto, perchè un esempio falso è giustissimo che abbia una probabilità prossima allo 0 che sia uguale ad 1 però comunque non funziona
			#è la probabilità che una sequenza modellata con tali valori siano  
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

pt, tpt, fpt, tnt, fnt, yt = classify(EI_test_true, np.ones(np.array(EI_test_true).shape[0]), 0.5)
pf, tpf, fpf, tnf, fnf, yf = classify(EI_test_false, np.zeros(np.array(EI_test_false).shape[0]), 0.5)

X_class = np.array(EI_train_true)
X_class = np.concatenate((X_class, np.array(EI_train_false)), axis = 0)

ptrain, tptrain, fptrain, tntrain, fntrain, ytrain = classify(X_class, Y_train, 0.5)
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

#positive, tp, fp, tn, fn = mtcs(EI_Y_pred, 0.5, Y_test)

print('positive positive class: ' )
print(pt)

print('true positive: ' )
print(tpt)
print('false positive: ')
print(fpt)
print('true negative: ' )
print(tnt)
print('false negative: ')
print(fnt)
print('yTrue maggiore: ')
print(yt)

print('positive false class: ' )
print(pf)

print('true positive: ' )
print(tpf)
print('false positive: ')
print(fpf)
print('true negative: ' )
print(tnf)
print('false negative: ')
print(fnf)
print('yTrue maggiore: ')
print(yf)


print('positive positive class: ' )
print(ptrain)

print('true positive: ' )
print(tptrain)
print('false positive: ')
print(fptrain)
print('true negative: ' )
print(tntrain)
print('false negative: ')
print(fntrain)
print('yTrue maggiore: ')
print(ytrain)


#X_true_EI = np.expand_dims(X_true_EI, axis=3)
#X_true_EI = np.insert(X_true_EI, 1, ones, axis = 3)
#print(X_true_EI.shape)

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.legend()

#plot dell'accuracy
pyplot.subplot(212)
pyplot.title('accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.legend()
pyplot.show()
