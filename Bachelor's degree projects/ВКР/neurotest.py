from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2

from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

import shutil
import os, sys
from os import listdir
from os.path import isfile, join
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import collections
from collections import defaultdict


shape = ()

def make_path(p):
	return sys.path[0] + '/' + p

def make_model_path(args, p):
	return make_path('models/' + args.network + '/' + p)

def make_dataset_path(args, p):
	return make_path(args.dataset + '/' + p)


# случайная модификация изображений 
# (поворот, сдвиг, растяжение, зеркалироваание,..)
def setup_generator(train_path, test_path, batch_size, dims):
	train_datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		rescale=1. / 255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
		train_path,
		target_size=dims,
		batch_size=batch_size)

	validation_generator = test_datagen.flow_from_directory(
		test_path,
		target_size=dims,
		batch_size=batch_size)

	return train_generator, validation_generator


def load_image(img_path, dims, rescale=1. / 255):
	img = load_img(img_path, target_size=dims)
	x = img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x *= rescale
	return x

def load_strings(file_path):
	with open(file_path) as f:
		strings = f.read().splitlines()
	return strings


def create_model(m_type, num_classes, dropout, shape):
	modelclass = globals()[m_type]
	base_model = modelclass(
			weights='imagenet',
			include_top=False,
			input_tensor=Input(
			shape=shape))
	
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(dropout)(x)
	predictions = Dense(num_classes, activation='softmax')(x)

	model_final = Model(inputs=base_model.input, outputs=predictions)

	return model_final


def train_model(model_final, train_generator, validation_generator, callbacks, args):
	model_final.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	model_final.fit_generator(train_generator, 
				validation_data=validation_generator,
				epochs=args.epochs, callbacks=callbacks,
				steps_per_epoch=train_generator.samples//args.batch_size,
				validation_steps=validation_generator.samples//args.batch_size)


def load_model_by_name(args, network, num_classes, shape):
	model_final = create_model(network, num_classes, 0, shape)
	model_final.load_weights(make_path('models/' + network + '/' + args.model_path))
	return model_final

def load_model(args, num_classes, shape):
	model_final = create_model(args.network, num_classes, 0, shape)
	model_final.load_weights(make_model_path(args, args.model_path))
	return model_final


def plot_accuracy_loss(args):	   
	log = pd.read_csv(make_path('models/'+args.network+'/history_log.csv'))
	
	fig, ax = plt.subplots(1, 2, figsize=(18, 8))
	
	# accuracy
	ax[0].plot(log['accuracy'], 'r')
	ax[0].plot(log['val_accuracy'], 'b')
	ax[0].set_title(args.network+' | Зависимость точности (accuracy) от эпохи')
	ax[0].set(xlabel='epoch', ylabel='accuracy')
	ax[0].legend(['train', 'val'], loc='upper left')

	# loss
	ax[1].plot(log['loss'], 'r')
	ax[1].plot(log['val_loss'], 'b')
	ax[1].set_title(args.network+'. Зависимость потерь (loss) от эпохи')
	ax[1].set(xlabel='epoch', ylabel='loss')
	ax[1].legend(['train', 'val'], loc='upper left')

	# сохранение графика в файл
	plt.savefig(make_path('models/'+args.network+'/'+args.network+'_accuracy_loss.png'))

	# демонстрация графика
	plt.show()


def plots(args):
	networks = ['ResNet50', 'InceptionV3', 'InceptionResNetV2']
	colors = ['r', 'g', 'b']
	c = 0

	fig, ax = plt.subplots(2, 2, figsize=(18, 12))
	
	for n in networks:
		log = pd.read_csv(make_path('models/'+ n +'/history_log.csv'))
		
		ax[0,0].plot(log['accuracy'], colors[c])
		ax[1,0].plot(log['val_accuracy'], colors[c])
		ax[0,1].plot(log['loss'], colors[c])
		ax[1,1].plot(log['val_loss'], colors[c])
		
		c += 1

	ax[0,0].legend(networks, loc='upper left')
	ax[1,0].legend(networks, loc='upper left')
	ax[0,1].legend(networks, loc='upper right')
	ax[1,1].legend(networks, loc='upper right')

	fig.legend(networks, loc='upper left')

	# accuracy
	ax[0,0].set_title('Зависимость точности (accuracy) от эпохи')
	ax[0,0].set(xlabel='epoch', ylabel='accuracy')
	
	# val_accuracy
	ax[1,0].set_title('Зависимость точности (val_accuracy) от эпохи')
	ax[1,0].set(xlabel='epoch', ylabel='val_accuracy')
	
	# loss
	ax[0,1].set_title('Зависимость потерь (loss) от эпохи')
	ax[0,1].set(xlabel='epoch', ylabel='loss')
	
	# val_loss
	ax[1,1].set_title('Зависимость потерь (val_loss) от эпохи')
	ax[1,1].set(xlabel='epoch', ylabel='val_loss')
	
	# сохранение графика в файл
	plt.savefig(make_path('models/models_accuracy_loss.png'))

	# демонстрация графика
	plt.show()


def train(args):
	X_train, X_test = setup_generator(make_dataset_path(args, 'train'), make_dataset_path(args, 'test'), args.batch_size, shape[:2])
	print(X_train)
	callbacks = []
	callbacks.append(ModelCheckpoint(filepath=make_model_path(args, args.dataset + '-weights.epoch-{epoch:02d}-val_loss-{val_loss:.4f}-val_accuracy-{val_accuracy:.4f}.hdf5'), verbose=1, save_best_only=True))
	callbacks.append(CSVLogger(make_model_path(args, 'history_log.csv')))

	model_final = create_model(args.network, X_train.num_classes, args.dropout, shape)
	train_model(model_final, X_train, X_test, callbacks, args)


def testimage(args):
	classes = load_strings(make_dataset_path(args, 'meta/classes.txt'))
	trained_model = load_model(args, len(classes), shape)
	image = load_image(args.image_path, shape[:2])
	preds = trained_model.predict(image)
	print("класс изображения: ", classes[np.argmax(preds)])
	print("с вероятностью: ", np.max(preds)*100, "%")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def fulltest0(args):
	classes = load_strings(make_dataset_path(args, 'meta/classes.txt'))
	trained_model = load_model(args, len(classes), shape)

	test_images = load_strings(make_dataset_path(args, 'meta/test.txt'))
		
	n_true, n_false = 0, 0
	corrects = collections.defaultdict(int)
	incorrects = collections.defaultdict(int)
	for i in test_images:
		(iclass, iname) = i.split('/')
		image_path = make_dataset_path(args, 'test/'+i+'.jpg')
		image = load_image(image_path, shape[:2])
		preds = trained_model.predict(image)
		
		if classes[np.argmax(preds)] == iclass:
			# print(i, iclass, "+")
			n_true += 1
			corrects[iclass] += 1
		else:
			# print(i, classes[np.argmax(preds)], iclass, "-")
			n_false += 1
			incorrects[iclass] += 1

	print('\nПравильно распознаны: ', n_true)
	print(n_true/(n_true+n_false)*100,' %')
	print('\nОшибочно распознаны: ', n_false)
	print(n_false/(n_true+n_false)*100,' %')

	# print(corrects)
	# print('\n',incorrects)

	print('Точность распоздавания каждого класса:')
	class_accuracies = {}
	for ix in range(101):
		class_accuracies[ix] = corrects[classes[ix]]/250
		print(classes[ix], class_accuracies[ix])

	fig = plt.hist(list(class_accuracies.values()), bins=20, color='r')
	# plt.title('Accuracy by Class histogram')
	plt.title('Гистограмма точности распознавания классов')

	plt.savefig('histogram.png')
	plt.show()

	# sorted_class_accuracies = sorted(class_accuracies.items(), key=lambda x: -x[1])
	# print( [(ix_to_class[c[0]], c[1]) for c in sorted_class_accuracies] )



def fulltest(args):
	classes = load_strings(make_dataset_path(args, 'meta/classes.txt'))
	trained_model = load_model(args, len(classes), shape)

	# test_images = load_strings(make_dataset_path(args, 'meta/test.txt'))
	test_images = load_strings(make_dataset_path(args, 'meta/-test_part.txt'))

	n, n_true, n_false = 0, 0, 0
	preds, real = [], []
	corrects = collections.defaultdict(int)
	incorrects = collections.defaultdict(int)
	for i in test_images:
		(iclass, iname) = i.split('/')
		image_path = make_dataset_path(args, 'test/'+i+'.jpg')
		image = load_image(image_path, shape[:2])

		real.append(iclass)
		preds.append( trained_model.predict(image))
		
		# if classes[np.argmax(preds)] == iclass:
		# 	# print(i, iclass, "+")
		# 	n_true += 1
		# 	corrects[iclass] += 1
		# else:
		# 	# print(i, classes[np.argmax(preds)], iclass, "-")
		# 	n_false += 1
		# 	incorrects[iclass] += 1

	print(real,'\n----\n', preds)

	from sklearn.metrics import confusion_matrix
	import itertools
		
	# Compute confusion matrix
	cnf_matrix = confusion_matrix(real, preds)
	np.set_printoptions(precision=2)

	#--- class_names = [ix_to_class[i] for i in range(101)]

	plt.figure()
	fig = plt.gcf()
	fig.set_size_inches(32, 32)
	plot_confusion_matrix(cnf_matrix, classes=classes,
						title='Confusion matrix, without normalization',
						cmap=plt.cm.cool)
	plt.show()


def prepare(args):
	if os.path.isdir(make_dataset_path(args, 'test')) or os.path.isdir(make_dataset_path(args, 'train')):
		return

	def copytree(src, dst, ignore = None):
		if not os.path.exists(dst):
			os.makedirs(dst)
			shutil.copystat(src, dst)
		lst = os.listdir(src)
		if ignore:
			excl = ignore(src, lst)
			lst = [x for x in lst if x not in excl]
		for item in lst:
			s = os.path.join(src, item)
			d = os.path.join(dst, item)
			if os.path.isdir(s):
				copytree(s, d, ignore)
			else:
				shutil.copy2(s, d)

	def generate_dir_file_map(path):
		dir_files = defaultdict(list)
		with open(path, 'r') as txt:
			files = [l.strip() for l in txt.readlines()]
			for f in files:
				dir_name, id = f.split('/')
				dir_files[dir_name].append(id + '.jpg')
		return dir_files

	train_dir_files = generate_dir_file_map(make_dataset_path(args, 'meta/train.txt'))
	test_dir_files = generate_dir_file_map(make_dataset_path(args, 'meta/test.txt'))


	def ignore_train(d, filenames):
		print(d)
		subdir = d.split('/')[-1]
		to_ignore = train_dir_files[subdir]
		return to_ignore

	def ignore_test(d, filenames):
		print(d)
		subdir = d.split('/')[-1]
		to_ignore = test_dir_files[subdir]
		return to_ignore

	copytree(make_dataset_path(args, 'images'), make_dataset_path(args, 'test'), ignore=ignore_train)
	copytree(make_dataset_path(args, 'images'), make_dataset_path(args, 'train'), ignore=ignore_test)


def main():
	modes = {
		'prepare':		prepare,
		'train':		train,
		'testimage':	testimage,
		'fulltest':		fulltest,
		'plot':			plot_accuracy_loss,
		'plots':		plots
	}
	
	parser = argparse.ArgumentParser(description='Neural Network Analyser (NeuroTestAL)')
	parser.add_argument('-m',	help='режим работы: prepare/train/testimage/fulltest/plot/plots', 
								dest='mode', type=str, default='train')
	parser.add_argument('-b',	help='размер пакета данных',
								dest='batch_size', type=int, default=32)
	parser.add_argument('-p',	help='имя сохраненной модели',
								dest='model_path', type=str, default='food.hdf5')
	parser.add_argument('-i',	help='путь до тестового изображения',
								dest='image_path',type=str, default='')
	parser.add_argument('-s',	help='каталог набора данных', 
								dest='dataset', type=str, default='food-101')
	parser.add_argument('-e',	help='количество эпох',
								dest='epochs', type=int, default=25)
	parser.add_argument('-d',	help='величина dropout',
								dest='dropout', type=float, default=0.2)
	parser.add_argument('-n',	help='выбор сети (ResNet50, InceptionV3, InceptionResNetV2)',
								dest='network', type=str, default='InceptionResNetV2')
	parser.add_argument('-x', 	help='ширина целевого изображения',
								dest='x', type=int, default=256)
	parser.add_argument('-y', 	help='высота целевого изображения',
								dest='y', type=int, default=256)

	args = parser.parse_args()
	global shape
	shape = (args.x, args.y, 3)
	
	#проверка и создание при необходимости каталога для моделей
	if not os.path.exists(make_path('models/' + args.network)):
		os.makedirs(make_path('models/' + args.network))
	modes[args.mode](args)

if __name__ == '__main__':
	main()
