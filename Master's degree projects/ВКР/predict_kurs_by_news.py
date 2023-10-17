import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re
import time
import datetime, timedelta
from random import randint

import nltk
import string, re
from collections import Counter, OrderedDict
from nltk.tokenize import RegexpTokenizer

import itertools
from transformers import BertTokenizer, BertModel
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Открытие истории изменения курса USD, EUR и CNY к рублю
# Курс на момент открытия биржи с 11.01.2022 по 28.04.2023
# Дата рядом с курсом значит, что этот курс установлен в последний день работы биржы до этого дня
kurses = pd.read_csv('kurses.csv', ';')
#print(kurses)
# Удаляю первую строку в курсах, курс 11.01.2022 был установлен в 2021, для определения нее нету новостей
kurses = kurses.drop(index=[0]).reset_index(drop=True)
# Преобразую несколько столбцов в int, где только 0 или 1
for k in ['USD_ch_bi', 'EUR_ch_bi', 'CNY_ch_bi']:
	kurses[k] = kurses[k].astype('int')
print(kurses)
print('\n')


#----------------------------------------------------------------------------------------------------------


# Изменение дат в спаршенной таблицы новостей и сохранение в файл
flag_change_dates = False
if flag_change_dates:
	# Открытие спаршенной таблицы новостей с 01.01.2022 по 27.04.2023
	news_all = pd.read_csv('economy_news.csv', sep=";", encoding='utf-8-sig')
	print(news_all)
	print('\n')
	print(news_all.info())
	print('\nОсновные характеристики столбца количества просмотров новости')
	print(news_all.describe())


	# Список всех дней в заданном диапазоне
	dates_list = pd.date_range('2022-01-01', '2023-04-27').strftime('%d.%m.%Y').tolist()

	# По выходным биржа не работает, потому интересуют не все даты
	# Список дат, на которые менялся курс
	kurs_change_dates = list(kurses['Date'])

	# Выделение дней, на которые курс не менялся
	just_days = []
	for day in dates_list:
		if day not in kurs_change_dates:
			just_days.append(day)

	#----------------------------------------------------------------------------------------------------------

	# Изменение дат в таблице и сохранение в файл
	# Изменение, чтобы новости соответствовала дата, на которую она теоретически влияет

	# Сдвиг всеx новости на день вперед (идя с конца)
	for i in reversed(range(news_all.shape[0])):
		d = news_all.loc[i,'date']
		today = datetime.date(int(d[6:10]),int(d[3:5]),int(d[:2])) 	# для '22.02.2022'
		tomorrow = today + datetime.timedelta(days=1)
		news_all.loc[i,'date'] = tomorrow.strftime('%d.%m.%Y')
		#news_all.loc[i,'date1'] = tomorrow.strftime('%d.%m.%Y')

	# Сдвиг всеx новостей из списка just_days на день вперед (идя с начала)
	for day in just_days:
		for i in range(news_all.shape[0]):
			d = news_all.loc[i,'date']
			if d == day:
				today = datetime.date(int(d[6:10]),int(d[3:5]),int(d[:2])) 	# для '22.02.2022'
				tomorrow = today + datetime.timedelta(days=1)
				news_all.loc[i,'date'] = tomorrow.strftime('%d.%m.%Y')

	# Сохранение результата в файл
	news_all.to_csv('economy_news_changed_dates.csv', index=False, sep=";", encoding='utf-8-sig')


# Открытие файла всех новостей с измененными датами
news_all = pd.read_csv('economy_news_changed_dates.csv', ';', encoding='utf-8-sig')
print(news_all)


#----------------------------------------------------------------------------------------------------------


# Создание новостного фрейма, одна строка - один день
news = pd.DataFrame(columns=['date', 'news_number'])

list_of_dates = list(news_all['date'].unique())
news_numbers = news_all['date'].value_counts() [news_all['date']]


# Начало объединения всего нужного в один фрейм
# Добавление даты и количества новостей за эту дату
for i in list_of_dates:
	d = datetime.date(int(i[6:10]),int(i[3:5]),int(i[:2]))
	news.loc[len(news.index)] = [d.strftime('%d.%m.%Y'), int(news_numbers[i][0])]
	#news_numbers.append(news_numbers_all[i][0])

#print('\n')
#print(news)


# Изучение news_number
flag_descrive_news_number = False
if flag_descrive_news_number:
	index_list = ['Количество значений',
				'Количество уникальных значений',
				'Сумма всех',
				'Среднее значение',
				'Медианное значение',
				'Максимальноеное значение',
				'Минимальное значение'
				]

	vals = [len(news['news_number']),
			len(news['news_number'].unique()),
			sum(news['news_number']),
			np.mean(news['news_number']),
			np.median(news['news_number']),
			max(news['news_number']),
			min(news['news_number'])
			]

	nn_describe = pd.DataFrame(index=index_list, columns=['news_number'], data = vals)
	print('\n',nn_describe)


# Построение гистограмм
flag_hist = False
if flag_hist:
	# Гистограмма: Количество новостей каждый день
	fig = px.histogram(news, x = 'date', y = 'news_number', nbins = len(news['date']),
					title="Количество новостей каждый день")
	fig.update_xaxes(title_text="Дата")
	fig.update_yaxes(title_text="Количество новостей")
	fig.show()
	
	# Гистограмма: Количество дней для количества новостей
	fig = px.histogram(news['news_number'],
					title="Количество дней для количества новостей")
	fig.update_xaxes(title_text="Количество новостей")
	fig.update_yaxes(title_text="Количество дней")
	fig.show()

	# Гистограмма: Количество дней для количества новостей
	fig = px.histogram(news['news_number'],  nbins = 346,
					title="Количество дней для количества новостей")
	fig.update_xaxes(title_text="Количество новостей")
	fig.update_yaxes(title_text="Количество дней")
	fig.show()

#----------------------------------------------------------------------------------------------------------

# Задание интересуемого количества самых популярных новостей за день
number_of_top = 18
#number_of_top = 40
file_name_ending = str(number_of_top)+'.csv'
# Создание списка названия колонок
list_tops = []
for i in range(number_of_top):
	list_tops.append('top'+str(i+1))

# Объединение самых просматриваемых новостей по дням
flag_make_news_top_n = False
if flag_make_news_top_n:
	# В большинство групп дней было опубликовано 40 и более новостей

	empty_cols = pd.DataFrame(columns=list_tops)
	news = pd.concat([news,empty_cols])

	# Добавление самых просматриваемых новостей за каждый день в news
	for d in list_of_dates:
		# Достаю записи за день
		news_one_date = news_all[news_all['date'] == d]
		# Сортирую новости по количеству просмотров и обновляю индексы
		news_one_date = news_one_date.sort_values(by='views', ascending=False).reset_index(drop=True)
		#print(news_one_date)

		# Количество топ новостей за день
		number_of_top_in = number_of_top
		nn = len(news_one_date)
		if nn < number_of_top_in:
			number_of_top_in = nn

		# Добавление самых просматриваемых новостей в news
		for i in range(number_of_top_in):
			top = 'top'+str(i+1)
			news.loc[news['date'] == d, top] = news_one_date.name.loc[i]

	# Сохранение результата в файл
	news.to_csv('news_top'+file_name_ending, index=False, sep=";", encoding='utf-8-sig')


# Открытие файла top number_of_top новостей за каждый день
news = pd.read_csv('news_top'+file_name_ending, ';', encoding='utf-8-sig')
#print(news)

# Изучение данных на предмет пропусков
flag_chek_NaN = False
if flag_chek_NaN:
	print('\nКоличество пропусков (NaN) по столбцам:')
	print(news.isnull().sum())

	n_all = news.shape[0]*(news.shape[1]-2)
	n_miss = news.isnull().sum().sum()
	print('\nВсего ячеек новостей: ', n_all)
	print('Количество пропусков: ', n_miss)
	print('Процент пропусков: ', 100 * n_miss/n_all)
	
	# До top18 вообще без потерь

#----------------------------------------------------------------------------------------------------------

# Предобработка данных
# Очистка данных

tokenizer = RegexpTokenizer('\W+', gaps=True)
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('russian')

# Минимальная обработка текста
def clean(text):
    text = re.sub('ё','е',str(text))
    text = re.sub('\\n',' ',str(text)) # удаляет переносы строк
    text = re.sub('\s{2,}', " ", text) # удаляет лишние пробелы
    text = re.sub(r'\w*\d+\w*', '', text) # удаляет цифры
    text = re.sub(r'[^\яа-яА-Я0-9@]',' ',str(text))
    return text

# Чистит от стоп слов токенезированный текст
def clean_from_stopwords(text):
	text = [token for token in text if re.match(r'[а-яА-Я]', token) and token not in stopwords]
	
	flag_stay_tokenize = False
	# Строки ниже, чтобы убрать токинезацию
	if not flag_stay_tokenize:
		text = re.sub(r'[^\яа-яА-Я0-9@]',' ',str(text))
		text = re.sub(r'\w*\d+\w*', '', text) # удаляет цифры
		text = re.sub('\s{2,}', " ", text) # удаляет лишние пробелы

	return text


# Добавление лемматизированного текста
symbol_mapper = {
    'ё': 'е',
    '“': '«',
    '”': '»',
    '—': '-',
    '–': '-'
}

def pos_lemmas(annot):
    """ Returns: Мама_NOUN мыть_VERB рама_NOUN """
    
    postags = itertools.chain.from_iterable(annot['postag'])
    lemmas = itertools.chain.from_iterable(annot['lemma'])
    result = [(next(lemmas), next(postags)) for _ in annot['tokens']]
    result = ' '.join(['_'.join(pair) for pair in result if not pair[0].isdigit()])  # remove numbers
    
    for symbol in symbol_mapper.keys():
        result = result.replace(symbol, symbol_mapper[symbol])
    
    return result


file_name = 'news_top'+str(number_of_top)+'_cleaned.csv'

# Очистка всех новостных заголовков
flag_cleane_news = False
if flag_cleane_news:
	for d in news['date']:
		for i in range(number_of_top):
			top = 'top'+str(i+1)

			# Очистка данных
			news.loc[news['date']==d, top] = news.loc[news['date']==d, top].apply(clean)
			# Токенизирование
			news.loc[news['date']==d, top] = news.loc[news['date']==d, top].map(lambda row: tokenizer.tokenize(row.lower()))
			# Избавление от стоп слов
			news.loc[news['date']==d, top] = news.loc[news['date']==d, top].apply(clean_from_stopwords)
			# Добавление лемматизированного текста
			#news.loc[news['date']==d, top] = pos_lemmas(news.loc[news['date']==d, top])

	# Сохранение результата в файл
	news.to_csv(file_name, index=False, sep=";", encoding='utf-8-sig')


# Открытие очищенного файла top 40 новостей за каждый день
news = pd.read_csv(file_name, ';', encoding='utf-8-sig')
#news_cleaned = pd.read_csv(file_name, ';', encoding='utf-8-sig')
print(news)


#----------------------------------------------------------------------------------------------------------

# Дополнительная обработка

# Метод TF-IDF оценки важности слов на основе частот появления 
flag_tfidf_features = True
if flag_tfidf_features:
	def tfidf_features(dataframe):
		tfidf = TfidfVectorizer()  # минимальная частота для рассмотрения слова
		features = tfidf.fit_transform(dataframe)
		feature_names = tfidf.get_feature_names()
		return features, feature_names

	# Применение TF-IDF ко всем столбцам новостей
	for col in list_tops:
		text = news[col].apply(str)  # преобразование серии pandas в список строк
		features, feature_names = tfidf_features(text)
		news_tfidf = pd.DataFrame(features.todense(), columns=feature_names)
		news = pd.concat([news, news_tfidf], axis=1)

	print(news)
	print(news[list_tops])


# Метод выделения n-грамм, последовательностей из n элементов
flag_char_ngrams_features = False
if flag_char_ngrams_features:
	def char_ngrams_features(dataframe):
		vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), lowercase=False)
		features = vectorizer.fit_transform(dataframe)
		feature_names = vectorizer.get_feature_names()
		return features, feature_names

	# Применение выделения n-грамм ко всем столбцам новостей
	for col in news[list_tops].columns:
		features, feature_names = char_ngrams_features(news[col])
		news_ngrams = pd.DataFrame(features.todense(), columns=feature_names)
		news = pd.concat([news, news_ngrams], axis=1)

	print(news)


# Второй способ выделения n-грамм, представляющих собой комбинации n слов
flag_char_ngrams_features_2 = False
if flag_char_ngrams_features_2:
	n = 3 	# длина n-граммы
	vectorizer = CountVectorizer(ngram_range=(n,n))

	# Применение выделения n-грамм ко всем столбцам новостей
	for col in news[list_tops].columns:
		X = vectorizer.fit_transform(news[col].values)
		
		# Добавление столбцов с n-грамм
		ngrams = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
		news = pd.concat([news, ngrams], axis=1)

	print(news)


#----------------------------------------------------------------------------------------------------------

# sklearn
# Функция для обучения регрессионных моделей, проверки предсказания и сохранения результатов в файл
def predict_sklearn(X, currency, test_size, res_file_name):
	# Задание лейблов
	y = kurses[currency]

	# Разбиение X и y на обучающую и тестовую выборку
	train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=test_size, random_state=42)

	# Создание пустого датафрейма для дальнейшего накомпления информации
	index_list = ['Models - '+currency+' - top'+str(number_of_top), 'Accuracy', 'Precision', 'Recall', 'F1 Score']
	prediction_results = pd.DataFrame(columns=index_list)

	# Задание моделей для дальнейшего обучения
	models = [LogisticRegression(), SVC(), RandomForestClassifier(), GradientBoostingClassifier(), KNeighborsClassifier(), DecisionTreeClassifier()]
	model_names = ["Logistic Regression", "Support Vector Classifier", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors", "Decision Tree"]

	# Обучение каждой модели на train и получение предсказаний для test
	for model, model_name in zip(models, model_names):
		model.fit(train_data, train_labels)
		predictions = model.predict(test_data)
		
		# Calculate evaluation metrics for each model
		accuracy = accuracy_score(test_labels, predictions)
		precision = precision_score(test_labels, predictions)
		recall = recall_score(test_labels, predictions)
		f1 = f1_score(test_labels, predictions)
		
		# Добавление результатов в общую таблицу
		i = len(prediction_results.index)
		prediction_results.loc[i] = [model_name, accuracy*100, precision*100, recall*100, f1*100]

	print(prediction_results)

	# Cохранение таблицы в файл
	prediction_results.to_csv(res_file_name, index=False, sep=";", encoding='utf-8-sig')


# Выделение только заголовков новостей
news_text = news[list_tops]
# Объединение всех текстов в один string для каждой строки
text_data = news_text.apply(lambda x: " ".join(str(x) for x in x), axis=1)
# Конвертирование строки к численным данным
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)

# обучение и проверка работы
predict_sklearn(X, 'USD_ch_bi', 0.2, 'pred_res_USD_bi_top'+file_name_ending)
predict_sklearn(X, 'EUR_ch_bi', 0.2, 'pred_res_EUR_bi_top'+file_name_ending)
predict_sklearn(X, 'CNY_ch_bi', 0.2, 'pred_res_CNY_bi_top'+file_name_ending)


#----------------------------------------------------------------------------------------------------------


# Объединение данных всех интересующих валют 
currency_list = ['USD', 'EUR', 'CNY']

# Формирование имен столбцов сводной таблицы
cols = ['Date']
for k in currency_list:
	cols.append(k)
	cols.append(k+'_ch')
	cols.append(k+'_ch_bi')


# Выделение только заголовков новостей
news_text = news[list_tops]
# Объединение всех текстов в один string для каждой строки
text_data = news_text.apply(lambda x: " ".join(str(x) for x in x), axis=1)
# Конвертирование строки к численным данным
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)


#for currency in currency_list:
for currency in ['USD_ch_bi']:
	y = kurses[currency]
	train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=.2, random_state=42)





