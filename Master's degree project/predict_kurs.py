#import warnings
#warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re

import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#----------------------------------------------------------------------------------------------------------

# Открытие истории изменения курса открытия USD, EUR и CNY к рублю
kurses = pd.read_csv('kurses.csv', ';')
print(kurses)
print('\n')

#----------------------------------------------------------------------------------------------------------
#print('---------------------------------------------------------------------------------------------------')

class Currency:
  def __init__(self, name, kurses):
    self.name = name
    self.df = kurses[['Date', name]]
    self.scaler = MinMaxScaler(feature_range = (0, 1))

  def make_features_labels(self, data):
      set_features = []
      set_labels = []
      len_df_train = len(data)
      for i in range(30, len_df_train):
          set_features.append(data[i-30:i, 0])
          set_labels.append(data[i, 0])
      return set_features, set_labels

  def prepare_features_labels(self, data):
      data_scaled = self.scaler.fit_transform(data.reshape(-1, 1))
      set_features, set_labels = self.make_features_labels(data_scaled)
      set_features, set_labels = np.array(set_features), np.array(set_labels)
      
      # **LSTM layers work on 3D data with the following structure (nb_sequence, nb_timestep, nb_feature).
      # 
      #     nb_sequence corresponds to the total number of sequences in your dataset (or to the batch size if you are using mini-batch learning).
      #     nb_timestep corresponds to the size of your sequences.
      #     nb_feature corresponds to number of features describing each of your timesteps.
      # 
      set_features = np.reshape(set_features, (set_features.shape[0], set_features.shape[1], 1))
      
      return set_features, set_labels

  def prepare_currency(self, currency):

      split_point = len(currency) - len(currency) // 10
      #print(len(currency))
      #print(split_point)

      #-----------

      currency_train = currency[:split_point]
      currency_test = currency[split_point-30:]

      train_features, train_labels = self.prepare_features_labels(currency_train)
      test_features, test_labels = self.prepare_features_labels(currency_test)

      #print('!!! Shapes !!! ', train_features.shape, train_labels.shape, test_features.shape, test_labels.shape)

      return train_features, train_labels, test_features, test_labels

  def train_model_1(self, train_features, train_labels, model_name, units_number):
      # создание
      model = Sequential()

      model.add(LSTM(units=units_number, return_sequences=True, input_shape=(train_features.shape[1], 1)))
      model.add(Dropout(0.2))
      model.add(LSTM(units=100, return_sequences=True))
      model.add(Dropout(0.2))
      model.add(LSTM(units=100, return_sequences=True))
      model.add(Dropout(0.2))
      model.add(LSTM(units=100))
      model.add(Dropout(0.2))

      model.add(Dense(units = 1))

      model.compile(optimizer = 'adam', loss = 'mean_squared_error')

      model.summary()

      # - # - # - # - #

      # обучение
      model.fit(train_features, train_labels, epochs = 300, batch_size = 32)
      # сохранение в файл для дальнейшего использования
      model.save(model_name)

  def predict(self, test_features, test_labels, model):
      predictions = model.predict(test_features)

      predictions = self.scaler.inverse_transform(predictions)
      real_labels = self.scaler.inverse_transform(test_labels.reshape(-1, 1))
      return real_labels, predictions

  # - # - # - # - #

  # Сравнение реальных значений с предсказанными по графику
  # Для одной модели
  def plot_preds(self, real_labels, predictions, model_name):
      currency = model_name[6:9].upper()
      plt.figure(figsize=(10,6))
      plt.plot(real_labels, color='blue', label='Реальный курс ' + currency)
      plt.plot(predictions , color='red', label='Предсказанный ' + model_name)
      plt.title('Сравнение реальных значений курса '+currency+' с предсказанными ' + model_name)
      plt.xlabel('Дни')
      plt.ylabel('Курс ' + currency)
      plt.legend()
      plt.show()

  # Сравнение реальных значений с предсказанными по графику
  # Для двух моделей
  def compare_plot_preds(self, real_labels, predictions_1, predictions_2, m_1, m_2):
      currency = m_1[6:9].upper()
      plt.figure(figsize=(10,6))
      plt.plot(real_labels, color='green', label='Реальный курс ' + currency)
      plt.plot(predictions_1 , color='blue', label='Предсказанный ' + m_1)
      plt.plot(predictions_2 , color='red', label='Предсказанный ' + m_2)
      plt.title('Сравнение реальных значений курса '+currency+' с предсказанными')
      plt.xlabel('Дни')
      plt.ylabel('Курс ' + currency)
      plt.legend()
      plt.show()

  # - # - # - # - #

  # Сравнение реальных значений с предсказанными
  def compare_values(self, real_labels, predictions, flag_print):
      p = predictions[:,0]
      r = real_labels[:,0]

      res = np.array([explained_variance_score(r, p)])
      res = np.append(res, mean_squared_log_error(r, p))
      res = np.append(res, mean_squared_error(r, p))
      res = np.append(res, np.sqrt(mean_squared_error(r, p)))
      res = np.append(res, mean_absolute_error(r, p))
      res = np.append(res, median_absolute_error(r, p))
      res = np.append(res, max_error(r, p))

      #print('!!! res модели !!! ', res)

      if flag_print:
          print('   explained_variance_score = ', res[0])
          print('   mean_squared_log_error = ', res[1])
          print('   mean_squared_error = ', res[2])
          print('   root_mean_squared_error = ', res[3])
          print('   mean_absolute_error = ', res[4])
          print('   median_absolute_error = ', res[5])
          print('   max_error = ', res[6])
      
      return res


  # - # - # - # - #


  def predict_check_model(self, test_features, test_labels, model, flag_print):
      real_labels, predictions = self.predict(test_features, test_labels, model)
      return self.compare_values(real_labels, predictions, flag_print)

  # для одной модели
  def predict_and_plot(self, test_features, test_labels, model, name):
      real_labels, predictions = self.predict(test_features, test_labels, model)
      self.plot_preds(real_labels, predictions, name)

  # для двух моделей
  def predict_and_plot_2(self, test_features, test_labels, model_1, model_2, m_1, m_2):
      real_labels, predictions_1 = self.predict(test_features, test_labels, model_1)
      real_labels, predictions_2 = self.predict(test_features, test_labels, model_2)

      self.compare_plot_preds(real_labels, predictions_1, predictions_2, m_1, m_2)


  # - # - # - # - #

  def do(self, flag_train=False, flag_print_values=False, flag_plot=False):
    train_f, train_l, test_f, test_l = self.prepare_currency(self.df.iloc[:,1].values)


    #----------------------------------------------------------------------------------------------------------
    #---------------------------------------------- LSTM ------------------------------------------------------
    print('------------------')

    # Функция для создания и обучения моделей по одному принципц

    # - # - # - # - #

    # Создание и обучение моделий по одному принципу, если надо
    if flag_train:
        self.train_model_1(train_f, train_l, f"model_{self.name}_LSTM_100.h5", 100)
        self.train_model_1(train_f, train_l, f"model_{self.name}_LSTM_300.h5", 300)
        
        print('!!!!!!!!!!!!!!! Models got trained !!!!!!!!!!!!!!!!!')


    # Загрузка моделей из файлов
    model_LSTM_100 = load_model(f"model_{self.name}_LSTM_100.h5")
    model_LSTM_300 = load_model(f"model_{self.name}_LSTM_300.h5")

    print('!!!!!!!!!!!!!!! Models got loaded !!!!!!!!!!!!!!!!!')

    #----------------------------------------------------------------------------------------------------------

    # Функция для предсказания значений выбранной моделью по выбранным тестовым данным

    flag_print = False

    # Общий вывод всех значений для сравнения работы всех моделей для валюты
    if flag_print_values:
        index_list = ['explained_variance_score', 'mean_squared_log_error', 'mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'max_error']
        compare_values_dict = { f"{self.name}_LSTM_100": self.predict_check_model(test_f, test_l, model_LSTM_100, flag_print),
                                f"{self.name}_LSTM_300": self.predict_check_model(test_f, test_l, model_LSTM_300, flag_print)}
        df_compare_values = pd.DataFrame(compare_values_dict, index=index_list)
        print('Models')
        print(df_compare_values)


    # - # - # - # - #

    if flag_plot:
        self.predict_and_plot_2(test_f, test_l, model_LSTM_100, model_LSTM_300, f'model_{self.name}_LSTM_100', f'model_{self.name}_LSTM_300')


#----------------------------------------------------------------------------------------------------------


cur_usd = Currency('USD', kurses)
cur_eur = Currency('EUR', kurses)
cur_cny = Currency('CNY', kurses)

# Флаги вызовов do
f_train = True
f_print_values = True
f_plot = True

cur_usd.do(f_train, f_print_values, f_plot)
cur_eur.do(f_train, f_print_values, f_plot)
cur_cny.do(f_train, f_print_values, f_plot)
