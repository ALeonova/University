'''
Данный код создан для парсинга новостей с сайта РИА Новости (ria.ru) по тегу Экономика (economy)
в задаваемом диапазоне дат. Даты и тег можно легко поменять.
Создается таблица с колонками: дата публикации, количество просмотров, заголовок статьи, ссылка на статью.
Таблица сохраняется в csv файйл.
'''
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from random import randint
from alive_progress import alive_bar

from bs4 import BeautifulSoup
import requests
import re

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import lxml

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select


# Убирает все скобочки, тэги и прочую ерунду
def html_stripper(text):
    return re.sub('<[^<]+?>', '', str(text)) 


# Изменение символов, неизвестных выбранной кодировке
def check_for_broken(text):
    text = re.sub(r'ü',r'ue', text)
    text = re.sub(r'ä',r'ae', text)
    text = re.sub(r'ö',r'oe', text)
    text = re.sub(r'ß',r'ss', text)
    text = re.sub(r'ó',r'o', text)
    text = re.sub(r'Ø',r'O', text)
    text = re.sub(r'&amp;',r'&', text)
    text = re.sub(r'﻿',r'', text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


# Листание страницы вниз
def scroll_page(driver, pause_time):
    # Получение высоты прокрутки
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Листание вниз до конца страницы
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Ожидание загрузки страницы
        time.sleep(pause_time)

        # Рассчет новой высоты прокрутки и сравнение ее с предыдущей 
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


# Нажатие на кнопу внизу для загрузки дополнительных материалов, если она есть
def more_news(driver, pause_time):
    driver.find_element(By.XPATH, '//*[@id="content"]/div/div[1]/div/div[3]').click()
    time.sleep(pause_time)


# Инициализация и запуск веб драйвера по ссылке
def start_driver(link):
    # опции для игнорирования некоторых ошибок и предупреждений
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    # Запускаем драйвер, открываем веб-страницу
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1500,1000)
    driver.implicitly_wait(randint(5, 10)) # рандомная задержка, в секундах
    driver.get(link)
    return driver 


# Выбора периода даты "За год", в итоге не используется
def date_range(driver):
    # открытие выбора периода даты
    driver.find_element(By.XPATH, '//*[@id="content"]/div/div[1]/div/div[1]/div[3]').click()
    time.sleep(3)
    # выбора периода даты "За год"
    driver.find_element(By.XPATH, '/html/body/div[20]/div[2]/ul/li[3]').click()


# Дополнение таблицы, актуализация данных
def data_actualization(new_start, new_end, df_csv, res_name_csv):
    df_old = pd.read_csv(df_csv, sep=";", encoding='utf-8-sig')
    # Получение новых данных
    parse_all(new_start, new_end, 'news_last_actualization_add.csv')
    df_new = pd.read_csv('news_last_actualization_add.csv', sep=";", encoding='utf-8-sig')
    # Соединение всех данных
    news_result = pd.concat([df_old, df_new])
    # Сохранение обновленной таблицы
    news_result.to_csv(res_name_csv, index=False, sep=";", encoding='utf-8-sig')


#----------------------------------------------------------------------------------------------------------


# Парсинг новостей за одну дату, создание супа данных
def parse_by_date(date):
    #print('Дата:', date)
    # создание ссылки новостного архива заданного дня 
    link = 'https://ria.ru/economy/' + date + '/' 

    driver = start_driver(link)

    # Суп из HTML-кода страницы
    soup = BeautifulSoup(driver.page_source, 'lxml')

    # Выделение количества новостей за день, оно влияет на количество страниц
    number_of_news = int(html_stripper(soup.find('div', class_ = 'rubric-count m-active').span))
    #print('!!!!!!!!!!!!')
    #print('Количество новостей: ', number_of_news)
    #print('!!!!!!!!!!!!\n')

    # 20 - максимальное количество новостей на странице ria.ru
    if number_of_news > 20:
        # Листание страницы вниз
        scroll_page(driver, 1)

        # Нажатие на кнопку внизу для загрузки дополнительных материалов, если она есть
        more_news(driver, 5)

        # Листание страницы вниз, дальше новые данны будут прогружаться автоматически
        scroll_page(driver, 1)

        # Обновление супа, появились новые новости
        soup = BeautifulSoup(driver.page_source, 'lxml')
    
    #time.sleep(randint(2, 4))

    # Закрытие драйвера, больше он не нужен
    driver.close()
    # Передача супа данных
    return soup


# Функция для парсинга данных в диапазоне дат [start, end] и сохранение таблицы в csv файл
def parse_all(start, end, file_name):
    # Гененирование списка всех интересующих дат для ссылок
    dates_list = pd.date_range(start, end).strftime('%Y%m%d').tolist()
    total = len(dates_list)
    print('\nКоличество дат = ', total, '\n')

    # Создание пустого датафрейма для дальнейшего накомпления информации
    # Столбцы: дата публикации, количество просмотров, заголовок статьи, ссылка на статью
    index_list = ['date', 'views', 'name', 'url']
    news_table = pd.DataFrame(columns=index_list)

    with alive_bar(total) as bar:   # индикатор прогресса
        # Вытаскивание данных для каждой даты
        for date in dates_list:
            # Получение супа данных для даты
            soup = parse_by_date(date)        

            # Выделение списка из кусков кода, относящихся к новостям
            news = soup.find_all(class_="list-item")

            # Добавление всех новостей дня в общую таблицу
            for i in news:
                name = html_stripper(i.find(class_="list-item__title color-font-hover-only"))
                name = check_for_broken(name)
                url = i.find(class_="list-item__title color-font-hover-only")['href']
                #date_time = html_stripper(i.find(class_="list-item__date"))
                # Получение даты из ссылки
                url_date = re.search(r'\d{8}', url)[0]
                date = url_date[:4]+'-'+url_date[4:6]+'-'+url_date[6:]
                views = html_stripper(i.find(class_="list-item__views-text"))
                #tags = html_stripper(i.find_all(class_="list-tag m-active color-border color-font color-svg m-add"))

                # Добавление новой строки в конец
                news_table.loc[len(news_table.index)] = [date, views, name, url]
                
            bar()   # Изменение индикатора прогресса

    print('\nКоличество новостей = ', news_table.shape[0], '\n')

    # Cохранение таблицы в файл
    # encoding='cp1251'
    news_table.to_csv(file_name, index=False, sep=";", encoding='utf-8-sig')


#----------------------------------------------------------------------------------------------------------


# Вызов парсинга данных в диапазоне дат [start, end] и сохранение таблицы в csv файл
flag_parse_all = False
if flag_parse_all:
    #file_name = 'economy_news.csv'
    file_name = 'news_table_.csv'
    parse_all('2022-01-01', '2023-03-28', file_name)

# Дополнение старой таблицы, актуализация данных
flag_actualize = False
if flag_actualize:
    old_table_name = 'news_table_all.csv'
    res_file_name = 'economy_news_.csv'
    data_actualization('2023-03-29', '2023-04-28', old_table_name, res_file_name)


# Открытие актуализированной таблицы новостей
economy_news = pd.read_csv('economy_news.csv', sep=";", encoding='utf-8-sig')

print(economy_news)
print('!!!!!!!!!!!!!!!!')
print(economy_news.info())
print('!!!!!!!!!!!!!!!!')
print(economy_news.describe())
