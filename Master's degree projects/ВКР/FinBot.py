from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.callback_data import CallbackData
from aiogram.dispatcher.filters.state import State, StatesGroup
#from aiogram.utils.emoji import emojize
import emoji

import requests
import re
from bs4 import BeautifulSoup

import pandas as pd
import time
import datetime, timedelta


API_TOKEN = '6260473010:AAE9WsOkxQ_Msa52mJ02n-0vJP6gvN5wVFI'
 
# Запуск сервера телеграм-бота
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

class Form(StatesGroup):
   currencyState = State() 

#---------------------------------------------------------------------------------------------------------------------------------------------------------

# команда help
# в идеале, добавить всё основное в main как InlineKeyboardButton 
# добавить /main - все основные функции?!!!
help_message = '''
   Доступные команды:
   /start - приветствие
   /help - список команд
   
   /links - полезные ссылки
   /kurs_now [интересующая валюта]- нынешний курс интересующей валюты, вызов без параметра выдаст инлайн кнопки для популярных валют 
   /kurs [интересующая валюта интересующая дата] - предсказание курса валют на указанную дату, для прошедшей даты вернет реальное значение, вызов без параметра даты выдаст предсказание на завтра (для usd, eur, cny)
   '''

@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply(help_message)

#---------------------------------------------------------------------------------------------------------------------------------------------------------

# команда start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
   await message.reply('Добро пожаловать!\nЯ FinAnalysisBot!\nМоё предназначение, помогать всем нуждающимся узнать актуальную информацию про финансовые инструменты.\nДля знакомства с моим функционалом вызовите /help')

#---------------------------------------------------------------------------------------------------------------------------------------------------------

# команда links
links = InlineKeyboardMarkup(row_width=1)
linksButton1 = InlineKeyboardButton(text='Официальные курсы валют', url='https://www.cbr.ru/currency_base/daily/')
linksButton2 = InlineKeyboardButton(text='Акции и котировки на сегодня', url='https://www.banki.ru/investment/shares/?source=submenu_share')
linksButton3 = InlineKeyboardButton(text='Вклады', url='https://www.banki.ru/products/deposits/?source=submenu_deposits')
links.add(linksButton1,linksButton2,linksButton3)
 
@dp.message_handler(commands='links')
async def url_command(message: types.Message):
   await message.answer('Основные ссылки:', reply_markup=links)

#---------------------------------------------------------------------------------------------------------------------------------------------------------


# убирает все скобочки, тэги и прочую ерунду
def html_stripper(text):
    return re.sub('<[^<]+?>', '', str(text)) 

# находит нынешний курс запрошенной валюты
def currentCurrency(currency):
#    return "???????????????? " + currency
   url = 'https://www.cbr.ru/currency_base/daily/'
    
   response = requests.get(url)
   #response.raise_for_status()
   soup = BeautifulSoup(response.text, 'lxml')

   cleanedSoup = html_stripper(soup.findAll('table', class_='data'))
   listCleanedSoup = cleanedSoup.split('\n\n')

   if __name__ == '__main__':
     print('Курс ' + currency)
   f = False
   all_res = ''

   for el in listCleanedSoup:
      if re.findall(currency.lower(), el.lower()) != []:
         res = el.split('\n')
         all_res += res[2] +' '+ res[4] +': '+ res[5] +' к '+ res[3] + '\n'
         f = True
       
   if f:
      return all_res
   else:
      return 'Такая валюта не найдена'


kurs_cb = CallbackData("currency_prefix", 'currency') 

def get_kb():
   kurs_now = InlineKeyboardMarkup(row_width=1)
   knButton1 = InlineKeyboardButton(text=emoji.emojize("🇺🇸") + ' USD', callback_data=kurs_cb.new(currency = 'USD'))
   knButton2 = InlineKeyboardButton(text=emoji.emojize('🇪🇺') + ' EUR', callback_data=kurs_cb.new(currency = 'EUR'))
   knButton3 = InlineKeyboardButton(text=emoji.emojize("🇨🇳") + ' CNY', callback_data=kurs_cb.new(currency = 'CNY'))
#   knButton4 = InlineKeyboardButton(text='Введите интересующую валюту', callback_data=kurs_cb.new(currency = 'other'))
   kurs_now.add(knButton1,knButton2,knButton3)  # ,knButton4
   return kurs_now;


# команда kurs_now
@dp.message_handler(commands='kurs_now')
async def url_command(message: types.Message):
   if int(len(message.text)) < 10:
      await message.answer('Узнать нынешний курс', reply_markup=get_kb())
   else:
      print('!kurs_now\n', message.text)
      await message.answer(text = currentCurrency(message.text[10:]))


# !!!!!!!!!!!! не работает ! не нужно
'''
@dp.callback_query_handler(kurs_cb.filter(currency='other'))
async def kbhandler(query: types.CallbackQuery, callback_data: dict):
   await bot.send_message(text = "Введите название валюты", chat_id=query.from_user.id)
'''

@dp.callback_query_handler(kurs_cb.filter())
async def kbhandler(query: types.CallbackQuery, callback_data: dict):
   await bot.send_message(text = currentCurrency(callback_data.get("currency")), chat_id=query.from_user.id)
#   bot.edit_message_text(,
#query.from_user.id,
#query.message.message_id)

#---------------------------------------------------------------------------------------------------------------------------------------------------------

kurses = pd.read_csv('kurses.csv', ';')
#print(kurses)

# находит нынешний курс запрошенной валюты
def currencyByDate(currency, d):
   supported_cur = ['USD', 'EUR', 'CNY']

   print('Курс ' + currency)
   print('Дата ' + d)

   if currency.upper() not in supported_cur:
      return 'Такая валюта не найдена'


   today = datetime.date.today()
   day = datetime.date(int(d[6:10]),int(d[3:5]),int(d[:2]))  # для '22.02.2022'

   if day < datetime.date(2022,1,1):
      return 'Не знаю такой ранней информации'

   if day > today + datetime.timedelta(days=7):
      return 'Слишком далекое предсказание'

   res_row = kurses.loc[kurses['Date'] == d]
   print(res_row)
   answ = res_row[currency.upper()].reset_index(drop=True)
   print(answ.loc[0])

   return 'Курс ' + currency + ' ' + d + ': ' + str(float(re.sub(',','.',str(answ.loc[0]))))
   return 'Кря'
       



# команда kurs
@dp.message_handler(commands='kurs')
async def url_command(message: types.Message):
   if int(len(message.text)) < 10:
      await message.answer(text = currentCurrency(message.text[6:9]))
      print('!kurs\n',message.text[6:9])
   else:
      print('!kurs\n',message.text)
      await message.answer(text = currencyByDate(message.text[6:9], message.text[10:]))



#---------------------------------------------------------------------------------------------------------------------------------------------------------

# команда main !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mb = InlineKeyboardMarkup(row_width=1)
mainButton1 = InlineKeyboardButton(text='Полезные ссылки', callback_data=links) # !!!!!!!!!!! ндаааааааааааааааааа
# курс валют
# инвестиции
mainButton2 = InlineKeyboardButton(text='Вклады0', url='https://www.banki.ru/products/deposits/?source=submenu_deposits')
mb.add(mainButton1,mainButton2)
 
@dp.message_handler(commands='main')
async def url_command(message: types.Message):
   await message.answer('Основные функции:', reply_markup=mb)

#---------------------------------------------------------------------------------------------------------------------------------------------------------

# основная реакция
@dp.message_handler()
async def echo(message: types.Message):
   await message.answer(message.text)

#---------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
   executor.start_polling(dp, skip_updates=True)