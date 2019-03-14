#!/usr/bin/python3

# logging settings
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# giorgiobot neural network classifier
from classify import classify

# telegram bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
updater = Updater(token='447738286:AAEmBxdnrD5gaFIJjsgv6PQ6u4v3RRvuaxs')

def start(bot, update):
	bot.send_message(chat_id=update.message.chat_id, text="Ciao, sai tutto.")

def message(bot, update):
	results = classify(update.message.text)
	if (len(results)>0):
		bot.send_message(chat_id=update.message.chat_id, text=results[0][0])

start_handler = CommandHandler('start', start)
updater.dispatcher.add_handler(start_handler)

message_handler = MessageHandler(Filters.text, message)
updater.dispatcher.add_handler(message_handler)

updater.start_polling()