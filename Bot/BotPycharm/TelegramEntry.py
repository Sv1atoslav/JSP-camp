import telebot

from BotPycharm.Chatting import answer_weather

bot_token = '1051426246:AAF9vQ3UOQ9R4YxA_SYQ5gdmNY_cJqEYZXY'

bot = telebot.TeleBot(token=bot_token)


def start_messaging():
    @bot.message_handler(commands=['start'])
    def greeting(message):
        bot.reply_to(message, 'Hi there,you can ask me for the weather in some city. ("Tell me the weather'
                              ' in London")')

    @bot.message_handler()
    def greeting(message):
        answer = answer_weather(message.text)
        if message.text.lower()=='yes':
            bot.reply_to(message, 'NO')
        else: bot.reply_to(message, answer)

    bot.polling()


def start_bot():
    start_messaging()

start_bot()
