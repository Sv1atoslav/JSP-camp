from BotPycharm.Words import preprocess
from BotPycharm.weather import answer_for_weather

if __name__ == '__main__':
    answered = False
    while (True):
        text = input()
        if text == 'q':
            break
        sent = preprocess(text)
        answered = answer_for_weather(sent, answered)
    #             What is the weather in Lviv?
