from BotPycharm import preprocess, answer_for_weather

answered_weather = False
def answer_weather(text):
    sent = preprocess(text)
    global answered_weather
    answered_weather, answer = answer_for_weather(sent, answered=answered_weather)
    return answer