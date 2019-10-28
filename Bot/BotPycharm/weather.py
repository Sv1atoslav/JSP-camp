import random
import requests
from nltk import WordNetLemmatizer

from BotPycharm.Words import list_on_unknown_cities, list_on_unknown_questions, all_cities


def determine_weather(words):
    lemmatizer = WordNetLemmatizer()
    for x in words:
        if (x[0]).lower() == 'weather':
            return True
    return False


def get_weather(city_name):
    URL = 'https://api.openweathermap.org/data/2.5/weather?q='
    URL += city_name
    URL += '+&appid=886705b4c1182eb1c69f28eb8c520e20'
    location = 'Lviv'
    PARAMS = {'address': location}
    r = requests.get(url=URL, params=PARAMS)
    data = r.json()
    if data['cod'] != 200:
        return ''
    weather = [data['weather'][0]['description'], data['main']['temp'], data['main']['humidity']]
    return weather


def city_detection(sent):
    cities = []
    for x in range(len(sent)):
        if sent[x][1] == 'NNP' and all_cities.__contains__(sent[x][0]):
            print(sent[x][0])
            cities.append(sent[x][0].capitalize())
        else: continue
    print(cities)
    return cities


def answer_for_weather(sent, answered):
    if determine_weather(sent):
        cities = city_detection(sent)
        if len(cities) != 0:
            for city in cities:
                weather = get_weather(city.capitalize())
                if weather == '':
                    return True, list_on_unknown_cities[random.randint(0, len(list_on_unknown_cities) - 1)]
                elif not answered:
                    return True, ('Today is ' + str(weather[0].lower()) + ', the temperature is ' +
                                  str(weather[1]) + ' and a humidity - ' + str(weather[2]))
                else:
                    return True, ('In ' + city + ', today is ' + str(weather[0].lower()) +
                                  ', the temperature is ' + str(weather[1]) + ' and a humidity - ' + str(weather[2]))
        else:
            return False, list_on_unknown_questions[random.randint(0, len(list_on_unknown_questions) - 1)]
    else:
        return False, list_on_unknown_questions[random.randint(0, len(list_on_unknown_questions) - 1)]
