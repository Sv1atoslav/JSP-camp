import nltk
import pandas as pd

data = pd.read_csv('cities.csv')
data = data.drop(['geonameid'], axis=1)
all_cities = [y for x in data.values for y in x]

list_on_unknown_questions = ['Sorry, I don’t understand.', 'Sorry, what?', 'Wait, i can not understand.',
                             'I do not mind.', 'I can not understand you.', 'Something goes wrong.',
                             'I can not answer on this question.', 'You confused me!', 'Try another question.',
                             'It is hard to answer on it.', 'Hmmmm, what?', 'Are you sure in your question?']
list_on_unknown_cities = ['I think you entered incorrect city.', 'You should reenter city.',
                          'Does this city exist?', 'Looking for the city... \nNothing find.',
                          'City is incorrect.', 'City  you wrote  is wrong.', 'I can not find weather for this city.',
                          'Oops, you confused me!', 'I do not know this city.', 'Hmmm, my city API get an error.',
                          'Are you sure that it is right city?']


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent
