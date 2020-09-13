import numpy as np 
import pandas as pd

#movies_list = pd.read_csv("data/tytuly.csv")
movies_list_4000 = pd.read_csv("data02/title_id.csv")
#movies_list_6500 = pd.read_csv("data/6500/titles_and_ids.csv")
#titles6500 = movies_list_6500.title
titles4000 = movies_list_4000.title
#titles5000 = movies_list.title  # oba Series moza interowac jak liste

movies_list_4000 = movies_list_4000.sort_values(by='movieId')
movies_list_4000.drop(columns={'Unnamed: 0'}, inplace=True)
# print(movies_list_4000)

ranking = pd.read_csv('data02/ranking2.csv')

pozycja = [x + 1  for x in range(4000)]
ranking['ranking'] = pozycja  # dodajemy numer w rankingu w zakladce ranking filmow


