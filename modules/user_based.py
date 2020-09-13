rekomendacje = []

### musimy przekazac id filmow ocenionych z bara oraz ich oceny oraz movie2rating
def recom_u(filmy_ocenione, oceny, movie2rating):
    import pickle
    import numpy as np
    import pandas as pd
    # from sklearn.utils import shuffle
    from sortedcontainers import SortedList

    # loadujemy date

    with open('data02/json/user2movie.json', 'rb') as f:
        user2movie = pickle.load(f)  # jaki film oceniony przez jakeigo usera movie2user[3100]
        
    with open('data02/json/movie2user.json', 'rb') as f:
        movie2user = pickle.load(f) # jaki user oceniil jaki film [id_filmu] 
        
    with open('data02/json/usermovie2rating.json', 'rb') as f:
        usermovie2rating = pickle.load(f) # [film_id, user_id] i daje ci ocene

    
    print(movie2rating)
    K = 5  # ilosc neighboursow
    limit = 2  # number of common movies users must have ( minimum )
    neighbors = []
    averages = []
    deviations = []
    #  szukamy 25 closest sasiadow
    movies_i = filmy_ocenione  # filmy_ocenione ids ( z bara pobierane dane)
    movies_i_set = set(movies_i)

    # liczymy srednie i odchylenie
    ratings_i = { movie:movie2rating[movie] for movie in movies_i}  # rating dla filmy konkretnego usera  # array OCENY utworzone w app.py
    avg_i = np.mean(oceny)
    dev_i = {movie:(rating - avg_i) for movie, rating in ratings_i.items()}  # dict odchylen
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))  # dotem masz mnozenie item i macierz a razy item i macierz b 

    # zapisujemy for later use
    averages.append(avg_i)
    deviations.append(dev_i)

    sl = SortedList()  # bedzie sortowac automatycznie , wiec bedizemy brali tlyko top 25 entries
    common_movies = set()
    N=10000
    for j in range(N):
        # if j!= i: # zebys samego siebie nie liczyl ( sam nie mozesz byc dla siebie sasiadem)
        movies_j = user2movie[j]
        movies_j_set = set(movies_j)
        common_movies = (movies_i_set & movies_j_set) # czesc wspolna
        if len(common_movies) > limit:
            # srednie odchyelenie i sigma
            try:
                ratings_j = { movie:usermovie2rating[(j, movie)] for movie in movies_j}  # rating dla filmy konkretnego usera
            except KeyError:
                pass
            avg_j = np.mean(list(ratings_j.values()))
            dev_j = {movie:(rating - avg_j) for movie, rating in ratings_j.items()}  # dict odchylen
            dev_j_values = np.array(list(dev_j.values()))
            sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))  # dotem masz mnozenie

            # liczymy teraz wspolczynnik korelacji !!!!!!!!!  wzor w evernote na wage ta w_ij
            numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)  # suma produktu odchylen dla 2 userow dla fimlow ktore ocenili
            w_ij = numerator / (sigma_i * sigma_j)

            # tu ciekawostka O(N^2 / 2) = O(N^2), bo stale nic nie wnosza w zlozonosci
            # teraz storujemy to do naszej SL dla usera i 
            # max value (1) jest najblizsza
            # bierzemy ujemna wage, bo w korelacji Pearsona im wieksza tym lepsza, a SortedList
            # storuje nam to rosnąco, stad ujamna waga
            sl.add((-w_ij, j))
            if len(sl) > K: # Zeby bylo tlye ile chcemy sasiadow nie
                del sl[-1]
        
    neighbors.append(sl)
        
    print(neighbors)    
    best_match = neighbors[0][0][1]  # id usera z najwieksza waga
    # user2title potrzebne nam aby wybrac tutyl naszego najelpiej pasujacego usera ;)
    df = pd.read_csv('data02/do_user_based.csv')
    df.drop(columns={'Unnamed: 0'}, inplace=True)

    # pd.set_option('display.max_row', 1000)
    filt = df['userId'] == best_match  # z tym mamy najwieksza wage
    # neighbors
    sorted_df = df[filt].sort_values(by='rating', ascending=False)
    
    for index, row in sorted_df.head(25).iterrows():
        rekomendacje.append(row['title'])