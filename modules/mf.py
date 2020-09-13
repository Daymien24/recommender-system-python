results = []

def recom(filmy_ocenione, oceny, movies, same_tytuly, mylist):
    import numpy as np
    moviess = np.array(filmy_ocenione)
    ratings = np.array(oceny)

    # loadujemy updatniete juz macierze U(ile film zwiera k atrybutow) oraz c(bias filmu j)
    K = 10
    reg = 0.05
    U = np.loadtxt('U.txt', dtype=int)
    c = np.loadtxt('c.txt', dtype=int)
    b = np.zeros(1)
    mu = 3.2886697185957057
    V = np.random.randn(K)
    # b = np.zeros(1)

    m_ids = np.array(moviess)
    r = np.array(ratings)
    matrix = U[m_ids].T.dot(U[m_ids]) + np.eye(K) * reg
    vector = (r - c[m_ids] - mu).dot(U[m_ids])  # z b bedzie (r - b - c[m_ids} ....])
    # bi = (r - U[m_ids].dot(V) - c[m_ids] - mu).sum()

    # set the updates
    V = np.linalg.solve(matrix, vector)
    # b = bi / (len(movies) + reg)

    # ----------------- TUTAJ PREDYKCJA ---------------------------------
    wyniki = {}
    licznik = 0
    rekomendacje = []
    for j in range(4000):
        p = V.dot(U[j]) + c[j] + mu  # probuje bez biasu uzytkownika zobaczymy jak wyjdzie
        wyniki[j] = p
        # dobra to mozemmy iterowac od j in range 4000 i prediction dla kazdego bedzie sie liczyc i tlyko liste zrobic 
    wyniki = {k: v for k, v in sorted(wyniki.items(), key=lambda item: item[1],reverse=True)}
    for m, r in wyniki.items():
        print(f"fillm: {m} ocena : {r}")
        rekomendacje.append(m)
        licznik+=1
        if licznik == 20:
            break
    
    tytuly = []        
    for movie in rekomendacje:  # rekomendacje ma same id filmu
        filt = (movies['movieId'] == movie)
        filt_df = movies.loc[filt]  # filt_df bedize seriesem !!!
        for index, row in filt_df.iterrows():
            tytuly.append(row['title'])

    for movie in tytuly:      
        if (movie not in same_tytuly):
            results.append(movie)