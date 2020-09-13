from flask import Flask, render_template, request, redirect, url_for, flash
from flask import jsonify
from flask_sqlalchemy import SQLAlchemy
from filmy import movies_list_4000, titles4000, ranking
import pandas as pd
app = Flask(__name__)

ENV = 'dev'

if ENV == 'dev':
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost/filmownia'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class MyList(db.Model):
    __tablename__ = 'my_list'
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(50)) 
    title = db.Column(db.String(100))

    def __init__(self,name, title):        
        self.user = name
        self.title = title
        

    def __repr__(self):
        return f"MyList('{self.user}', '{self.title}')"



class Rating(db.Model):
    __tablename__ = 'rated_movies'
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(50)) 
    title = db.Column(db.String(100))
    movieId = db.Column(db.Integer)    
    rating = db.Column(db.Integer)
    

    def __init__(self,name, title,movieId, rating):        
        self.user = name
        self.title = title
        self.movieId = movieId
        self.rating = rating

    def __repr__(self):
        return f"Rating('{self.user}', '{self.title}', '{self.movieId}', '{self.rating}')"
        

name = ""

favs_dict = {}  # oceny nowego usera do rekomendacji
mylist = []  # moja lista (ulubione filmy)
movies = movies_list_4000  # do wyswietlania listy filmow ( type - DataFrame)
tytuly = titles4000  # do search bara tytulki (type - Series)
same_tytuly = []  # do item-item cos, zeby nie pokazywac w rekom ocenionych filmow !!!!!!!!
ranking = ranking  # do tabelki ranking naszej

@app.route('/', methods=['GET', 'POST'])
def index():
    
    return render_template('index.html')
    
@app.route('/insert_user', methods=['POST'])
def insert_user():
    if request.method =='POST':
        global name
        name = request.form['name']
        return redirect(url_for('start'))

@app.route('/start', methods=['GET'])
def start():
    # do rekomendacji item- item !!!
    print(name)    
    return render_template('start.html', movies = tytuly, user = name)



@app.route('/titles', methods=['GET'])
def titles():  
    return render_template('titles.html', movies= ranking)

@app.route('/favs', methods=['GET'])
def favs():
    global name
    data = Rating.query.filter_by(user=name).all()
    

    return render_template('favs.html', movies = data, user = name)

@app.route('/mylist_page', methods=['GET'])
def mylist_page():
    global name
    data = MyList.query.filter_by(user=name).all()      

    return render_template('mylist_page.html', movies = data, user = name)

movie2rating = {}
filmy_ocenione=[]  # ID FILMOW
oceny=[]


@app.route('/insert', methods=['POST'])
def insert():
    global name
    if request.method == 'POST':
        title = request.form['title']
        ocena = request.form['star']
        favs_dict[title] = int(ocena)
        filt = (movies['title'] == title)
        id_filmu = movies.loc[filt]  # tutaj dosnatniemy series z id i tytulem
        id_filmu = id_filmu.values[0][0]  # tu nam daje id filmu 
        filmy_ocenione.append(id_filmu)
        movie2rating[id_filmu] = int(ocena)
        oceny.append(int(ocena))       
        same_tytuly.append(title)  # robimy to teraz w bazie

        # dodajemy teraz do database
        data = Rating(name, title, id_filmu, ocena)
        db.session.add(data)
        db.session.commit()
        
        return redirect(url_for('start'))

# @app.route('/edit', methods=['GET', 'POST'])
# def edit():
#     if request.method == 'POST':
#         favs_dict[title] = request.form['rating']
        
#         flash(f'The rating has been edited!', 'success')

#         return redirect(url_for('favs'))


@app.route('/add_to_mylist', methods=['GET','POST'])
def add_to_mylist():
    if request.method == 'POST':
        title = request.form['title']
        # if title not in mylist:
        #     mylist.append(title)
        data = MyList(name, title)
        data2 = MyList.query.filter_by(user=name).all()
        dodane = []
        for row in data2:
            dodane.append(row.title)
        if title not in dodane:    
            db.session.add(data)
            db.session.commit()

        return redirect(url_for('recom_mf'))

@app.route('/delete/<id>', methods=['GET', 'POST'])
def delete(id):
    data = Rating.query.get(id)
    db.session.delete(data)
    db.session.commit()    
    # flash(f'The movie has been deleted!', 'success')

    return redirect(url_for('favs'))

@app.route('/delete_mylist/<id>', methods=['GET', 'POST'])
def delete_mylist(id):
    data = MyList.query.get(id)
    db.session.delete(data)
    db.session.commit()    
    # flash(f'The movie has been deleted!', 'success')

    return redirect(url_for('mylist_page'))


    ### REKOMENDACJE ###



@app.route('/recom_mf', methods=['GET', 'POST'])
def recom_mf():
    import numpy as np
    data = Rating.query.filter_by(user=name).all()
    oceny = []
    titles_rated = []
    for row in data:
        oceny.append(int(row.rating))
        titles_rated.append(int(row.movieId))

    moviess = np.array(titles_rated)
    ratings = np.array(oceny)
    print(ratings)
    

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
        if licznik == 35:
            break
    results = []
    tytuly = []        
    for movie in rekomendacje:  # rekomendacje ma same id filmu
        filt = (movies['movieId'] == movie)
        filt_df = movies.loc[filt]  # filt_df bedize seriesem !!!
        for index, row in filt_df.iterrows():
            tytuly.append(row['title'])

    for movie in tytuly:      
        if (movie not in titles_rated):
            results.append(movie)

    return render_template('recom_mf.html', rekomendacje=results)




@app.route('/recom', methods=['GET', 'POST'])
def recom():
    # caly skrypt user_user
    # co potrzebujemy ? dicty bez testowego oraz user2title csv
    import pickle
    import numpy as np
    import pandas as pd
    # from sklearn.utils import shuffle
    from sortedcontainers import SortedList

    # loadujemy date

    with open('data/json/user2movie.json', 'rb') as f:
        user2movie = pickle.load(f)  # jaki film oceniony przez jakeigo usera movie2user[3100]
        
    with open('data/json/movie2user.json', 'rb') as f:
        movie2user = pickle.load(f) # jaki user oceniil jaki film [id_filmu] 
        
    with open('data/json/usermovie2rating.json', 'rb') as f:
        usermovie2rating = pickle.load(f) # [film_id, user_id] i daje ci ocene

    # data = Rating.query.filter_by(user=name).all()
    # oceny = []
    # titles_rated = []
    # for row in data:
    #     oceny.append(int(row.rating))
    #     titles_rated.append(int(row.movieId))
    

    K = 5  # ilosc neighboursow
    limit = 1  # number of common movies users must have ( minimum )
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
    df = pd.read_csv('data/do_user_based.csv')
    df.drop(columns={'Unnamed: 0'}, inplace=True)

    # pd.set_option('display.max_row', 1000)
    filt = df['userId'] == best_match  # z tym mamy najwieksza wage
    # neighbors
    sorted_df = df[filt].sort_values(by='rating', ascending=False)
    rekomendacje = []
    for index, row in sorted_df.head(25).iterrows():
        title = row['title']
        if title not in same_tytuly:
            rekomendacje.append(row['title'])

    return render_template('recom.html', rekomendacje=rekomendacje)



@app.route('/recom_i_i', methods=['GET'])
def recom_i_i():
    # Metoda - Item_item Cos_similarity !!!!
    # Co potrzebujemy ? ratings oraz tytuly z idsami
    # import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    data = Rating.query.filter_by(user=name).all()
    tytuly = []
    oceny = []
    favs_dict = {}
    for row in data:        
        tytuly.append(row.title)
        oceny.append(int(row.rating))
        favs_dict[row.title] = int(row.rating)


    ratings = pd.read_csv('data/ratings.csv')
    titles = pd.read_csv('data/title_id.csv')  # Dalem w indexie to, zeby szybciej nam dzialal algorytm :)

    ratings.drop(columns={'Unnamed: 0',}, inplace=True)

    df2 = pd.merge(ratings, titles)
    df2.drop(columns={'Unnamed: 0'}, inplace = True)
    df2 = df2.pivot_table(index=['userId'], columns=['title'], values='rating')
    df2 = df2.fillna(0)

    def standarize(row):
        new_row = (row-row.mean())/(row.max() - row.min())
        return new_row
        
    df2 = df2.apply(standarize)

    podobne = cosine_similarity(df2.T)

    podobne_df = pd.DataFrame(podobne, index=df2.columns, columns=df2.columns)

    def rekomenduj(movie, rating):
        similar_score = podobne_df[movie]*(rating-2.5)
        similar_score = similar_score.sort_values(ascending=False)
        return similar_score

    
    similar_movies = pd.DataFrame()
    for movie, rating in favs_dict.items():
        similar_movies = similar_movies.append(rekomenduj(movie, rating), ignore_index=True)
    
    lista = similar_movies.sum().sort_values(ascending=False)  # to jest series
    
    
    rekomendacje = []

    for i in range(len(lista)):
        if(lista.index[i] not in tytuly):  # zeby nie wysweitlalo filmow juz ocenionych !!!!!
            rekomendacje.append(lista.index[i])
        if i == 25:
            break    


    return render_template('recom_i_i.html', rekomendacje=rekomendacje)


if __name__ == '__main__':
    app.run(debug=True)
    