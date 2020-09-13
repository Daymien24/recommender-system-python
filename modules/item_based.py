
rekomendacje = []
# Metoda - Item_item Cos_similarity !!!!
    # Co potrzebujemy ? ratings oraz tytuly z idsami
def recom_i(favs_dict, same_tytuly):
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity


    ratings = pd.read_csv('data02/ratings.csv')
    titles = pd.read_csv('data02/title_id.csv')  # Dalem w indexie to, zeby szybciej nam dzialal algorytm :)

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
    
    for i in range(len(lista)):
        if(lista.index[i] not in same_tytuly):  # zeby nie wysweitlalo filmow juz ocenionych !!!!!
            rekomendacje.append(lista.index[i])
        if i == 25:
            break