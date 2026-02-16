    #!/usr/bin/env python
    # coding: utf-8
    
    # In[2]:
    
    
    # Este fichero contiene las funciones para crear el dataframe que se va a usar como entrada de datos. 
    # en el entreno. 
    # se graba el notebook ocomo ficghero Python para poder importarlo desde otro modulo.
    
    
    # In[3]:
    
import pandas as pd
import numpy as np 
import os
    
    
    # In[4]:


def read_movielens_25m(sin_genero):
    # se leen los 4 ficheros de datos, engine python permite leer ficheros con 
    # problemas en su estructura, caractares no soportados, separadores mal puestos...
    ratings = pd.read_csv(r'C:\cursoEdx\ml-25m\ratings.csv', engine='python')
    movies = pd.read_csv(r'C:\cursoEdx\ml-25m\movies.csv', engine='python')
    #links = pd.read_csv(r'C:\cursoEdx\ml-25m\links.csv', engine='python')
    #tags = pd.read_csv(r'C:\cursoEdx\ml-25m\tags.csv', engine='python')
    
    # columna de generos es de tipo "Action|Adventure|Sci-Fi" por jemplo.
    # str.get_dummies() convierte lo leido en una tabla con tantas columnas como
    # generos y marca 1 donde el genero corresponda. es decir hace un hot encoding con 0 y 1.
    # y astype convierte el 1 en true y e l 0 en false . al final el join aÃ±ade
    # las columnas de genero al dataframe. No borra nada
    if ( sin_genero == True ) :
        print("sin genero")
        movies = movies.join(movies.genres.str.get_dummies().astype(bool))
    
    # se elimina genres, que ya no vale.
    movies.drop('genres', inplace=True, axis=1)

    # se aÃ±ade a movies el ratings por la izquierda por coincidencia con el movieid
    # si hay columnas coincidentes en los dos df metre _movie a la columna del segundo.
    df = ratings.merge(movies, on='movieId', how='left')  # â clave aquÃ­
    
    # esta es la instruccion en el original, pero hace el join por index
    # EstÃ¡ cruzando ratings.movieId con movies.index, y luego aÃ±adiendo 
    # la columna movies.movieId renombrada a movieId_movie, que puede ser cualquier cosa.
    # movies = movies.set_index('movieId')
    # df = ratings.join(movies, on='movieId', how='left', rsuffix='_movie')
    return df


# In[5]:

def preprocess_movielens_25m(df, min_number_of_reviews, tipo_de_recompensa, limite_rating, balanced_classes):
    # remove ratings of movies with < N ratings. too few ratings will 
    # cause the recsys to get stuck in offline evaluation
    # se eliminan las pelis con poco rating. estamos montando un conjunto de datos para 
    # evaluar un sistema de recomendacion. todo lo que este por debajo del numero de reviews fuera.
    # evalua un df con el numero de veces que sale cada peli id 
    
    # se crea una serie cuyo indice es el movieid y valor el numero de veces que sale en el df.
    # cada elemento de la serie es un id de peli y el numero de apariciones (visualizaciones)
    counts = df.movieId.value_counts()

    # se extare de la serie los elementos que tiene mas de un minimo de veces
    # popular movies es una serie booleana. es decir mantiene el id de la peli y un true false egun cumpla condicion
    popular_movies = counts >= min_number_of_reviews

    # movies to keep es un indice con los indices de los elementos true
    movies_to_keep = counts[popular_movies].index
    
    # se busca en el df indices de las pelis a guardar 
    df = df.loc[ df['movieId'].isin(movies_to_keep) ]

    """
    if balanced_classes is True:
        df = df.groupby('movieId')
        df = df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True))
    """
    
    # el dataset viene ordenado por userid, por tiemo o por lo que sea.
    # para evitar sesgos, se baraja de forma que hay un conjunto sin bias.
    df = df.sample(frac=1)
    
    # Se aÃ±ade una columna t que representa cada timestep del bandit para 
    # simluar un escenario en vivoo
    df['t'] = np.arange(len(df))
    df.index = df['t']
    
    if tipo_de_recompensa == "Categorica" :
        df["recompensa"] = df["rating"]
        
        # la recompensa debe ser una categoria en este caso
        mapping = { 0.5: "A", 1.0: "B", 1.5: "C", 2.0 : "D",
                         2.5: "E", 3.0: "F", 3.5: "G",
                         4.0: "H", 4.5: "I", 5.0: "J"}
        
        df["recompensa_cat"] = df["rating"].replace(mapping)
        
    # Si se quieren recompensas binarias o continuas se devuelven en el mismo campo.
    # si son continuas basta con cambiar de nombre a la columna rating
    if tipo_de_recompensa == "Continua" :
        df["recompensa"] = df["rating"]
        df["recompensa_cat"] = df["recompensa"] 
        
    # si es binaria se crea una columna nueva. Binario significa si gusta o no.
    # para eso esta el limite_rating
    if tipo_de_recompensa == "Bernouilli" :
        # se aÃ±ade una columna recompensa que vale 1 solo si rating es > 4.5
        df['recompensa'] = df['rating'].apply(lambda x: 1 if x >= limite_rating else 0)
        df["recompensa_cat"] = df["recompensa"]

    df.drop(columns=["rating"], inplace=True)
    return df


# In[6]:


# punto de entrada desde el programa principal.
# devuelve el df de peliculas evaluadas por otr sistema con los campos necesario y los generos como hot encoding
def get_ratings_25m(min_number_of_reviews, balanced_classes,
                    tipo_de_recompensa, cargar_generos, limite_rating):
	logs = read_movielens_25m(cargar_generos)
	logs = preprocess_movielens_25m(logs, min_number_of_reviews, tipo_de_recompensa, limite_rating, balanced_classes)
	return logs


# In[7]:


# carga el fichero de peliculas elegidas si existe en un dataframe o lo hace desde el set de datos.
def cargarPeliculasElegidas(dataset)-> pd.DataFrame :

    if ( dataset["tipo_de_recompensa"] == "Continua"  ) :
        path_completo = dataset["fichero_datos_continua"]
            
    elif ( dataset["tipo_de_recompensa"] == "Bernouilli"  ) :
        path_completo = dataset["fichero_datos_bernouilli"]

    elif ( dataset["tipo_de_recompensa"] == "Categorica"  ) :
        path_completo = dataset["fichero_datos_categorica"]

    else :
        raise ValueError("tipo_de_recompensa desconocido (Continua o Bernouilli)")    

    if os.path.exists(path_completo):
        print(f"Cargando {path_completo}")
        df = pd.read_csv(path_completo)
    else:
        print(f"No existe {path_completo}. Generando dataset 25M...")
        df = get_ratings_25m(dataset["min_number_of_reviews"],
                             dataset["balanceo_clases"], 
                             dataset["tipo_de_recompensa"], 
                             dataset["cargar_generos"],
                             dataset["limite_rating"] )
        df.to_csv(path_completo,index=False, header=True)
    return df


# In[8]:


def read_data_1m():
	print('reading movielens 1m data')
	ratings = pd.read_csv('../data/ml-1m/ratings.dat', 
		sep='::',
		names=[
			'userId',
			'movieId',
			'rating',
			'ts'
		],
		engine='python')
	movies = pd.read_csv('../data/ml-1m/movies.dat', 
		sep='::',
		names=[
			'movieId',
			'title',
			'genres'
		],
		engine='python')
	users = pd.read_csv('../data/ml-1m/users.dat', 
		sep='::', 
		names = [
			'userId',
			'gender',
			'age',
			'occupation',
			'zip'
		],
		engine='python')
	logs = ratings.join(movies, on='movieId', how='left', rsuffix='_movie')
	logs = logs.join(users, on='userId', how='left', rsuffix='_movie')
	return logs

def process_title():
	pass

def process_genres():
	pass

def preprocess_movie_data_1m(logs, min_number_of_reviews=1000):
	print('preparing ratings log')
	# remove ratings of movies with < N ratings. too few ratings will cause the recsys to get stuck in offline evaluation
	movies_to_keep = pd.DataFrame(logs.movieId.value_counts())\
		.loc[pd.DataFrame(logs.movieId.value_counts())['movieId']>=min_number_of_reviews].index
	logs = logs.loc[logs['movieId'].isin(movies_to_keep)]
	# shuffle rows to deibas order of user ids
	logs = logs.sample(frac=1)
	# create a 't' column to represent time steps for the bandit to simulate a live learning scenario
	logs['t'] = np.arange(len(logs))
	logs.index = logs['t']
	logs['liked'] = logs['liked'].apply(lambda x: 1 if x >= 4.5 else 0)
	return logs

def get_ratings_20m(min_number_of_reviews=20000, balanced_classes=False):
	logs = read_movielens_25m()
	logs = preprocess_movielens_25m(logs, min_number_of_reviews=20000, balanced_classes=balanced_classes)
	return logs

def get_ratings_1m(min_number_of_reviews=1000):
	logs = read_data_1m()
	logs = preprocess_movie_data_1m(logs, min_number_of_reviews=min_number_of_reviews)
	return logs

def __init__():
	pass


# In[ ]:





# In[ ]:





# In[ ]:




