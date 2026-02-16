import numpy as np
import pandas as pd 

# LAS MEDIAS REALES SON UN ENTORNO TEORICO PARA PODER HACER EL REGRET TEORICO. No son algo conocido por el bandit
# se definen y fijan para este caso.

def fijarMediasReales(df, claves_brazos) :
    true_means = np.array([
                df.loc[df.movieId == mid, 'recompensa'].mean()
                for mid in claves_brazos ])    
    return true_means



