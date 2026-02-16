#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# GESTION DE LOS ARGUMENTOS DEL SCRIPT
from types import SimpleNamespace
import argparse


# In[ ]:


# EXISTEN DOS FUNCIONES. ESTA ESTABLECE LOS PARAMETROS FIJOS EN UNA ESTRUCTURA QUE EL SCRIPT USA.
def recibeParametros():
    return SimpleNamespace(
        slate=5,
        epsilon=0.15,
        batch_size=1000,
        min_review_count=10000,
        balanced_classes=True,
        result_dir="C:\\cursoEdx\\ml-25m\\",
        verbose='TRUE'
    )
           


# In[ ]:


# EXISTEN DOS FUNCIONES. ESTA PERMITE LLAMAR AL SCRIPT COMO UN SCRIPT DE PYTHON CON PARAMETROS
# command line args for experiment params
# example: python3 epsilon_greedy.py --n=5 --epsilon=0.15 --batch_size=1000 --min_review_count=1500
def recibeParametros2():
    parser = argparse.ArgumentParser()

    parser.add_argument('--slate', help="slate size (number of recs per iteration)", type=int, default=20)
    parser.add_argument('--epsilon', help="epsilon for epsilon-greedy", type=float, default=0.15)
    parser.add_argument('--batch_size', help="batch size per iteration", type=int, default=10)
    parser.add_argument('--min_review_count', help="minimum reviews per movie", type=int, default=1500)
    parser.add_argument('--balanced_classes', help="balance dataset classes", type=bool, default=True)
    parser.add_argument('--result_dir', help="directory for results", type=str, default='/Users/jamesledoux/Documents/bandits/results/')
    parser.add_argument('--verbose', help="verbosity flag", type=str, default='TRUE')

    # Detecta si estás en Jupyter o en un script .py
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.argv[0]:
        args = parser.parse_args([])   # Jupyter
    else:
        args = parser.parse_args()     # Script .py

    return SimpleNamespace(
        slate=args.slate,
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        min_review_count=args.min_review_count,
        balanced_classes=args.balanced_classes,
        result_dir=args.result_dir,
        verbose=args.verbose
    )

