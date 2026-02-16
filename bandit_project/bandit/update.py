 
#este fichero queda vacio. Valdria por ejemplo para meter una funcion de actualizacion del bandit de forma que se
#pudiera injectra en la clase y asi tener varias opciones

#bastaria hacer algo como esto en el bandit.

#from bandit.update import actualizar_bandit
def actualizar(self, emparejados, t_dataset, t_bandit, history_list, sz_num_emparejados_batch):
    actualizar_bandit(
        self.brazos,
        self.dctBrazos,
        emparejados,
        t_dataset,
        t_bandit,
        history_list,
        sz_num_emparejados_batch,
        self.regret_tracker,
        self.mediasReales,
        self.mejorBrazoReal
    )
