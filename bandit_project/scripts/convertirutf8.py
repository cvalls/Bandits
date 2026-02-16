# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 13:45:50 2026

@author: Carlos
"""

import os

def convertir_a_utf8(root_folder):
    for folder, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(folder, file)
                try:
                    with open(path, "r", encoding="latin-1") as f:
                        content = f.read()
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print("Convertido:", path)
                except Exception as e:
                    print("ERROR:", path, e)

if __name__ == "__main__":
    convertir_a_utf8(r"C:\cursoEdx\bandit_project")
