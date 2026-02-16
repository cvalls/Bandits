# Ruta base del proyecto
$basePath  = "C:\cursoEdx\bandit_project"

# Directorio a excluir (scripts)
$excludeDir = Join-Path $basePath "scripts"

# Fichero de salida (se mantiene lo que ya tenga)
$outputFile = Join-Path $basePath "all_code_concatenated.txt"

# Obtener todos los .py en subdirectorios, excluyendo /scripts y /.ipynb_checkpoints
$files = Get-ChildItem -Path "C:\cursoEdx\bandit_project" -Recurse -File |
    Where-Object { $_.Extension -eq ".py" } |
    Where-Object {
        $_.FullName -notlike "*\scripts\*" -and
        $_.FullName -notlike "*\.ipynb_checkpoints\*" -and
        $_.FullName -notlike "*\__pycache__\*"
    }    



Write-Host "Encontrados $($files.Count) ficheros .py válidos"

foreach ($file in $files) {
    Add-Content -Path $outputFile -Value "`n`n# ================================================"
    Add-Content -Path $outputFile -Value ("# FILE: " + $file.FullName)
    Add-Content -Path $outputFile -Value "# ================================================`n"

    Get-Content $file.FullName | Add-Content -Path $outputFile
}


