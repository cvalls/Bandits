$root = "C:\cursoEdx\bandit_project"
$outfile = "C:\cursoEdx\bandit_project\proyecto_concatenado.txt"

# Crear o vaciar el fichero de salida
"" | Out-File $outfile -Encoding UTF8

# Recorrer todos los directorios y subdirectorios
Get-ChildItem -Path $root -Recurse -Directory | ForEach-Object {

    $dirPath = $_.FullName

    # Escribir nombre del directorio
    "==========================================" | Out-File $outfile -Append -Encoding UTF8
    "DIRECTORIO: $dirPath" | Out-File $outfile -Append -Encoding UTF8
    "==========================================" | Out-File $outfile -Append -Encoding UTF8
    "" | Out-File $outfile -Append -Encoding UTF8

    # Recorrer los ficheros dentro del directorio
    Get-ChildItem -Path $dirPath -File | ForEach-Object {
        $filePath = $_.FullName

        # Escribir nombre del fichero
        "----- FICHERO: $filePath" | Out-File $outfile -Append -Encoding UTF8
        "" | Out-File $outfile -Append -Encoding UTF8

        # Escribir contenido del fichero
        Get-Content $filePath | Out-File $outfile -Append -Encoding UTF8

        # Separador entre ficheros
        "`n" | Out-File $outfile -Append -Encoding UTF8
    }

    # Separador entre directorios
    "`n`n" | Out-File $outfile -Append -Encoding UTF8
}

Write-Host "Concatenación completada. Archivo generado en: $outfile"
