$ErrorActionPreference = "Stop"

latexmk `
    -pdf `
    -interaction=nonstopmode `
    -halt-on-error `
    main.tex

Write-Host ""
Write-Host "Informe generado: main.pdf"
