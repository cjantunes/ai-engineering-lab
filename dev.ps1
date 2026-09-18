# dev.ps1
# Prepara e valida o ambiente de desenvolvimento do projeto.

$VenvPython = ".\.venv\Scripts\python.exe"

Write-Host ""
Write-Host "=== Pos IA - ChatGPT ==="
Write-Host ""

# 1. Verifica se a .venv existe
if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERRO] Ambiente virtual .venv nao encontrado."
    Write-Host "Crie-o com: python -m venv .venv"
    exit 1
}

Write-Host "[OK] Ambiente virtual encontrado."

# 2. Mostra o Python da .venv
Write-Host ""
Write-Host "Python:"
& $VenvPython --version

# 3. Verifica as dependencias
Write-Host ""
Write-Host "Verificando dependencias..."
& $VenvPython -m pip check

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERRO] Problema nas dependencias."
    exit 1
}

Write-Host "[OK] Dependencias consistentes."

# 4. Executa os testes
Write-Host ""
Write-Host "Executando testes..."
& $VenvPython -m pytest -v

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERRO] Existem testes com falha."
    exit 1
}

Write-Host ""
Write-Host "[OK] Todos os testes passaram."