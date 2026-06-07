$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv311\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    py -3.11 -m venv (Join-Path $root ".venv311")
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $venvPython -m pip install pandas scikit-learn pyarrow

# Optional PyG wheels. These can still be blocked by Windows application control,
# but they are useful when the local machine allows the native extensions.
& $venvPython -m pip install torch_geometric
& $venvPython -m pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
& $venvPython -m pip install torch-geometric-temporal --no-deps
