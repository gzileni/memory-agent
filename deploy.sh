#!/bin/bash

echo ">>> Rimozione della cartella dist/"
rm -rf dist/

echo ">>> Inizio del deploy su PyPI"
python -m build

echo ">>> Upload su PyPI"
# Crea il branch se non esiste
twine upload dist/* --verbose