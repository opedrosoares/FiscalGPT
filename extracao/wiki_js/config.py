#!/usr/bin/env python3
"""
Configurações do scraper Wiki.js
"""

import os
from pathlib import Path

# URL base do Wiki.js
WIKI_JS_URL: str = os.getenv("WIKI_JS_URL", "https://wiki.antaq.gov.br")

# Token de API (Bearer) do Wiki.js
# Preferencialmente use a variável de ambiente WIKI_JS_API_TOKEN.
# Se não estiver definida, usa o valor fornecido pelo usuário.
WIKI_JS_API_TOKEN: str = os.getenv("WIKI_JS_API_TOKEN")

# Caminho padrão para salvar o Parquet de páginas do Wiki.js
DATA_OUTPUT_PATH: Path = Path("shared/data/wiki_js_paginas.parquet")

# Tamanho do lote para chamadas subsequentes, caso necessário (reserva)
PAGE_BATCH_SIZE: int = int(os.getenv("WIKI_JS_PAGE_BATCH_SIZE", "200"))

# Timeout requests
REQUEST_TIMEOUT: int = int(os.getenv("WIKI_JS_REQUEST_TIMEOUT", "30"))


