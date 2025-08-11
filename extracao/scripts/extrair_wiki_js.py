#!/usr/bin/env python3
"""
Script para extrair todas as páginas publicadas do Wiki.js e salvar em Parquet.

API: https://docs.requarks.io/dev/api
Wiki: https://wiki.antaq.gov.br/
"""

import os
from pathlib import Path
from extracao.wiki_js.scraper import WikiJSScraper
from extracao.wiki_js.config import DATA_OUTPUT_PATH, WIKI_JS_API_TOKEN


def main():
    print("🚀 EXTRAÇÃO WIKI.JS - PÁGINAS PUBLICADAS")
    if not WIKI_JS_API_TOKEN:
        print("❌ WIKI_JS_API_TOKEN não definido. Configure a variável de ambiente.")
        return 1

    scraper = WikiJSScraper()
    output_path = scraper.run(output_path=Path(DATA_OUTPUT_PATH))

    print(f"✅ Extração concluída. Arquivo salvo em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


