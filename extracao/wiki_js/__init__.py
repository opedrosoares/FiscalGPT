"""
Webscraper para Wiki.js da ANTAQ

Este pacote contém um cliente GraphQL e um scraper que coletam todas as páginas
publicadas no Wiki.js e salvam o conteúdo em Parquet.
"""

from .scraper import WikiJSScraper  # noqa: F401


