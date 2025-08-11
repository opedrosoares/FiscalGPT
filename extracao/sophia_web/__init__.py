"""
Pacote `sophia_web`

Reexporta o scraper atual baseado no Sistema Sophia da ANTAQ para manter compatibilidade
com o código existente enquanto organiza os webscrapers por fonte.
"""

try:
    # Reexporta a classe principal do scraper atual no novo caminho
    from extracao.sophia_web.core.extrator import SophiaANTAQScraper  # noqa: F401
    from extracao.sophia_web.core.controlador import ControladorExtracao  # noqa: F401
except Exception:
    # Durante instalação/setup, o módulo pode não estar disponível
    pass


