"""
Compat layer: os módulos foram movidos para `extracao.sophia_web.core`.
Importe diretamente de `extracao.sophia_web.core` daqui em diante.
"""

try:
    from extracao.sophia_web.core.extrator import SophiaANTAQScraper, PDFExtractor  # noqa: F401
except Exception:
    pass


