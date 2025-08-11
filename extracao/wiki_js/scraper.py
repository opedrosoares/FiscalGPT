#!/usr/bin/env python3
"""
Scraper para coletar todas as páginas publicadas no Wiki.js e salvar em Parquet.

Referência da API: https://docs.requarks.io/dev/api
Wiki alvo: https://wiki.antaq.gov.br/
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import pandas as pd
from pathlib import Path
import os
from bs4 import BeautifulSoup
import html as html_lib
import re
try:
    from markdownify import markdownify as md_convert
    MARKDOWNIFY_AVAILABLE = True
except Exception:
    MARKDOWNIFY_AVAILABLE = False

from .client import WikiJSClient
from .config import WIKI_JS_URL, WIKI_JS_API_TOKEN, DATA_OUTPUT_PATH, REQUEST_TIMEOUT


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class WikiPageRecord:
    id: int
    path: str
    title: str
    createdAt: Optional[str]
    updatedAt: Optional[str]
    content_html: Optional[str]
    content_markdown: Optional[str]
    conteudo: Optional[str] = None
    source: str = "wiki_js"


class WikiJSScraper:
    """
    Orquestra a coleta de todas as páginas publicadas no Wiki.js.
    """

    def __init__(self, base_url: str = WIKI_JS_URL, api_token: str = WIKI_JS_API_TOKEN, timeout: int = REQUEST_TIMEOUT):
        self.client = WikiJSClient(base_url=base_url, api_token=api_token, timeout=timeout)

    def fetch_all_pages(self) -> List[WikiPageRecord]:
        """
        Busca lista de páginas e, para cada uma, coleta conteúdo (html/markdown).
        """
        logger.info("Buscando lista de páginas do Wiki.js...")
        pages = self.client.list_pages(order_by="TITLE")
        logger.info(f"{len(pages)} páginas encontradas na listagem inicial")

        records: List[WikiPageRecord] = []

        for idx, page in enumerate(pages, 1):
            try:
                page_id = int(page.get("id"))
                path = page.get("path", "")
                title = page.get("title", "")
                createdAt = page.get("createdAt")
                updatedAt = page.get("updatedAt")

                html, md = self.client.get_page_content(page_id)
                conteudo = ""
                # Normaliza candidatos
                def has_html_tags(text: str) -> bool:
                    return bool(re.search(r'<[^>]+>', text))

                plain_md = None
                if isinstance(md, str) and md.strip():
                    plain_md = self._html_to_text(md) if has_html_tags(md) else self._clean_text(md)

                plain_html = None
                if isinstance(html, str) and html.strip():
                    plain_html = self._html_to_text(html)

                # Preferir markdown quando disponível; se vier com tags, converter.
                if plain_md and plain_md.strip():
                    conteudo = self._normalize_markdown(plain_md)
                elif plain_html and plain_html.strip():
                    conteudo = self._html_to_markdown(html)

                record = WikiPageRecord(
                    id=page_id,
                    path=path,
                    title=title,
                    createdAt=createdAt,
                    updatedAt=updatedAt,
                    content_html=html,
                    content_markdown=md,
                    conteudo=conteudo,
                )
                records.append(record)

                if idx % 25 == 0:
                    logger.info(f"Progresso: {idx}/{len(pages)} páginas processadas")
            except Exception as e:
                logger.error(f"Erro ao processar página: {page}. Detalhes: {e}")
                continue

        logger.info(f"Coleta concluída: {len(records)} páginas processadas")
        return records

    def save_to_parquet(self, records: List[WikiPageRecord], output_path: Path = DATA_OUTPUT_PATH, merge_with_existing: bool = True) -> Path:
        """
        Salva os registros em Parquet, mesclando por id quando arquivo existir.
        """
        if not records:
            logger.warning("Nenhum registro para salvar.")
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Converter para DataFrame
        df_new = pd.DataFrame([r.__dict__ for r in records])
        df_new["id"] = df_new["id"].astype(int)
        # Remover colunas de conteúdo brutas
        for col in ["content_html", "content_markdown"]:
            if col in df_new.columns:
                df_new = df_new.drop(columns=[col])

        if merge_with_existing and output_path.exists():
            try:
                df_existing = pd.read_parquet(output_path)
                if "id" in df_existing.columns:
                    df_existing["id"] = df_existing["id"].astype(int)
                # Garantir remoção de colunas de conteúdo brutas também no existente
                for col in ["content_html", "content_markdown"]:
                    if col in df_existing.columns:
                        df_existing = df_existing.drop(columns=[col])
                df_merged = pd.concat([df_existing, df_new], ignore_index=True)
                df_merged = df_merged.drop_duplicates(subset=["id"], keep="last")
                df_merged.to_parquet(output_path, engine="pyarrow", index=False)
                logger.info(f"Arquivo atualizado com {len(df_merged)} páginas: {output_path}")
                return output_path
            except Exception as e:
                logger.warning(f"Falha ao mesclar com existente, sobrescrevendo. Detalhes: {e}")

        # Salvar direto
        df_new.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Arquivo salvo com {len(df_new)} páginas: {output_path}")
        return output_path

    def run(self, output_path: Path = DATA_OUTPUT_PATH) -> Path:
        records = self.fetch_all_pages()
        return self.save_to_parquet(records, output_path=output_path, merge_with_existing=True)

    # Helpers
    def _html_to_text(self, html: str) -> str:
        """
        Converte HTML para texto puro, preservando somente os links como "texto URL".
        """
        if not html:
            return ""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remover scripts e estilos
            for tag in soup.find_all(['script', 'style']):
                tag.decompose()

            # Preservar links: substituir <a> por 'texto URL' (ou apenas URL se não houver texto)
            for a in soup.find_all('a'):
                href = (a.get('href') or '').strip()
                link_text = a.get_text(strip=True)
                replacement = ''
                if href and link_text and link_text != href:
                    replacement = f"{link_text} {href}"
                elif href:
                    replacement = href
                elif link_text:
                    replacement = link_text
                a.replace_with(replacement)

            # Extrair texto plano
            text = soup.get_text(separator=' ', strip=True)
            text = html_lib.unescape(text)
            # Normaliza espaços
            text = ' '.join(text.split())
            return text
        except Exception:
            return html

    def _html_to_markdown(self, html: str) -> str:
        """
        Converte HTML para Markdown. Tenta usar markdownify se disponível.
        Caso contrário, faz uma conversão básica preservando links.
        """
        if not html:
            return ""
        try:
            if MARKDOWNIFY_AVAILABLE:
                md = md_convert(
                    html,
                    heading_style='ATX',
                    strip=['script', 'style'],
                    bullets='-',
                )
                return self._normalize_markdown(md)
            # Fallback simples: texto com links no formato [texto](url)
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup.find_all(['script', 'style']):
                tag.decompose()
            for a in soup.find_all('a'):
                href = (a.get('href') or '').strip()
                link_text = a.get_text(strip=True)
                if href:
                    replacement = f"[{link_text or href}]({href})"
                else:
                    replacement = link_text
                a.replace_with(replacement)
            text = soup.get_text(separator='\n', strip=True)
            text = html_lib.unescape(text)
            return self._normalize_markdown(text)
        except Exception:
            return self._normalize_markdown(self._html_to_text(html))

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Limpeza leve: normaliza espaços e remove artefatos de markup simples
        cleaned = html_lib.unescape(text)
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def _normalize_markdown(self, text: str) -> str:
        if not text:
            return ""
        # Remove espaços duplicados e normaliza quebras de linha
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Colapsa linhas vazias múltiplas
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Normaliza espaços
        text = re.sub(r'[ \t]+', ' ', text)
        # Aparar
        return text.strip()


