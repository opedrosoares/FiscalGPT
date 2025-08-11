#!/usr/bin/env python3
"""
Cliente GraphQL para Wiki.js

Referência da API: https://docs.requarks.io/dev/api
"""

from typing import Any, Dict, List, Optional, Tuple
import requests
import logging


logger = logging.getLogger(__name__)


class WikiJSClient:
    """
    Cliente mínimo para chamadas GraphQL ao Wiki.js.
    """

    def __init__(self, base_url: str, api_token: str, timeout: int = 30):
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self.base_url: str = base_url
        self.api_token: str = api_token
        self.timeout: int = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    @property
    def graphql_endpoint(self) -> str:
        # Conforme docs: endpoint GraphQL é /graphql
        # Ref: https://docs.requarks.io/dev/api
        return f"{self.base_url}/graphql"

    def execute(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executa uma query/mutation GraphQL genérica.

        Args:
            query: string GraphQL
            variables: variáveis (opcional)

        Returns:
            payload JSON com data ou levanta exceção em erro
        """
        try:
            payload = {"query": query}
            if variables:
                payload["variables"] = variables

            resp = self.session.post(self.graphql_endpoint, json=payload, timeout=self.timeout)
            # Tenta parsear JSON mesmo em 400 para capturar erros de GraphQL
            data: Dict[str, Any] = {}
            try:
                data = resp.json()
            except Exception:
                pass
            if not resp.ok:
                # Inclui corpo para facilitar diagnóstico
                try:
                    msg = data.get("errors", [{}])[0].get("message", resp.text)
                except Exception:
                    msg = resp.text
                logger.error(f"Erro GraphQL HTTP {resp.status_code}: {msg}")
                resp.raise_for_status()

            if "errors" in data:
                # Loga primeiro erro para facilitar debug
                first_error = data["errors"][0]
                message = first_error.get("message", "Erro GraphQL")
                logger.warning(f"Erro GraphQL: {message}")
                raise RuntimeError(message)

            return data.get("data", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro de rede ao executar GraphQL: {e}")
            raise

    # Queries GraphQL
    PAGES_LIST_QUERY: str = (
        """
        {
          pages {
            list(orderBy: TITLE) {
              id
              path
              title
              createdAt
              updatedAt
            }
          }
        }
        """
    )

    PAGE_SINGLE_BOTH_QUERY: str = (
        """
        query PageSingle($id: Int!) {
          pages {
            single(id: $id) {
              id
              path
              title
              createdAt
              updatedAt
              render
              content
            }
          }
        }
        """
    )

    def list_pages(self, order_by: str = "TITLE") -> List[Dict[str, Any]]:
        """
        Retorna lista de páginas (metadados básicos).
        """
        # Usa query direta sem variables para evitar problemas de enum
        data = self.execute(self.PAGES_LIST_QUERY)
        return data.get("pages", {}).get("list", [])

    def get_page_content(self, page_id: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Retorna conteúdo da página (html, markdown). Tenta múltiplos esquemas.

        Returns:
            (html, markdown)
        """
        html: Optional[str] = None
        markdown: Optional[str] = None

        # Query unificada com campos simples (Strings)
        try:
            data = self.execute(self.PAGE_SINGLE_BOTH_QUERY, variables={"id": int(page_id)})
            single = data.get("pages", {}).get("single", {})
            if single:
                # Em muitos schemas: render = HTML, content = Markdown (ou vice-versa)
                html = single.get("render")
                markdown = single.get("content")
        except Exception as e:
            logger.debug(f"Falha ao buscar render/content simples para id={page_id}: {e}")

        return html, markdown


