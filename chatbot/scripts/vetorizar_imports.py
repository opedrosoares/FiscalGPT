#!/usr/bin/env python3
"""
Vetoriza o Parquet gerado a partir de shared/imports em uma nova coleção ChromaDB.

Uso típico:
  python -m chatbot.scripts.vetorizar_imports \
    --parquet /abs/path/to/shared/data/imports_latest.parquet \
    --collection imports_antaq
"""

import argparse
import os
from pathlib import Path


def main():
    # Resolver raiz do projeto para imports estáveis
    project_root = Path(__file__).resolve().parents[2]
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from chatbot.config.config import OPENAI_API_KEY, CHROMA_PERSIST_DIRECTORY
    from chatbot.core.vector_store import VectorStoreANTAQ

    parser = argparse.ArgumentParser(description="Vetorização dos imports em nova coleção")
    parser.add_argument(
        "--parquet",
        type=str,
        default=str(project_root / "shared" / "data" / "imports_latest.parquet"),
        help="Caminho do arquivo Parquet de entrada",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="imports_antaq",
        help="Nome da coleção Chroma a criar/usar",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Se definido, vetoriza apenas uma amostra de N registros (teste)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Se setado, apaga e recria a coleção",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"❌ Parquet não encontrado: {parquet_path}")
        return 1

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY não configurada em chatbot/config/config.py ou .env")
        return 1

    print("🚀 Inicializando VectorStore...")
    vs = VectorStoreANTAQ(
        openai_api_key=OPENAI_API_KEY,
        persist_directory=str(CHROMA_PERSIST_DIRECTORY),
        collection_name=None,  # forçamos escolha adiante
    )

    print(f"🔧 Coleção alvo: {args.collection}")
    print(f"📦 Parquet: {parquet_path}")

    # Usamos o modo wiki_js dentro do pipeline pois o Parquet tem 'conteudo' e não 'conteudo_pdf'
    success = vs.load_and_process_data(
        parquet_path=str(parquet_path),
        force_rebuild=args.force_rebuild,
        sample_size=args.sample_size,
        incremental=False,
        collection_name=args.collection,
    )

    if not success:
        print("❌ Erro durante a vetorização dos imports")
        return 1

    print("✅ Vetorização concluída!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


