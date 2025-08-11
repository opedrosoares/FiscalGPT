#!/usr/bin/env python3
"""
Gera um arquivo .parquet a partir dos TXT em shared/imports

Estrutura esperada de pastas:
- shared/imports/
  - dicionarios/
  - leis_decretos/
  - manuais/
  - processos_sancionadores/

Cada .txt vira um registro com colunas compatíveis com o pipeline de vetorização
('wiki_js like'): 'id', 'conteudo', 'title', 'path', 'tipo_material', 'categoria', etc.

Uso:
  python -m chatbot.scripts.gerar_parquet_imports \
    --imports-dir /abs/path/to/shared/imports \
    --output /abs/path/to/shared/data/imports_latest.parquet
"""

import argparse
import os
from pathlib import Path
import hashlib
from datetime import datetime
import pandas as pd


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback para arquivos com encoding irregular
        return path.read_text(encoding="latin-1", errors="replace")


def compute_deterministic_id(category: str, file_path: Path) -> str:
    base = f"{category}::{file_path.as_posix()}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def collect_records(imports_dir: Path) -> pd.DataFrame:
    categories = [
        "dicionarios",
        "leis_decretos",
        "manuais",
        "processos_sancionadores",
    ]

    records = []
    for category in categories:
        cat_dir = imports_dir / category
        if not cat_dir.exists():
            continue

        for txt_path in cat_dir.rglob("*.txt"):
            try:
                content = read_text_file(txt_path).strip()
                if not content:
                    continue

                rec_id = compute_deterministic_id(category, txt_path)
                title = txt_path.stem.replace("_", " ").strip()

                records.append(
                    {
                        "id": rec_id,
                        "conteudo": content,
                        "title": title,
                        "path": f"{category}/{txt_path.name}",
                        "categoria": category,
                        # Campo utilizado pelo pipeline como metadado
                        "tipo_material": f"Imports::{category}",
                        # Campos auxiliares para compatibilidade, embora não usados no modo wiki_js
                        "vetorizado": False,
                        "ultima_verificacao_vetorizacao": pd.NaT,
                    }
                )
            except Exception as e:
                print(f"⚠️ Falha ao processar {txt_path}: {e}")
                continue

    if not records:
        return pd.DataFrame(
            columns=[
                "id",
                "conteudo",
                "title",
                "path",
                "categoria",
                "tipo_material",
                "vetorizado",
                "ultima_verificacao_vetorizacao",
            ]
        )

    return pd.DataFrame.from_records(records)


def main():
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Gerar Parquet a partir de shared/imports")
    parser.add_argument(
        "--imports-dir",
        type=str,
        default=str(project_root / "shared" / "imports"),
        help="Diretório base dos TXT (shared/imports)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "shared" / "data" / "imports_latest.parquet"),
        help="Caminho do arquivo .parquet de saída",
    )
    args = parser.parse_args()

    imports_dir = Path(args.imports_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not imports_dir.exists():
        print(f"❌ Diretório não encontrado: {imports_dir}")
        return 1

    print(f"📥 Lendo TXT em: {imports_dir}")
    df = collect_records(imports_dir)

    if df.empty:
        print("⚠️ Nenhum conteúdo encontrado em shared/imports")
        return 0

    # Também produz uma versão com timestamp para histórico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_output = output_path.with_name(f"imports_{timestamp}.parquet")

    print(f"💾 Salvando: {versioned_output}")
    df.to_parquet(versioned_output, index=False)

    print(f"🔗 Atualizando link estável: {output_path}")
    df.to_parquet(output_path, index=False)

    print(f"✅ Gerado com {len(df)} registros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


