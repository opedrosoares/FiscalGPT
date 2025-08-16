#!/usr/bin/env python3
"""
Script de teste para demonstrar a funcionalidade FAISS localmente
sem precisar da API OpenAI
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Adicionar o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent))

def create_mock_embeddings(text, dimension=1536):
    """Cria embeddings simulados baseados no hash do texto"""
    import hashlib
    
    # Gerar hash do texto
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Converter hash para números
    seed = int(text_hash[:8], 16)
    np.random.seed(seed)
    
    # Gerar embedding simulado
    embedding = np.random.rand(dimension).astype(np.float32)
    
    # Normalizar para similaridade cosseno
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.tolist()

def test_faiss_without_openai():
    """Testa o sistema FAISS sem API OpenAI"""
    
    print("🚀 Testando sistema FAISS sem API OpenAI...")
    
    try:
        # Importar nossa implementação FAISS
        from chatbot.core.vector_store_faiss import VectorStoreANTAQ
        
        # Criar instância com chave fake
        vs = VectorStoreANTAQ("fake-key-for-testing")
        
        print(f"✅ VectorStore FAISS criado em: {vs.persist_directory}")
        print(f"📊 Índice inicial: {vs.index.ntotal} vetores")
        
        # Carregar dados do Wiki.js
        wiki_file = "shared/data/wiki_js_paginas.parquet"
        if Path(wiki_file).exists():
            print(f"\n📖 Carregando dados de: {wiki_file}")
            
            # Ler dados
            df = pd.read_parquet(wiki_file)
            print(f"   Total de registros: {len(df)}")
            print(f"   Colunas: {df.columns.tolist()}")
            
            # Processar apenas 3 registros para teste
            sample_df = df.head(3).copy()
            
            print(f"\n🔧 Processando {len(sample_df)} registros de teste...")
            
            # Substituir o método de geração de embeddings
            original_generate_embedding = vs._generate_embedding
            
            def mock_generate_embedding(text):
                """Método mock para gerar embeddings sem API OpenAI"""
                return create_mock_embeddings(text)
            
            vs._generate_embedding = mock_generate_embedding
            
            # Processar dados
            success = vs.load_and_process_data(
                wiki_file, 
                sample_size=3, 
                incremental=False,
                force_rebuild=True
            )
            
            if success:
                print(f"\n✅ Vetorização concluída com sucesso!")
                print(f"📊 Total de vetores no índice: {vs.index.ntotal}")
                
                # Testar busca
                print(f"\n🔍 Testando busca semântica...")
                results = vs.search("fiscalização afretamento", n_results=2)
                
                if results:
                    print(f"   Resultados encontrados: {len(results)}")
                    for i, result in enumerate(results, 1):
                        print(f"   {i}. Similaridade: {result['similarity']:.3f}")
                        print(f"      Título: {result['metadata'].get('titulo', 'N/A')}")
                else:
                    print("   Nenhum resultado encontrado")
                
                # Estatísticas da coleção
                stats = vs.get_collection_stats()
                print(f"\n📊 Estatísticas da coleção:")
                print(f"   Total chunks: {stats.get('total_chunks', 0)}")
                print(f"   Normas únicas: {stats.get('total_normas_unicas', 0)}")
                
            else:
                print("❌ Erro na vetorização")
                
        else:
            print(f"❌ Arquivo não encontrado: {wiki_file}")
            
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_faiss_without_openai()
