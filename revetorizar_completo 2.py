#!/usr/bin/env python3
"""
Script para revetorização completa usando FAISS e API OpenAI real
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Adicionar o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent))

def revetorizar_completo():
    """Executa revetorização completa dos dados"""
    
    print("🚀 Iniciando revetorização completa com FAISS e OpenAI...")
    
    try:
        # Verificar API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY não encontrada no arquivo .env")
            return False
            
        print(f"🔑 API Key configurada: {api_key[:20]}...")
        
        # Importar nossa implementação FAISS
        from chatbot.core.vector_store_faiss import VectorStoreANTAQ
        
        # Criar instância com API real
        vs = VectorStoreANTAQ(api_key)
        
        print(f"✅ VectorStore FAISS inicializado em: {vs.persist_directory}")
        print(f"📊 Índice atual: {vs.index.ntotal} vetores")
        
        # 1. Revetorizar Wiki.js (forçar rebuild)
        wiki_file = "shared/data/wiki_js_paginas.parquet"
        if Path(wiki_file).exists():
            print(f"\n📖 Revetorizando Wiki.js: {wiki_file}")
            
            success = vs.load_and_process_data(
                wiki_file, 
                force_rebuild=True,  # Forçar rebuild completo
                incremental=False
            )
            
            if success:
                print(f"✅ Wiki.js revetorizado com sucesso!")
                print(f"📊 Total de vetores após Wiki.js: {vs.index.ntotal}")
            else:
                print("❌ Erro na revetorização do Wiki.js")
                return False
        else:
            print(f"⚠️ Arquivo Wiki.js não encontrado: {wiki_file}")
        
        # 2. Revetorizar normas ANTAQ (forçar rebuild)
        normas_file = "shared/data/normas_antaq_completo.parquet"
        if Path(normas_file).exists():
            print(f"\n📋 Revetorizando normas ANTAQ: {normas_file}")
            
            # Processar TODAS as normas (remover sample_size para produção)
            success = vs.load_and_process_data(
                normas_file, 
                force_rebuild=True,  # Forçar rebuild completo
                incremental=False
                # Removido sample_size=5 para processar todas as normas
            )
            
            if success:
                print(f"✅ Normas ANTAQ revetorizadas com sucesso!")
                print(f"📊 Total de vetores após normas: {vs.index.ntotal}")
            else:
                print("❌ Erro na revetorização das normas ANTAQ")
                return False
        else:
            print(f"⚠️ Arquivo de normas não encontrado: {normas_file}")
        
        # 3. Testar busca semântica
        print(f"\n🔍 Testando busca semântica...")
        test_queries = [
            "fiscalização portuária",
            "licenciamento de terminais",
            "procedimentos de fiscalização",
            "afretamento de embarcações"
        ]
        
        for query in test_queries:
            print(f"\n   🔎 Consulta: '{query}'")
            results = vs.search(query, n_results=3)
            
            if results:
                print(f"      Resultados: {len(results)}")
                for i, result in enumerate(results[:2], 1):
                    print(f"      {i}. {result['metadata'].get('titulo', 'N/A')} (sim: {result['similarity']:.3f})")
            else:
                print(f"      Nenhum resultado encontrado")
        
        # 4. Estatísticas finais
        print(f"\n📊 ESTATÍSTICAS FINAIS:")
        print(f"   Total de vetores: {vs.index.ntotal}")
        
        stats = vs.get_collection_stats()
        print(f"   Total de chunks: {stats.get('total_chunks', 0)}")
        print(f"   Normas únicas: {stats.get('total_normas_unicas', 0)}")
        
        # 5. Verificar arquivos salvos
        faiss_files = list(vs.persist_directory.glob("*.faiss"))
        metadata_files = list(vs.persist_directory.glob("*.pkl"))
        
        print(f"\n💾 Arquivos salvos:")
        for faiss_file in faiss_files:
            print(f"   📁 {faiss_file.name}")
        for metadata_file in metadata_files:
            print(f"   📄 {metadata_file.name}")
        
        print(f"\n🎉 Revetorização completa concluída com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro durante a revetorização: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = revetorizar_completo()
    if success:
        print("\n✅ Sistema FAISS funcionando perfeitamente em produção!")
    else:
        print("\n❌ Falha na revetorização")
        sys.exit(1)

