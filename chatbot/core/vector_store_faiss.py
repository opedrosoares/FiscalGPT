#!/usr/bin/env python3
"""
Sistema de Banco Vetorial para o Chatbot ANTAQ usando FAISS
Gerencia embeddings e busca semântica das normas
"""

import pandas as pd
import numpy as np
import faiss
import openai
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
import os
from pathlib import Path
import logging
from tqdm import tqdm
import tiktoken
import re
from datetime import datetime
import pickle

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreANTAQ:
    """
    Classe para gerenciar o banco vetorial das normas ANTAQ usando FAISS
    """
    
    def __init__(
        self, 
        openai_api_key: str,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        chunk_size: int = 600,
        chunk_overlap: int = 100
    ):
        """
        Inicializa o sistema de banco vetorial com FAISS
        
        Args:
            openai_api_key: Chave da API OpenAI
            persist_directory: Diretório para persistir o banco
            collection_name: Nome da coleção no FAISS
            chunk_size: Tamanho dos chunks de texto
            chunk_overlap: Sobreposição entre chunks
        """
        
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Resolver diretório de persistência padrão para a raiz do projeto
        if persist_directory is None:
            try:
                from chatbot.config.config import FAISS_PERSIST_DIRECTORY as DEFAULT_FAISS_DIR
                persist_directory = str(DEFAULT_FAISS_DIR)
            except Exception:
                persist_directory = str(Path(__file__).parent.parent.parent / 'faiss_db')

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Resolver nome da coleção
        resolved_collection_name: Optional[str] = collection_name
        if resolved_collection_name is None:
            try:
                from chatbot.config.config import COLLECTION_NAME as DEFAULT_COLLECTION_NAME
                resolved_collection_name = DEFAULT_COLLECTION_NAME
            except Exception:
                resolved_collection_name = "default"
        
        self.collection_name: Optional[str] = resolved_collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Inicializar FAISS
        self.index_file = self.persist_directory / f"{self.collection_name}_index.faiss"
        self.metadata_file = self.persist_directory / f"{self.collection_name}_metadata.pkl"
        
        # Carregar ou criar índice FAISS
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, 'rb') as f:
                    self.metadata_list = pickle.load(f)
                logger.info(f"FAISS index carregado: {self.index.ntotal} vetores")
            except Exception as e:
                logger.warning(f"Erro ao carregar índice FAISS existente: {e}. Criando novo índice.")
                self._create_new_index()
        else:
            self._create_new_index()
        
        # Tokenizer para contagem de tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"VectorStore FAISS inicializado em: {self.persist_directory}")
        if self.collection_name:
            logger.info(f"Coleção ativa: {self.collection_name}")
    
    def _create_new_index(self):
        """Cria um novo índice FAISS"""
        dimension = 1536  # Dimensão do embedding text-embedding-3-small
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product para similaridade cosseno
        self.metadata_list = []
        logger.info("Novo índice FAISS criado")
    
    def _save_index(self):
        """Salva o índice FAISS e metadados"""
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata_list, f)
            logger.info(f"Índice FAISS salvo: {self.index.ntotal} vetores")
        except Exception as e:
            logger.error(f"Erro ao salvar índice FAISS: {e}")
    
    def _count_tokens(self, text: str) -> int:
        """Conta tokens no texto"""
        return len(self.tokenizer.encode(text))
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Divide texto em chunks inteligentes
        
        Args:
            text: Texto para dividir
            metadata: Metadados do documento
            
        Returns:
            Lista de chunks com metadados
        """
        
        if not text or len(text.strip()) < 50:
            return []
        
        chunks = []
        
        # Limpar e normalizar texto
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Dividir por parágrafos primeiro
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self._count_tokens(paragraph)
            
            # Se parágrafo é muito grande, dividir por sentenças
            if paragraph_tokens > self.chunk_size:
                sentences = re.split(r'[.!?]+', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    sentence_tokens = self._count_tokens(sentence)
                    
                    if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                        # Salvar chunk atual
                        chunks.append({
                            'text': current_chunk.strip(),
                            'metadata': {
                                **metadata,
                                'chunk_index': len(chunks),
                                'tokens': current_tokens
                            }
                        })
                        
                        # Iniciar novo chunk com overlap
                        overlap_text = self._get_overlap(current_chunk)
                        current_chunk = overlap_text + " " + sentence
                        current_tokens = self._count_tokens(current_chunk)
                    else:
                        current_chunk += " " + sentence
                        current_tokens += sentence_tokens
            else:
                # Parágrafo normal
                if current_tokens + paragraph_tokens > self.chunk_size and current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            **metadata,
                            'chunk_index': len(chunks),
                            'tokens': current_tokens
                        }
                    })
                    
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = overlap_text + " " + paragraph
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk += " " + paragraph
                    current_tokens += paragraph_tokens
        
        # Adicionar último chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'tokens': current_tokens
                }
            })
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Obter texto de overlap do final do chunk"""
        words = text.split()
        overlap_words = min(self.chunk_overlap // 4, len(words))  # Aproximação
        return " ".join(words[-overlap_words:]) if overlap_words > 0 else ""
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Gera embedding usando OpenAI
        
        Args:
            text: Texto para gerar embedding
            
        Returns:
            Lista de floats representando o embedding
        """
        
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            raise
    
    def _generate_document_id(self, codigo_registro: str, chunk_index: int) -> str:
        """Gera ID único para documento"""
        return f"{codigo_registro}_chunk_{chunk_index}"
    
    def _atualizar_status_vetorizacao(self, parquet_path: str, codigos_registro: List[str]) -> None:
        """
        Atualiza o status de vetorização das normas no arquivo parquet
        
        Args:
            parquet_path: Caminho para o arquivo parquet
            codigos_registro: Lista de códigos de registro das normas processadas
        """
        try:
            logger.info(f"Atualizando status de vetorização para {len(codigos_registro)} normas...")
            
            # Carregar dados
            df = pd.read_parquet(parquet_path)
            
            # Marcar normas como vetorizadas
            mask = df['codigo_registro'].isin(codigos_registro)
            df.loc[mask, 'vetorizado'] = True
            df.loc[mask, 'ultima_verificacao_vetorizacao'] = datetime.now()
            
            # Salvar arquivo atualizado
            df.to_parquet(parquet_path, index=False)
            
            logger.info(f"✅ Status de vetorização atualizado com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao atualizar status de vetorização: {e}")
            import traceback
            traceback.print_exc()
    
    def load_and_process_data(self, parquet_path: str, force_rebuild: bool = False, sample_size: Optional[int] = None, incremental: bool = True, collection_name: Optional[str] = None) -> bool:
        """
        Carrega e processa dados do parquet para o banco vetorial FAISS
        
        Args:
            parquet_path: Caminho para o arquivo parquet
            force_rebuild: Se deve reconstruir o banco mesmo se existir
            sample_size: Tamanho da amostra para teste
            incremental: Se deve processar apenas normas não vetorizadas
            
        Returns:
            True se processado com sucesso
        """
        
        try:
            # Resolver coleção ativa (prioriza parâmetro explícito)
            active_collection_name = collection_name or self.collection_name or "default"

            # Forçar rebuild se solicitado
            if force_rebuild:
                logger.info(f"Reconstruindo índice FAISS para coleção '{active_collection_name}'...")
                self._create_new_index()
                # Remover arquivos antigos
                if self.index_file.exists():
                    self.index_file.unlink()
                if self.metadata_file.exists():
                    self.metadata_file.unlink()

            # Carregar dados
            logger.info(f"Carregando dados de: {parquet_path}")
            df = pd.read_parquet(parquet_path)
            
            # Verificar se coluna vetorizado existe
            if 'vetorizado' not in df.columns:
                logger.warning("Coluna 'vetorizado' não encontrada. Adicionando coluna...")
                df['vetorizado'] = False
                df['ultima_verificacao_vetorizacao'] = None
                # Salvar arquivo atualizado
                df.to_parquet(parquet_path, index=False)
                logger.info("Arquivo atualizado com coluna 'vetorizado'")
            
            # Seleção de conteúdo: se for dataset wiki_js, usar coluna 'conteudo';
            # caso contrário, usar 'conteudo_pdf' e filtros originais.
            is_wiki_js = 'conteudo' in df.columns and 'codigo_registro' not in df.columns
            if is_wiki_js:
                text_col = 'conteudo'
                df_filtered = df[(df[text_col].notna()) & (df[text_col].astype(str).str.len() > 50)].copy()
                # Harmonizar metadados mínimos
                df_filtered['codigo_registro'] = df_filtered['id'].astype(str)
                df_filtered['titulo'] = df_filtered.get('title', df_filtered.get('path', ''))
                df_filtered['assunto'] = df_filtered.get('path', '')
                df_filtered['situacao'] = df_filtered.get('situacao', 'N/A')
                df_filtered['link_pdf'] = df_filtered.get('link_pdf', '')
                # Preservar tipo_material se existir (ex.: Imports::<categoria>); senão, assumir WikiJS
                if 'tipo_material' not in df_filtered.columns:
                    df_filtered['tipo_material'] = 'WikiJS'
                else:
                    df_filtered['tipo_material'] = df_filtered['tipo_material'].fillna('WikiJS')
                df_filtered['assinatura'] = df_filtered.get('assinatura', pd.NaT)
                df_filtered['publicacao'] = df_filtered.get('publicacao', pd.NaT)
            else:
                text_col = 'conteudo_pdf'
                df_filtered = df[
                    (df['situacao'] == 'Em vigor') & 
                    (df[text_col].notna()) & 
                    (df[text_col].str.len() > 100)
                ].copy()
            
            # Se modo incremental, filtrar apenas não vetorizadas
            if incremental and not force_rebuild:
                df_filtered = df_filtered[~df_filtered['vetorizado']].copy()
                logger.info(f"Modo incremental: {len(df_filtered)} normas não vetorizadas encontradas")
            else:
                logger.info(f"Processando todas as {len(df_filtered)} normas em vigor com conteúdo...")
            
            # Aplicar amostra se especificado
            if sample_size and sample_size < len(df_filtered):
                df_filtered = df_filtered.sample(n=sample_size, random_state=42).copy()
                logger.info(f"Usando amostra de {sample_size} normas para teste rápido...")
            
            if len(df_filtered) == 0:
                logger.info("Nenhuma norma para processar!")
                return True
            
            logger.info(f"Processando {len(df_filtered)} normas...")
            
            # Processar cada norma individualmente
            total_chunks = 0
            normas_processadas = []
            
            # Preparar dados para inserção em lote
            all_embeddings = []
            all_metadatas = []
            
            # Processar cada norma individualmente para garantir persistência
            for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processando normas"):
                try:
                    # Preparar metadados
                    metadata = {
                        'codigo_registro': str(row['codigo_registro']),
                        'titulo': str(row['titulo']),
                        'autor': str(row.get('autor', '')),
                        'assunto': str(row.get('assunto', '')),
                        'situacao': str(row.get('situacao', '')),
                        'link_pdf': str(row.get('link_pdf', '')),
                        'tipo_material': str(row.get('tipo_material', '')),
                        # Identificar a coleção de origem para a camada de interface
                        'collection': active_collection_name,
                        'assinatura': row['assinatura'].strftime('%Y-%m-%d') if pd.notna(row['assinatura']) else 'N/A',
                        'publicacao': row['publicacao'].strftime('%Y-%m-%d') if pd.notna(row['publicacao']) else 'N/A',
                        'tamanho_pdf': int(row.get('tamanho_pdf', 0)) if (('tamanho_pdf' in row.index) and pd.notna(row.get('tamanho_pdf'))) else 0,
                        'paginas_extraidas': int(row.get('paginas_extraidas', 0)) if (('paginas_extraidas' in row.index) and pd.notna(row.get('paginas_extraidas'))) else 0
                    }
                    
                    # Criar texto combinado para busca
                    texto_completo = f"""
                    TÍTULO: {row['titulo']}
                    ASSUNTO: {metadata.get('assunto','')}
                    CONTEÚDO: {row[text_col]}
                    """.strip()
                    
                    # Dividir em chunks
                    chunks = self._chunk_text(texto_completo, metadata)
                    
                    # Processar chunks
                    for chunk in chunks:
                        # Gerar embedding
                        embedding = self._generate_embedding(chunk['text'])
                        
                        # Adicionar à lista de embeddings
                        all_embeddings.append(embedding)
                        all_metadatas.append(chunk['metadata'])
                        total_chunks += 1
                    
                    # Adicionar à lista de normas processadas
                    normas_processadas.append(row['codigo_registro'])
                    
                    # Atualizar status de vetorização individualmente
                    if incremental:
                        if not is_wiki_js:
                            self._atualizar_status_vetorizacao(parquet_path, [row['codigo_registro']])
                    
                    # Log da norma sendo processada
                    logger.info(f"📄 Processado: {row['titulo']} (Código: {row['codigo_registro']}) - {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"❌ Erro ao processar norma {row['codigo_registro']}: {e}")
                    continue
            
            # Inserir todos os embeddings no índice FAISS de uma vez
            if all_embeddings:
                embeddings_array = np.array(all_embeddings, dtype=np.float32)
                self.index.add(embeddings_array)
                
                # Atualizar lista de metadados
                self.metadata_list.extend(all_metadatas)
                
                # Salvar índice
                self._save_index()
                
                logger.info(f"✅ {len(all_embeddings)} embeddings adicionados ao índice FAISS")
            
            logger.info(f"✅ Processamento concluído! {total_chunks} chunks inseridos no banco vetorial")
            logger.info(f"✅ {len(normas_processadas)} normas marcadas como vetorizadas")
            
            # Listar nomes das normas vetorizadas
            if normas_processadas:
                logger.info("📋 NORMAS VETORIZADAS NESTE LOTE:")
                for codigo in normas_processadas:
                    # Buscar o título da norma no DataFrame
                    norma_info = df_filtered[df_filtered['codigo_registro'] == codigo]
                    if not norma_info.empty:
                        titulo = norma_info.iloc[0]['titulo']
                        logger.info(f"   • {titulo} (Código: {codigo})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar dados: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search(
        self, 
        query: str, 
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca semântica no banco vetorial FAISS
        
        Args:
            query: Consulta do usuário
            n_results: Número de resultados
            filters: Filtros de metadados (implementação básica)
            
        Returns:
            Lista de resultados ranqueados
        """
        
        try:
            if self.index.ntotal == 0:
                logger.warning("Índice FAISS vazio")
                return []
            
            # Gerar embedding da consulta
            query_embedding = self._generate_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Realizar busca no FAISS
            distances, indices = self.index.search(query_vector, min(n_results, self.index.ntotal))
            
            # Preparar resultados
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata_list):
                    metadata = self.metadata_list[idx]
                    
                    # Aplicar filtros básicos se especificados
                    if filters:
                        skip_result = False
                        for key, value in filters.items():
                            if key in metadata and metadata[key] != value:
                                skip_result = True
                                break
                        if skip_result:
                            continue
                    
                    # Calcular similaridade (FAISS retorna distância, converter para similaridade)
                    similarity = float(distance)  # FAISS com IndexFlatIP retorna similaridade cosseno
                    
                    results.append({
                        'document': metadata.get('text', ''),  # FAISS não armazena texto, apenas metadados
                        'metadata': metadata,
                        'similarity': similarity,
                        'distance': 1 - similarity
                    })
            
            # Ordenar por similaridade e retornar top n_results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Erro na busca: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas da coleção FAISS."""
        try:
            total_count = self.index.ntotal
            
            if total_count == 0:
                return {
                    'total_chunks': 0,
                    'total_normas_unicas': 0,
                    'top_autores': {},
                    'top_assuntos': {},
                    'distribuicao_anos': {},
                    'tipos_normas': {},
                    'situacoes': {}
                }
            
            # Análise dos metadados
            autores = {}
            assuntos = {}
            anos = {}
            tipos_normas = {}
            situacoes = {}
            normas_unicas = set()
            
            for meta in self.metadata_list:
                # Contagem de normas únicas
                codigo_registro = meta.get('codigo_registro', '')
                if codigo_registro:
                    normas_unicas.add(codigo_registro)
                
                # Análise de autores
                autor = meta.get('autor', 'Desconhecido')
                autores[autor] = autores.get(autor, 0) + 1
                
                # Análise de assuntos
                assunto = meta.get('assunto', 'Desconhecido')
                assuntos[assunto] = assuntos.get(assunto, 0) + 1
                
                # Análise de anos
                if meta.get('assinatura'):
                    ano = meta['assinatura'][:4]
                    anos[ano] = anos.get(ano, 0) + 1
                
                # Análise de tipos de normas
                titulo = meta.get('titulo', '')
                if titulo:
                    titulo_upper = titulo.upper()
                    if 'RESOLUÇÃO' in titulo_upper:
                        tipo = 'Resolução'
                    elif 'PORTARIA' in titulo_upper:
                        tipo = 'Portaria'
                    elif 'TERMO DE AUTORIZAÇÃO' in titulo_upper:
                        tipo = 'Termo de Autorização'
                    elif 'INSTRUÇÃO NORMATIVA' in titulo_upper:
                        tipo = 'Instrução Normativa'
                    elif 'DELIBERAÇÃO' in titulo_upper:
                        tipo = 'Deliberação'
                    elif 'ACÓRDÃO' in titulo_upper:
                        tipo = 'Acórdão'
                    else:
                        tipo = 'Outros'
                    
                    tipos_normas[tipo] = tipos_normas.get(tipo, 0) + 1
                
                # Análise de situações
                situacao = meta.get('situacao', 'Desconhecida')
                situacoes[situacao] = situacoes.get(situacao, 0) + 1
            
            return {
                'total_chunks': total_count,
                'total_normas_unicas': len(normas_unicas),
                'top_autores': dict(list(sorted(autores.items(), key=lambda x: x[1], reverse=True))[:10]),
                'top_assuntos': dict(list(sorted(assuntos.items(), key=lambda x: x[1], reverse=True))[:10]),
                'distribuicao_anos': dict(sorted(anos.items())),
                'tipos_normas': dict(sorted(tipos_normas.items(), key=lambda x: x[1], reverse=True)),
                'situacoes': dict(sorted(situacoes.items(), key=lambda x: x[1], reverse=True))
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {'error': str(e)}

    def list_collections_with_counts(self) -> List[Tuple[str, int]]:
        """Lista coleções disponíveis com seus tamanhos (FAISS usa arquivos separados)."""
        try:
            collections = []
            for file_path in self.persist_directory.glob("*_index.faiss"):
                collection_name = file_path.stem.replace("_index", "")
                try:
                    # Tentar carregar o índice para obter contagem
                    temp_index = faiss.read_index(str(file_path))
                    count = temp_index.ntotal
                    collections.append((collection_name, count))
                except Exception:
                    collections.append((collection_name, 0))
            return sorted(collections, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Erro ao listar coleções: {e}")
            return []

    def get_total_documents_count(self) -> int:
        """Retorna o número total de documentos no índice FAISS."""
        try:
            return self.index.ntotal
        except Exception:
            return 0
    
    def get_vetorizacao_stats(self, parquet_path: str) -> Dict[str, Any]:
        """
        Obtém estatísticas de vetorização do arquivo parquet
        
        Args:
            parquet_path: Caminho para o arquivo parquet
            
        Returns:
            Dicionário com estatísticas de vetorização
        """
        try:
            df = pd.read_parquet(parquet_path)
            
            # Verificar se coluna vetorizado existe
            if 'vetorizado' not in df.columns:
                return {
                    'error': 'Coluna vetorizado não encontrada',
                    'total_normas': len(df)
                }
            
            # Estatísticas básicas
            total_normas = len(df)
            normas_vetorizadas = df['vetorizado'].sum()
            normas_nao_vetorizadas = total_normas - normas_vetorizadas
            
            # Normas em vigor com conteúdo
            normas_em_vigor = df[
                (df['situacao'] == 'Em vigor') & 
                (df['conteudo_pdf'].notna()) & 
                (df['conteudo_pdf'].str.len() > 100)
            ]
            
            normas_em_vigor_vetorizadas = normas_em_vigor['vetorizado'].sum()
            normas_em_vigor_nao_vetorizadas = len(normas_em_vigor) - normas_em_vigor_vetorizadas
            
            # Última verificação
            ultima_verificacao = df['ultima_verificacao_vetorizacao'].max() if 'ultima_verificacao_vetorizacao' in df.columns else None
            
            return {
                'total_normas': total_normas,
                'normas_vetorizadas': int(normas_vetorizadas),
                'normas_nao_vetorizadas': int(normas_nao_vetorizadas),
                'percentual_vetorizado': float(normas_vetorizadas / total_normas * 100) if total_normas > 0 else 0,
                'normas_em_vigor_com_conteudo': len(normas_em_vigor),
                'normas_em_vigor_vetorizadas': int(normas_em_vigor_vetorizadas),
                'normas_em_vigor_nao_vetorizadas': int(normas_em_vigor_nao_vetorizadas),
                'percentual_em_vigor_vetorizado': float(normas_em_vigor_vetorizadas / len(normas_em_vigor) * 100) if len(normas_em_vigor) > 0 else 0,
                'ultima_verificacao': ultima_verificacao.isoformat() if ultima_verificacao else None
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas de vetorização: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Teste básico
    try:
        from chatbot.config.config import OPENAI_API_KEY
    except ImportError:
        print("❌ Erro ao importar configurações do chatbot")
        exit(1)
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY não encontrada no arquivo .env")
        exit(1)
    
    # Inicializar vector store
    vs = VectorStoreANTAQ(OPENAI_API_KEY)
    
    # Verificar estatísticas de vetorização
    parquet_path = "../normas_antaq_completo.parquet"
    vetorizacao_stats = vs.get_vetorizacao_stats(parquet_path)
    
    print("\n📊 ESTATÍSTICAS DE VETORIZAÇÃO:")
    if 'error' not in vetorizacao_stats:
        print(f"   Total de normas: {vetorizacao_stats['total_normas']}")
        print(f"   Normas vetorizadas: {vetorizacao_stats['normas_vetorizadas']}")
        print(f"   Normas não vetorizadas: {vetorizacao_stats['normas_nao_vetorizadas']}")
        print(f"   Percentual vetorizado: {vetorizacao_stats['percentual_vetorizado']:.1f}%")
        print(f"   Normas em vigor com conteúdo: {vetorizacao_stats['normas_em_vigor_com_conteudo']}")
        print(f"   Normas em vigor vetorizadas: {vetorizacao_stats['normas_em_vigor_vetorizadas']}")
        print(f"   Percentual em vigor vetorizado: {vetorizacao_stats['percentual_em_vigor_vetorizado']:.1f}%")
        if vetorizacao_stats['ultima_verificacao']:
            print(f"   Última verificação: {vetorizacao_stats['ultima_verificacao']}")
    else:
        print(f"   Erro: {vetorizacao_stats['error']}")
    
    # Carregar dados (modo incremental)
    print(f"\n🚀 Iniciando vetorização incremental...")
    success = vs.load_and_process_data(parquet_path, incremental=True, sample_size=5)
    
    if success:
        # Teste de busca
        results = vs.search("licenciamento portuário", n_results=5)
        
        print("\n🔍 TESTE DE BUSCA:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similaridade: {result['similarity']:.3f}")
            print(f"   Título: {result['metadata'].get('titulo', 'N/A')}")
            print(f"   Texto: {result['document'][:200]}...")
        
        # Estatísticas da coleção
        stats = vs.get_collection_stats()
        print(f"\n📊 ESTATÍSTICAS DA COLEÇÃO:")
        print(f"   Total chunks: {stats.get('total_chunks', 0)}")
        
        # Verificar estatísticas atualizadas
        vetorizacao_stats_atualizada = vs.get_vetorizacao_stats(parquet_path)
        print(f"\n📊 ESTATÍSTICAS ATUALIZADAS:")
        print(f"   Normas vetorizadas: {vetorizacao_stats_atualizada['normas_vetorizadas']}")
        print(f"   Percentual vetorizado: {vetorizacao_stats_atualizada['percentual_vetorizado']:.1f}%")
