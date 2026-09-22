# =============================================================
# FAILURE ANALYSIS
# =============================================================

RETRIEVAL_STAGES = (
    "vector_results",
    "lexical_results",
    "rrf_results",
    "reranked_results",
    "final_results",
)


def analyze_retrieval_trace(relevant_documents, trace):
    """Analyze the trajectory of relevant documents through retrieval stages."""

    analysis = {}

    for document_id in relevant_documents:
        positions = {}

        for stage in RETRIEVAL_STAGES:
            stage_results = trace.get(stage, [])

            position = None

            for result in stage_results:
                if result["document_id"] == document_id:
                    position = result["rank"]
                    break
        
            positions[stage] = position

        first_seen_stage = None
        last_seen_stage = None

        for stage in RETRIEVAL_STAGES:
            if positions[stage] is not None:
                if first_seen_stage is None:
                    first_seen_stage = stage

                last_seen_stage = stage

        failure_stage = None

        if last_seen_stage is not None and last_seen_stage != RETRIEVAL_STAGES[-1]:
            last_seen_index = RETRIEVAL_STAGES.index(last_seen_stage)
            failure_stage = RETRIEVAL_STAGES[last_seen_index + 1]

        if first_seen_stage is None:
            failure_reason = "never_retrieved"
        elif failure_stage is not None:
            failure_reason = "lost_after_retrieval"
        else:
            failure_reason = "success"

        analysis[document_id] = {
            "positions": positions,
            "first_seen_stage": first_seen_stage,
            "last_seen_stage": last_seen_stage,
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
        }

    return analysis

def diagnose_retrieval_quality(metrics):

    recall    = metrics["recall_at_k"]
    precision = metrics["precision_at_k"]

    if recall is None:
        return "not_applicable"
    
    if recall == 0.0:
        return "no_relevant_results"

    if recall == 1.0 and precision == 1.0:
        return "excellent"
    
    if recall == 1.0 and precision < 1.0:
        return "high_recall_low_precision"
    
    if (0 < recall < 1.0) and precision > 0.0:
        return "partial"
    
    return "unknown"

def classify_retrieval_quality(metrics):

    recall = metrics["recall_at_k"]

    if recall is None:
        return "not_applicable"
    
    if recall == 1.0:
        return "complete"

    if 0 < recall < 1.0:
        return "partial"

    if recall == 0.0:
        return "miss"

    return "unknown"

def validate_retrieval_input(
    relevant_documents,
    retrieved_documents,
    k
):
    # Validar k
    if not isinstance(k, int):
        raise TypeError(f"k deve ser um inteiro. Recebido: {type(k).__name__}")
    
    if k <= 0:
        raise ValueError(f"k deve ser maior que 0. Recebido: {k}")    
   
    # Validar relevant_documents
    validate_relevant_documents(relevant_documents)

    # Validar retrieved_documents
    validate_retrieved_documents(retrieved_documents)

    # Validar rank
    validate_ranking(retrieved_documents)

    return True    

def validate_ranking(retrieved_documents):
    # Validação obrigatória, Não pode ser duplicado e precisa começar em 1
    ranks = [item["rank"] for item in retrieved_documents]

    if not ranks:
        return True
        
    if len(ranks) != len(set(ranks)):
        raise ValueError("rank não pode se repetir")
    
    if min(ranks) != 1:
        raise ValueError("rank deve começar em 1")
    
    return True    

def validate_relevant_documents(relevant_documents):
    # Regra 1 - Validar se é uma lista
    if not isinstance(relevant_documents, list):
        raise TypeError(
            f"relevant_documents deve ser uma lista. "
            f"Recebido: {type(relevant_documents).__name__}"
        )

    # Regra 2 - Validar se os IDs são strings
    for item in relevant_documents:
        if not isinstance(item, str):
            raise TypeError(
                f"Cada document_id deve ser uma string. "
                f"Recebido: {item!r} "
                f"do tipo {type(item).__name__}"
            )

    # Regra 3 - Identificar todas as duplicidades
    seen = set()
    duplicates = set()

    for doc_id in relevant_documents:
        if doc_id in seen:
            duplicates.add(doc_id)
        else:
            seen.add(doc_id)

    if duplicates:
        raise ValueError(
            f"Documentos duplicados no Golden Dataset: "
            f"{sorted(duplicates)}"
        )


    return True

def validate_retrieved_documents(retrieved_documents):
    # 1.3 Validar retrieved_documents
    if not isinstance(retrieved_documents, list):
        raise TypeError(
            f"retrieved_documents deve ser uma lista. "
            f"Recebido: {type(retrieved_documents).__name__}"
        )
    
    # Validar que cada item tem os campos obrigatórios
    required_fields = {"document_id", "chunk_id", "rank"}
    for idx, item in enumerate(retrieved_documents):
        if not isinstance(item, dict):
            raise TypeError(
                f"Cada item em retrieved_documents deve ser um dicionário. "
                f"Item no índice {idx} é {type(item).__name__}"
            )
        
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            raise ValueError(
                f"Item no índice {idx} está faltando campos obrigatórios: "
                f"{', '.join(missing_fields)}"
            )
        
        # Validar tipo do rank
        if isinstance(item["rank"], (float)):
            raise TypeError(
                #f"rank não pode ser float. Item {idx}: {type(item['rank']).__name__}"
                f"Esperado: int > 0. Recebido: Item {idx} {type(item['rank']).__name__} = {item['rank']}"
            )
        
        if isinstance(item["rank"], bool):
            raise TypeError(
                f"rank no índice {idx} não pode ser booleano (bool). "
                f"Esperado: int > 0. Recebido: {type(item['rank']).__name__} = {item['rank']}"
            )
        
        if not isinstance(item["rank"], (int)):
            raise TypeError(
                f"rank deve ser numérico inteiro. Item {idx}: {type(item['rank']).__name__}"
            )
        
        if item["rank"] <= 0:
            raise ValueError(
                f"rank deve ser maior que 0. Item {idx}: Recebido: {item['rank']}"
            )
        
        if not isinstance(item["document_id"], str):
            raise TypeError(
                    f"document_id deve ser uma string. "
                    f"Item {idx}: {type(item['document_id']).__name__}"
            )
        
        if not isinstance(item["chunk_id"], str):
            raise TypeError(
                f"chunk_id deve ser uma string. "
                f"Item {idx}: {type(item['chunk_id']).__name__}"
            )
    
    return True

def deduplicate_documents(retrieved_documents):
    '''
    Remove duplicatas de documentos mantendo a primeira ocorrência

    Args:
         retrieved_documents: Lista de dicionários com:
            -document_id: Id do documento
            -chunk_id: Id do chunk
            -rank:posição na lista de resutados

    Returns:
         Lista de documentos únicos com:
            -document_id: Id do documento
            -first_rank: posição da primeira ocorrência
            -chunks: lista de chunk_ids daquele documento
    '''
    seen = {}

    for item in retrieved_documents:
        doc_id   = item["document_id"]
        chunk_id = item["chunk_id"]
        rank     = item["rank"]

        if doc_id in seen:
            # documento já existe na lista, adiciona o chunk
            seen[doc_id]["chunks"].append(chunk_id)
        else:
            # primeira ocorrência, cria nova entrada
            seen[doc_id]={"first_rank":rank, 
                          "chunks": [chunk_id]
            }

    # convert o dicionário em lista
                        
    result=[]

    for doc_id, data in seen.items():
        result.append({
            "document_id": doc_id,
            "first_rank": data["first_rank"],
            "chunks": data["chunks"]
        })

    # Ordena os documentos pelo ranking da primeira ocorrência
    result.sort(key=lambda doc: doc["first_rank"])

    return result

def get_top_k_documents(deduplicated_documents, k):
    """
    A lista de entrada já está ordenada por first_rank, pois a função deduplicate_documents() preserva a ordem de primeira aparição.
    Portanto, os primeiros k elementos da lista são exatamente os primeiros k documentos únicos.
    A função pega esses k elementos e extrai apenas os document_id.
    Extrai os primeiros k documentos únicos de uma lista deduplicada,
    preservando a ordem de first_rank.
    
    Args:
        deduplicated_documents: Lista de dicionários com:
            - document_id: ID do documento
            - first_rank: posição da primeira ocorrência
            - chunks: lista de chunks daquele documento
        
        k: Número de documentos a serem retornados
    
    Returns:
        Lista com os primeiros k documentos (limitada ao tamanho da lista)
    """
    # Pega os primeiros k documentos (ou todos, se a lista for menor que k)
    return [doc["document_id"] for doc in deduplicated_documents[:k]]

def compare_retrieval(relevant_documents, top_k_documents):
    """
    Compara os documentos relevantes com os recuperados no Top-K.
    Preserva a ordem dos documentos para facilitar a leitura.
    """
    relevant_set = set(relevant_documents)
    top_k_set = set(top_k_documents)
    
    # hits: mantém a ordem de top_k_documents (primeira aparição)
    hits = [doc for doc in top_k_documents if doc in relevant_set]
    
    # misses: mantém a ordem de relevant_documents
    misses = [doc for doc in relevant_documents if doc not in top_k_set]
    
    # false_positives: mantém a ordem de top_k_documents
    false_positives = [doc for doc in top_k_documents if doc not in relevant_set]
    
    return {
        "hits": hits,
        "misses": misses,
        "false_positives": false_positives
    }

def calculate_metrics(relevant_documents, top_k_documents):
    # Deriva hits internamente (garantia de consistência)
    relevant_set = set(relevant_documents)

    hits = [
        doc for doc in top_k_documents
        if doc in relevant_set
    ]

    # Métricas básicas
    total_relevant = len(relevant_documents)
    retrieved_relevant_count = len(hits)
    retrieved_total = len(top_k_documents)
    
    # 1. Recall@K
    # Só é aplicável se houver documentos relevantes
    if total_relevant > 0:
        recall_at_k = retrieved_relevant_count / total_relevant
    else:
        recall_at_k = None  # Não aplicável (divisão por zero)
    
    # 2. Precision@K
    # Sempre é aplicável (denominador é o total recuperado, que nunca é zero)
    if retrieved_total > 0:
        precision_at_k = retrieved_relevant_count / retrieved_total
    else:
        precision_at_k =  None # Se não recuperou nada, precisão é 0
    
    # 3. Reciprocal Rank (RR)
    # Encontra a posição do primeiro hit
    first_relevant_position = None
    reciprocal_rank = 0.0
    
    if hits:
        # Encontra a posição do primeiro hit no Top-K
        for idx, doc in enumerate(top_k_documents, start=1):
            if doc in hits:
                first_relevant_position = idx
                reciprocal_rank = 1.0 / idx
                break
    
    return {
        
        "total_relevant": total_relevant,
        "retrieved_relevant_count": retrieved_relevant_count,
        "retrieved_total": retrieved_total,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "first_relevant_position": first_relevant_position,
        "reciprocal_rank": reciprocal_rank
    }

def evaluate_retrieval(
    relevant_documents,
    retrieved_documents,
    k
):
    validate_retrieval_input(relevant_documents, retrieved_documents, k)
            
    try:
        # 1-Deduplicar documentos
        deduplicated_documents = deduplicate_documents(retrieved_documents)
        # 2-Obter top_k documental    
        top_k_documents = get_top_k_documents(deduplicated_documents, k)
        # 3-Comparar com o Golden Dataset
        comparison = compare_retrieval(relevant_documents, top_k_documents)
        hits = comparison["hits"]
        misses = comparison["misses"]
        false_positives = comparison["false_positives"]
        # 4-Cálcular métricas    
        metrics = calculate_metrics(relevant_documents, top_k_documents)
        # 5- Classificação
        quality = classify_retrieval_quality(metrics)
        diagnosis = diagnose_retrieval_quality(metrics) 
        # 6-Montar resultado final
        return {
            "deduplicated_documents": deduplicated_documents,
            "top_k_documents": top_k_documents,
            "hits": hits,
            "misses": misses,
            "false_positives": false_positives,
            "metrics": metrics,
            "quality": quality,
            "diagnosis": diagnosis
        }
       
    except Exception as e:
        # Relançamos a exceção com contexto adicional
        raise RuntimeError(f"Erro durante a execução do pipeline: {str(e)}") from e


