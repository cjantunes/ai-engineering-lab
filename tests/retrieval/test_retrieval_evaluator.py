from src.retrieval.retrieval_evaluator import (
    analyze_retrieval_trace,
    evaluate_retrieval,
    summarize_retrieval_failures,
)

import pytest

def test_retrieval_complete():
    """Teste 1 — Retrieval completo (todos os relevantes encontrados)."""
    relevant_documents = ["A", "B", "C"]

    retrieved_documents = [
        {"document_id": "A", "chunk_id": "A1", "rank": 1},
        {"document_id": "B", "chunk_id": "B1", "rank": 2},
        {"document_id": "C", "chunk_id": "C1", "rank": 3}
    ]

    k = 3

    result = evaluate_retrieval(relevant_documents, retrieved_documents, k)

    # Comparação
    assert result["hits"]   == ["A", "B", "C"]
    assert result["misses"] == []           
    assert result["false_positives"] == []

    # Métricas
    assert result["metrics"]["recall_at_k"] == 1.0
    assert result["metrics"]["precision_at_k"] == 1.0
    assert result["metrics"]["reciprocal_rank"] == 1.0

    # Classificação e diagnóstico
    assert result["quality"] == "complete"
    assert result["diagnosis"] == "excellent"

    print("Teste 1 — Retrieval completo: PASSOU")


def test_retrieval_partial():
    """Teste 2 — Retrieval parcial (alguns relevantes encontrados, com ruído)."""
    relevant_documents = ["A", "B", "C"]

    retrieved_documents = [
        {"document_id": "A", "chunk_id": "A1", "rank": 1},
        {"document_id": "D", "chunk_id": "D1", "rank": 2},
        {"document_id": "B", "chunk_id": "B1", "rank": 3}
    ]

    k = 3

    result = evaluate_retrieval(relevant_documents, retrieved_documents, k)

    # Comparação
    assert result["hits"] == ["A", "B"]
    assert result["misses"] == ["C"]
    assert result["false_positives"] == ["D"]

    # Métricas
    assert result["metrics"]["recall_at_k"] == 2 / 3
    assert result["metrics"]["precision_at_k"] == 2 / 3
    assert result["metrics"]["reciprocal_rank"] == 1.0

    # Classificação e diagnóstico
    assert result["quality"] == "partial"
    assert result["diagnosis"] == "partial"

    print("Teste 2 — Retrieval parcial: PASSOU")


def test_retrieval_no_relevant_results():
    """Teste 3 — Nenhum documento relevante recuperado."""
    relevant_documents = ["A", "B"]

    retrieved_documents = [
        {"document_id": "C", "chunk_id": "C1", "rank": 1},
        {"document_id": "D", "chunk_id": "D1", "rank": 2}
    ]

    k = 2

    result = evaluate_retrieval(relevant_documents, retrieved_documents, k)

    # Comparação
    assert result["hits"] == []
    assert result["misses"] == ["A", "B"]
    assert result["false_positives"] == ["C", "D"]

    # Métricas
    assert result["metrics"]["recall_at_k"] == 0.0
    assert result["metrics"]["precision_at_k"] == 0.0
    assert result["metrics"]["reciprocal_rank"] == 0.0

    # Classificação e diagnóstico
    assert result["quality"] == "miss"
    assert result["diagnosis"] == "no_relevant_results"

    print("Teste 3 — Nenhum relevante recuperado: PASSOU")


def test_retrieval_empty_golden():
    """Teste 4 — Golden Dataset vazio (Recall não aplicável)."""
    relevant_documents = []

    retrieved_documents = [
        {"document_id": "A", "chunk_id": "A1", "rank": 1},
        {"document_id": "B", "chunk_id": "B1", "rank": 2}
    ]

    k = 2

    result = evaluate_retrieval(relevant_documents, retrieved_documents, k)

    # Comparação
    assert result["hits"] == []
    assert result["misses"] == []
    assert result["false_positives"] == ["A", "B"]

    # Métricas
    assert result["metrics"]["recall_at_k"] is None
    assert result["metrics"]["precision_at_k"] == 0.0
    assert result["metrics"]["reciprocal_rank"] == 0.0

    # Classificação e diagnóstico
    assert result["quality"] == "not_applicable"
    assert result["diagnosis"] == "not_applicable"

    print("Teste 4 — Golden Dataset vazio: PASSOU")


def test_retrieval_relevant_in_second_position():
    """Teste 5 — Relevante na segunda posição (Recall perfeito, Precision baixa)."""
    relevant_documents = ["A"]

    retrieved_documents = [
        {"document_id": "B", "chunk_id": "B1", "rank": 1},
        {"document_id": "A", "chunk_id": "A1", "rank": 2},
        {"document_id": "C", "chunk_id": "C1", "rank": 3}
    ]

    k = 3

    result = evaluate_retrieval(relevant_documents, retrieved_documents, k)

    # Comparação
    assert result["hits"] == ["A"]
    assert result["misses"] == []
    assert result["false_positives"] == ["B", "C"]

    # Métricas
    assert result["metrics"]["recall_at_k"] == 1.0
    assert result["metrics"]["precision_at_k"] == 1 / 3
    assert result["metrics"]["reciprocal_rank"] == 0.5

    # Classificação e diagnóstico
    assert result["quality"] == "complete"
    assert result["diagnosis"] == "high_recall_low_precision"

    print("Teste 5 — Relevante na 2ª posição: PASSOU")

def test_invalid_k_zero():
    """Teste 6 — k = 0 deve lançar ValueError."""
    relevant_documents = ["A", "B", "C"]
    retrieved_documents = [
        {"document_id": "A", "chunk_id": "A1", "rank": 1}
    ]

    try:
        evaluate_retrieval(relevant_documents, retrieved_documents, k=0)
        # Se chegar aqui, o teste falhou (nenhuma exceção foi lançada)
        assert False, "Esperado ValueError, mas nenhuma exceção foi lançada"
    except ValueError as e:
        print(f"Teste 6 — k=0 lançou ValueError: PASSOU")
        print(f"Mensagem: {e}")
    except Exception as e:
        assert False, f"Esperado ValueError, mas obtido {type(e).__name__}: {e}"


def test_duplicate_relevant_documents():
    """Teste 7 — Documentos relevantes duplicados deve lançar ValueError."""
    relevant_documents = ["A", "A", "B"]
    retrieved_documents = [
        {"document_id": "A", "chunk_id": "A1", "rank": 1}
    ]
    k = 1

    try:
        evaluate_retrieval(relevant_documents, retrieved_documents, k)
        assert False, "Esperado ValueError, mas nenhuma exceção foi lançada"
    except ValueError as e:
        print(f"Teste 7 — Documentos relevantes duplicados lançou ValueError: PASSOU")
        print(f"Mensagem: {e}")
    except Exception as e:
        assert False, f"Esperado ValueError, mas obtido {type(e).__name__}: {e}"


def test_duplicate_rank():
    """Teste 8 — Rank duplicado deve lançar ValueError."""
    relevant_documents = ["A", "B"]
    retrieved_documents = [
        {"document_id": "A", "chunk_id": "A1", "rank": 1},
        {"document_id": "B", "chunk_id": "B1", "rank": 1}
    ]
    k = 2

    try:
        evaluate_retrieval(relevant_documents, retrieved_documents, k)
        assert False, "Esperado ValueError, mas nenhuma exceção foi lançada"
    except ValueError as e:
        print(f"Teste 8 — Rank duplicado lançou ValueError: PASSOU")
        print(f"Mensagem: {e}")
    except Exception as e:
        assert False, f"Esperado ValueError, mas obtido {type(e).__name__}: {e}"


def test_rank_not_starting_at_one():
    """Teste 9 — Rank começando em 2 deve lançar ValueError."""
    relevant_documents = ["A"]
    retrieved_documents = [
        {"document_id": "A", "chunk_id": "A1", "rank": 2}
    ]
    k = 1

    try:
        evaluate_retrieval(relevant_documents, retrieved_documents, k)
        assert False, "Esperado ValueError, mas nenhuma exceção foi lançada"
    except ValueError as e:
        print(f"Teste 9 — Rank começando em 2 lançou ValueError: PASSOU")
        print(f"Mensagem: {e}")
    except Exception as e:
        assert False, f"Esperado ValueError, mas obtido {type(e).__name__}: {e}"


def test_rank_boolean():
    """Teste 10 — Rank booleano deve lançar TypeError."""
    relevant_documents = ["A"]
    retrieved_documents = [
        {"document_id": "A", "chunk_id": "A1", "rank": True}
    ]
    k = 1

    try:
        evaluate_retrieval(relevant_documents, retrieved_documents, k)
        assert False, "Esperado TypeError, mas nenhuma exceção foi lançada"
    except TypeError as e:
        print(f"Teste 10 — Rank booleano lançou TypeError: PASSOU")
        print(f"Mensagem: {e}")
    except Exception as e:
        assert False, f"Esperado TypeError, mas obtido {type(e).__name__}: {e}"

def test_ranking_physical_order_consistency():

    relevant_documents = ["E"]

    retrieved_documents = [
        {"document_id": "D", "chunk_id": "D1", "rank": 2},
        {"document_id": "C", "chunk_id": "C1", "rank": 3},
        {"document_id": "E", "chunk_id": "E1", "rank": 1}
    ]

    k = 1

    result = evaluate_retrieval(relevant_documents, retrieved_documents, k)

    try:
        # Verifica se o Top-K respeita o rank
        assert result["top_k_documents"] == ["E"]
        # Verifica se o Recall está correto
        assert result["metrics"]["recall_at_k"] == 1.0

        print("Teste 11 — Consistência do ranking: PASSOU")

    except AssertionError as e:
        print("Teste 11 — Consistência do ranking: FALHOU")

def test_analyze_retrieval_trace():
    relevant_documents = ["DOC_A", "DOC_B", "DOC_X"]

    trace = {
        "vector_results": [
            {"document_id": "DOC_C", "rank": 1},
            {"document_id": "DOC_A", "rank": 2},
            {"document_id": "DOC_B", "rank": 3},
        ],
        "lexical_results": [
            {"document_id": "DOC_B", "rank": 1},
        ],
        "rrf_results": [
            {"document_id": "DOC_B", "rank": 1},
            {"document_id": "DOC_A", "rank": 2},
        ],
        "reranked_results": [
            {"document_id": "DOC_A", "rank": 1},
            {"document_id": "DOC_B", "rank": 2},
        ],
        "final_results": [
            {"document_id": "DOC_A", "rank": 1},
        ],
    }

    result = analyze_retrieval_trace(relevant_documents, trace)

    assert result["DOC_A"]["failure_reason"] == "success"
    assert result["DOC_A"]["first_seen_stage"] == "vector_results"
    assert result["DOC_A"]["last_seen_stage"] == "final_results"
    assert result["DOC_A"]["failure_stage"] is None

    assert result["DOC_B"]["failure_reason"] == "lost_before_final"
    assert result["DOC_B"]["last_seen_stage"] == "reranked_results"
    assert result["DOC_B"]["failure_stage"] == "final_results"

    assert result["DOC_X"]["failure_reason"] == "never_retrieved"
    assert result["DOC_X"]["first_seen_stage"] is None
    assert result["DOC_X"]["last_seen_stage"] is None
    assert result["DOC_X"]["failure_stage"] is None

def test_document_lost_during_reranking():
    relevant_documents = ["DOC_A"]

    trace = {
        "vector_results": [
            {"document_id": "DOC_A", "rank": 2},
        ],
        "lexical_results": [],
        "rrf_results": [
            {"document_id": "DOC_A", "rank": 1},
        ],
        "reranked_results": [],
        "final_results": [],
    }

    result = analyze_retrieval_trace(relevant_documents, trace)

    assert result["DOC_A"]["first_seen_stage"] == "vector_results"
    assert result["DOC_A"]["last_seen_stage"] == "rrf_results"
    assert result["DOC_A"]["failure_stage"] == "reranked_results"
    assert result["DOC_A"]["failure_reason"] == "lost_during_reranking"
    assert result["DOC_A"]["diagnostic_area"] == "reranking_or_reranking_cutoff"

def test_document_lost_during_fusion():
    relevant_documents = ["DOC_A"]

    trace = {
        "vector_results": [
            {"document_id": "DOC_A", "rank": 2},
        ],
        "lexical_results": [
            {"document_id": "DOC_A", "rank": 1},
        ],
        "rrf_results": [],
        "reranked_results": [],
        "final_results": [],
    }

    result = analyze_retrieval_trace(relevant_documents, trace)

    assert result["DOC_A"]["first_seen_stage"] == "vector_results"
    assert result["DOC_A"]["last_seen_stage"] == "lexical_results"
    assert result["DOC_A"]["failure_stage"] == "rrf_results"
    assert result["DOC_A"]["failure_reason"] == "lost_during_fusion"

def test_document_retrieved_only_by_vector_and_lost_before_fusion():
    relevant_documents = ["DOC_A"]

    trace = {
        "vector_results": [
            {"document_id": "DOC_A", "rank": 2},
        ],
        "lexical_results": [],
        "rrf_results": [],
        "reranked_results": [],
        "final_results": [],
    }

    result = analyze_retrieval_trace(relevant_documents, trace)

    assert result["DOC_A"]["first_seen_stage"] == "vector_results"
    assert result["DOC_A"]["last_seen_stage"] == "vector_results"
    assert result["DOC_A"]["failure_stage"] == "rrf_results"
    assert result["DOC_A"]["failure_reason"] == "lost_during_fusion"

def test_summarize_retrieval_failures():
    trace_analysis = {
        "DOC_A": {
            "failure_reason": "success",
        },
        "DOC_B": {
            "failure_reason": "lost_before_final",
        },
        "DOC_X": {
            "failure_reason": "never_retrieved",
        },
    }

    result = summarize_retrieval_failures(trace_analysis)

    assert result["total_relevant"] == 3
    assert result["success"] == 1
    assert result["never_retrieved"] == 1
    
    assert result["lost_before_final"] == 1
    assert result["lost_during_fusion"] == 0
    assert result["lost_during_reranking"] == 0
    assert result["rates"]["success"] == pytest.approx(1 / 3)
    assert result["rates"]["never_retrieved"] == pytest.approx(1 / 3)
    assert result["rates"]["lost_before_final"] == pytest.approx(1 / 3)
    assert result["rates"]["lost_during_fusion"] == 0.0
    assert result["rates"]["lost_during_reranking"] == 0.0

def test_summarize_retrieval_failures_without_relevant_documents():
    trace_analysis = {}

    result = summarize_retrieval_failures(trace_analysis)

    assert result["total_relevant"] == 0
    assert result["success"] == 0
    assert result["never_retrieved"] == 0

    assert result["rates"]["success"] is None
    assert result["rates"]["never_retrieved"] is None
    assert result["rates"]["lost_during_fusion"] is None
    assert result["rates"]["lost_during_reranking"] is None
    assert result["rates"]["lost_before_final"] is None
    assert result["rates"]["lost_during_pipeline"] is None

def test_evaluate_retrieval_with_failure_analysis():
    relevant_documents = ["DOC_A", "DOC_B"]

    retrieved_documents = [
    {
        "document_id": "DOC_A",
        "chunk_id": "CHUNK_A1",
        "rank": 1,
    },
]

    trace = {
        "vector_results": [
            {"document_id": "DOC_A", "rank": 1},
            {"document_id": "DOC_B", "rank": 2},
        ],
        "lexical_results": [
            {"document_id": "DOC_B", "rank": 1},            
        ],
        "rrf_results": [
            {"document_id": "DOC_A", "rank": 1},
            {"document_id": "DOC_B", "rank": 2},
        ],
        "reranked_results": [
            {"document_id": "DOC_A", "rank": 1},
            {"document_id": "DOC_B", "rank": 2},
        ],
        "final_results": [
            {"document_id": "DOC_A", "rank": 1},
        ],
    }

    result = evaluate_retrieval(
        relevant_documents,
        retrieved_documents,
        k=1,
        trace=trace,
    )

    assert result["failure_analysis"]["DOC_A"]["failure_reason"] == "success"
    assert (
        result["failure_analysis"]["DOC_B"]["failure_reason"]
        == "lost_before_final"
    )

    assert result["failure_summary"]["total_relevant"] == 2
    assert result["failure_summary"]["success"] == 1
    assert result["failure_summary"]["lost_before_final"] == 1

    assert result["metrics"]["recall_at_k"] == pytest.approx(0.5)
    assert result["metrics"]["precision_at_k"] == pytest.approx(1.0)
    assert result["metrics"]["reciprocal_rank"] == pytest.approx(1.0)

def test_evaluate_retrieval_with_incomplete_trace():
    relevant_documents = ["DOC_A"]

    retrieval_documents = [
        {
            "document_id": "DOC_A",
            "chunk_id": "CHUNK_A1",
            "rank": 1,
        },
    ]

    incomplete_trace = {}

    with pytest.raises(ValueError):
        evaluate_retrieval(
            relevant_documents,
            retrieval_documents,
            k=1,
            trace=incomplete_trace,
        )

