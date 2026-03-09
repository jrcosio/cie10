import json, time
import pandas as pd
from colorama import Fore, Style
from qdrant_client.models import Distance, VectorParams, PointStruct
from google.genai import types

from config import client, qdrant, MODEL_NAME, EMBEDDING_MODEL, COLLECTION_NAME
from prompts import prompt_extraccion, prompt_decision


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def obtener_embeddings(textos: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=textos,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
        )
        return [e.values for e in response.embeddings]
    except Exception as e:
        raise RuntimeError(f"Error al obtener embeddings: {e}") from e


# ---------------------------------------------------------------------------
# Base de datos vectorial
# ---------------------------------------------------------------------------

def _upsert_batch(docs: list[str], metas: list[dict], offset: int) -> None:
    vectores = obtener_embeddings(docs)
    puntos = [
        PointStruct(id=offset + j, vector=v, payload={"descripcion": docs[j], **metas[j]})
        for j, v in enumerate(vectores)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=puntos)


def preparar_base_datos_vectorial(ruta_excel: str) -> None:
    print(Fore.CYAN + "Leyendo Excel y configurando Qdrant...")
    df = pd.read_excel(ruta_excel)
    documentos = df[df.columns[1]].astype(str).tolist()
    metadatos  = [{"codigo": str(cod)} for cod in df[df.columns[0]].tolist()]

    start_index = 0
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    else:
        start_index = qdrant.count(collection_name=COLLECTION_NAME).count
        print(Fore.GREEN + f"¡Qdrant ya tiene {Style.BRIGHT}{start_index}{Style.NORMAL} códigos guardados!")
        if start_index >= len(documentos):
            print(Fore.GREEN + Style.BRIGHT + "El catálogo está 100% completo. Saltando la fase de creación.")
            return

    batch_size = 100
    print(Fore.CYAN + f"Reanudando la subida desde la fila {start_index}...")

    for i in range(start_index, len(documentos), batch_size):
        batch_docs = documentos[i:i + batch_size]
        batch_meta = metadatos[i:i + batch_size]
        try:
            _upsert_batch(batch_docs, batch_meta, i)
        except Exception:
            print(Fore.YELLOW + f"\nError en la fila {i}. Reintentando en 5s...")
            time.sleep(5)
            _upsert_batch(batch_docs, batch_meta, i)  # bug fix: antes faltaba el upsert en el retry
        print(Fore.CYAN + f"Procesadas {Style.BRIGHT}{i + len(batch_docs)}{Style.NORMAL} filas...")
        time.sleep(0.5)

    print(Fore.GREEN + Style.BRIGHT + "\n¡Base de datos vectorial Qdrant lista y completa!")


# ---------------------------------------------------------------------------
# Pipeline RAG
# ---------------------------------------------------------------------------

def medir_tiempo(fn):
    inicio = time.perf_counter()
    resultado = fn()
    return resultado, time.perf_counter() - inicio


def codificar_paciente(historial: str) -> dict:
    # Fase 1: extracción del diagnóstico
    enfermedad, t1 = medir_tiempo(
        lambda: client.models.generate_content(
            model=MODEL_NAME, contents=prompt_extraccion(historial)
        ).text.strip()
    )

    # Fase 2: búsqueda vectorial
    def _busqueda():
        vec = obtener_embeddings([enfermedad], task_type="RETRIEVAL_QUERY")[0]
        return qdrant.query_points(collection_name=COLLECTION_NAME, query=vec, limit=5)

    resultados, t2 = medir_tiempo(_busqueda)

    candidatos_str = "\n".join(
        f"- Código {h.payload['codigo']}: {h.payload['descripcion']} (Confianza: {h.score:.2f})"
        for h in resultados.points
    )

    # Fase 3: decisión final
    decision_texto, t3 = medir_tiempo(
        lambda: client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_decision(historial, candidatos_str),
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
        ).text
    )

    try:
        decision = json.loads(decision_texto)
    except Exception as e:
        decision = {"CIE10": "ERROR", "Justificación": f"Fallo al parsear JSON: {e}"}

    return {
        "resultado": decision,
        "tiempos": {
            "gemini_extraccion_s":  round(t1, 2),
            "busqueda_vectorial_s": round(t2, 2),
            "gemini_decision_s":    round(t3, 2),
            "total_s":              round(t1 + t2 + t3, 2),
        },
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def leer_excel_de_diagnosticos(excel_path: str, columna: int = 1) -> list[str]:
    df = pd.read_excel(excel_path)
    return df[df.columns[columna]].astype(str).tolist()
