# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Requisitos previos

- **Docker** corriendo con Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
- **Variable de entorno** `GEMINI_API_KEY` definida, o archivo `.env` en la raíz del proyecto con `GEMINI_API_KEY=tu_clave`
- **Python 3.12+** con `uv` como gestor de entornos

## Comandos

```bash
# Instalar dependencias
uv sync

# Ejecutar el script principal
uv run main.py

# Añadir una dependencia
uv add <paquete>
```

## Arquitectura

Pipeline RAG (Retrieval-Augmented Generation) de un solo archivo (`main.py`) para codificación automática CIE-10:

```
Historial médico (texto libre)
  → [Gemini 2.5 Flash] extrae diagnóstico principal (5 palabras)
  → [gemini-embedding-001] convierte a vector (768 dims, output_dimensionality forzado)
  → [Qdrant] búsqueda coseno → top 5 candidatos del catálogo CIE-10
  → [Gemini 2.5 Flash, temperature=0] decide el código final
  → Devuelve dict: {codigo, justificacion, enfermedad_detectada, candidatos}
```

**Fuente de datos:** `CIE10ES_2026_Finales.xlsx` — columna 0 = código CIE-10, columna 1 = descripción.

**Base de datos vectorial:** Qdrant local en `http://localhost:6333`, colección `cie10_catalogo`. La carga es idempotente: si la colección ya tiene puntos, se salta. Batch size de 100 con pausa de 0.5s entre lotes para respetar límites de la API.

## Modelos Google GenAI

- **Generación:** `gemini-2.5-flash` — usa la API `v1beta` (default del SDK)
- **Embeddings:** `gemini-embedding-001` con `output_dimensionality=768` — también disponible en `v1beta`. **No usar `text-embedding-004`**: ese modelo solo existe en `v1` y el SDK actual lo resuelve como 404.

## Extracción del código CIE-10

`extraer_codigo_cie10(texto)` usa regex `\b[A-Z]\d{2}(?:\.\d+)?\b` para parsear el código de la respuesta libre del LLM. Si el LLM no incluye el código en formato estándar, devuelve `None`.
