def prompt_extraccion(historial: str) -> str:
    return f"""### ROL
Actúa como un Especialista en Auditoría de Documentación Clínica. Tu misión es analizar notas crudas y generar el término de búsqueda perfecto para consultar una base de datos vectorial CIE-10.

### REGLAS DE EXTRACCIÓN CLÍNICA
1. PROHIBIDO LENGUAJE VAGO: NUNCA uses "Daño", "Problema", "Mal". Usa términos nosológicos específicos ("Desgarro", "Estenosis", "Neoplasia", etc.).
2. ETIOLOGÍA VS SÍNTOMA: Prioriza la causa estructural. Los síntomas (dolor, fiebre) se ignoran si existe una causa raíz identificada.
3. DETALLES OBLIGATORIOS CIE-10: Si el historial lo menciona, debes incluir obligatoriamente:
   - Localización anatómica exacta y Lateralidad (Derecho/Izquierdo/Bilateral).
   - Temporalidad (Agudo/Crónico/Recurrente).
   - Tipo de contacto (Inicial/Sucesivo).

### HISTORIAL DEL PACIENTE
{historial}

### FORMATO DE SALIDA ESTRICTO
Primero, haz un brevísimo análisis de 1 línea.
Después, genera la etiqueta <busqueda_cie10> que contendrá ÚNICAMENTE la frase técnica condensada (máximo 8 palabras) que enviaremos a la base de datos.

Ejemplo de salida:
Análisis: Se describe inflamación del apéndice con signos agudos, sin perforación indicada.
<busqueda_cie10>Apendicitis aguda sin perforación</busqueda_cie10>"""


def prompt_decision(historial: str, candidatos_str: str) -> str:
    return f"""Eres un auditor médico experto.
Historial del paciente: {historial}

Aquí tienes 5 posibles códigos CIE-10 extraídos de nuestro catálogo oficial:
{candidatos_str}

Analiza las opciones y elige el código exacto que mejor describe la condición del paciente.
Debes responder ÚNICAMENTE con un objeto JSON válido, usando exactamente esta estructura:
{{
    "CIE10": "código elegido",
    "Justificación": "breve justificación clínica"
}}"""
