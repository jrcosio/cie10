import os, re, time, dotenv
import pandas as pd
from colorama import init, Fore, Style
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google import genai
from google.genai import types

init(autoreset=True)

# Carga variables de entorno desde el archivo .env (en el directorio del script)
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# 1. Configuración inicial
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "Variable de entorno GEMINI_API_KEY no definida. "
        "Ejecútalo con: set GEMINI_API_KEY=tu_clave (Windows) o export GEMINI_API_KEY=tu_clave (Linux/Mac)"
    )
client = genai.Client(api_key=GEMINI_API_KEY)

# Configuraciones
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "cie10_catalogo"

# Inicializar Qdrant (Conectando al Docker local)
qdrant = QdrantClient(url="http://localhost:6333")


def obtener_embeddings_google(textos, task_type="RETRIEVAL_DOCUMENT"):
    """Llama a la API de Google para convertir textos en vectores matemáticos."""
    try:
        response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=textos,
        config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768 # Tamaño del vector de salida
            )
        )
        # Devuelve la lista de vectores generados
        return [e.values for e in response.embeddings]
    except Exception as e:
        raise RuntimeError(f"Error al obtener embeddings de Google: {e}") from e

def preparar_base_datos_vectorial(ruta_excel):
    print(Fore.CYAN + "Leyendo Excel y configurando Qdrant...")
    df = pd.read_excel(ruta_excel)

    col_codigo = df.columns[0]
    col_descripcion = df.columns[1]
    documentos = df[col_descripcion].astype(str).tolist()
    metadatos = [{"codigo": str(cod)} for cod in df[col_codigo].tolist()]

    # 1. LÓGICA DE REANUDACIÓN INTELIGENTE
    start_index = 0
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    else:
        # Si la colección existe, le preguntamos cuántos vectores tiene dentro
        start_index = qdrant.count(collection_name=COLLECTION_NAME).count
        print(Fore.GREEN + f"¡Qdrant ya tiene {Style.BRIGHT}{start_index}{Style.NORMAL} códigos guardados a salvo!")

        # Si ya terminamos antes, salimos de la función
        if start_index >= len(documentos):
            print(Fore.GREEN + Style.BRIGHT + "El catálogo está 100% completo. Saltando la fase de creación.")
            return

    batch_size = 100
    print(Fore.CYAN + f"Reanudando la subida desde la fila {start_index}...")

    # 2. EL BUCLE AHORA EMPIEZA EN 'start_index' EN LUGAR DE 0
    for i in range(start_index, len(documentos), batch_size):
        batch_docs = documentos[i:i+batch_size]
        batch_meta = metadatos[i:i+batch_size]

        # Generar embeddings y preparar puntos (tu código se mantiene igual aquí)
        try:
            vectores = obtener_embeddings_google(batch_docs, task_type="RETRIEVAL_DOCUMENT")

            puntos = []
            for j in range(len(batch_docs)):
                puntos.append(
                    PointStruct(
                        id=i + j,
                        vector=vectores[j],
                        payload={"descripcion": batch_docs[j], **batch_meta[j]}
                    )
                )

            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=puntos
            )
            print(Fore.CYAN + f"Procesadas {Style.BRIGHT}{i + len(batch_docs)}{Style.NORMAL} filas...")
            time.sleep(0.5)

        # Añadimos un pequeño bloque de seguridad por si Google vuelve a fallar
        except Exception as e:
            print(Fore.YELLOW + f"\nError de red con Google en la fila {i}. Pausando 5 segundos antes de reintentar...")
            time.sleep(5)
            # Volvemos a intentar el mismo lote
            vectores = obtener_embeddings_google(batch_docs, task_type="RETRIEVAL_DOCUMENT")
            # (Si quieres que el script sea aún más robusto a fallos, se puede
            # usar la librería 'tenacity' aquí, pero con esto es suficiente por ahora).
            
    print(Fore.GREEN + Style.BRIGHT + "\n¡Base de datos vectorial Qdrant lista y completa!")

def extraer_codigo_cie10(texto):
    """Extrae el primer código CIE-10 del texto (letra + 2 dígitos + opcional .subdígitos)."""
    patron = r'\b[A-Z]\d{2}(?:\.\d+)?\b'
    match = re.search(patron, texto)
    return match.group(0) if match else None


def medir_tiempo(fn):
    """Ejecuta fn(), devuelve (resultado, segundos_transcurridos)."""
    inicio = time.perf_counter()
    resultado = fn()
    return resultado, time.perf_counter() - inicio

def codificar_paciente(historial):
    # FASE 1: Extraer el diagnóstico con el LLM
    prompt_extraccion = f"""
    ### ROL
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
    <busqueda_cie10>Apendicitis aguda sin perforación</busqueda_cie10>
    """
    
    
    enfermedad_extraida, t_gemini1 = medir_tiempo(
        lambda: client.models.generate_content(
            model=MODEL_NAME, contents=prompt_extraccion
        ).text.strip()
    )

    print(f"{Fore.MAGENTA}IA detectó: {Style.BRIGHT}{enfermedad_extraida}")

    # FASE 2: Búsqueda Vectorial (embedding + Qdrant)
    def _busqueda_vectorial():
        vec = obtener_embeddings_google([enfermedad_extraida], task_type="RETRIEVAL_QUERY")[0]
        res = qdrant.query_points(collection_name=COLLECTION_NAME, query=vec, limit=5)
        return vec, res

    (_, resultados), t_busqueda = medir_tiempo(_busqueda_vectorial)

    # Preparamos los candidatos para el LLM
    candidatos_str = ""
    for hit in resultados.points:
        desc = hit.payload['descripcion']
        cod = hit.payload['codigo']
        candidatos_str += f"- Código {cod}: {desc} (Confianza: {hit.score:.2f})\n"

    # FASE 3: El LLM toma la decisión final
    prompt_final = f"""
    Eres un auditor médico experto.
    Historial del paciente: {historial}

    Aquí tienes 5 posibles códigos CIE-10 extraídos de nuestro catálogo oficial:
    {candidatos_str}

    ¿Cuál es el código exacto que mejor describe la condición del paciente?
    Responde con el CÓDIGO y una breve justificación.
    """
    decision_final, t_gemini2 = medir_tiempo(
        lambda: client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_final,
            config=types.GenerateContentConfig(temperature=0.0)
        ).text
    )

    return {
        "resultado": decision_final,
        "tiempos": {
            "gemini_extraccion_s": round(t_gemini1, 2),
            "busqueda_vectorial_s": round(t_busqueda, 2),
            "gemini_decision_s": round(t_gemini2, 2),
            "total_s": round(t_gemini1 + t_busqueda + t_gemini2, 2),
        }
    }

# ==========================================
# EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    # # preparar_base_datos_vectorial('CIE10ES_2026_Finales.xlsx') # Esto lo dejamos comentado
    
    historial = """
    Motivos de la baja: PATOLG CERVICAL DEGR DE AÑOS
    Exploración: ha disminuído el dolor del lado izq , si no hace actividad.
    Evaluación: Refiere   Esta en la misma situación, sigue el dolor menos pero le pasa igual que si hace algún esfuerzo le aumenta el dolor. es menor del lado izq mas del lado derecho , pero si hace esfuerzo se iguala. Próxima cita para ver resultado de la otra infiltración.
    Anotaciones: Motivo: Consulta CC Realización: 04/12/2025 12:07:41 Tareas profesionales: BOMBERO Antigüedad en la empresa: 17 años Antecedentes: Antecedentes personales - Sin interés. Quirúrgicos - Amigdalectomía. Sinus pilonidas. SIN ALERGIAS Médico atención primaria: Antonio De la Toerre Centro de salud: C.S CASTROS Motivo baja: PATOLG CERVICAL DEGR DE AÑOS Evolución: Refiere Esta en la misma situación, sigue el dolor menos pero le pasa igual que si hace algún esfuerzo le aumenta el dolor. es menor del lado izq mas del lado derecho , pero si hace esfuerzo se iguala. Otras patologías: Alergia a acaros, polvo, Anisakis; AP artrosis, Qca amigdalas, sinus pilonidal Traumatologicas Esguince tobillos derecho e Izq. Fuma Niega OL social y /o ocasional. CONSULTA Atención primaria - Últ. cta: 11/11/2025 - Próx. cta: 16/12/2025 Otros: UME - Últ. cta: 20/10/2025 - Próx. cta: 20/01/2026 TRATAMIENTO Atención primaria : Dalingo 1-0-0 (suspendida) Unidad del dolor - Realizada: 20/10/2025 - Programada: 20/01/2026 - PTE Tratamiento SIN PRUEBAS Plan actuación próximo control: Próxima cita para ver resultado de la otra infiltración. Se elabora Atrium con PP cita el 20/02/2026 Fecha prevista alta: 27/02/2026 | Motivo: Consulta CC Realización: 05/11/2025 11:15:24 Tareas profesionales: BOMBERO Antigüedad en la empresa: 17 años Antecedentes: Antecedentes personales - Sin interés. Quirúrgicos - Amigdalectomía. Sinus pilonidas. SIN ALERGIAS Médico atención primaria: Antonio De la Toerre Centro de salud: C.S CASTROS Motivo baja: PATOLG CERVICAL DEGR DE AÑOS Evolución: Refiere que:continúa con el dolor cervical, sin cambios. le han citado de la UDD 20/10/2025, guiada con Eco: le hicieron radiofrecuencia al nervio occipital izq, anestesia , corticoide. Ha suspendido la pregabalina por mejoría. Pendiente cita para abordar el occipital derecho. Ha hecho algo de actividad, y le ha dado dolor. O sea que el dolor continúa. Exploración actual: ha disminuído el dolor del lado izq , si no hace actividad. Otras patologías: Alergia a acaros, polvo, Anisakis; AP artrosis, Qca amigdalas, sinus pilonidal Traumatologicas Esguince tobillos derecho e Izq. Fuma Niega OL social y /o ocasional. CONSULTA Atención primaria - Últ. cta: 07/10/2025 - Próx. cta: 11/11/2025 Otros: UME - Últ. cta: 20/10/2025 - Próx. cta: 20/01/2026 TRATAMIENTO Atención primaria : Dalingo 1-0-0 (suspendida) Unidad del dolor - Realizada: 20/10/2025 - Programada: 20/01/2026 - PTE Tratamiento SIN PRUEBAS Plan actuación próximo control: Proxima cita Fecha prevista alta: 27/02/2026 | Motivo: Consulta CC Realización: 29/09/2025 11:52:28 Tareas profesionales: BOMBERO Antigüedad en la empresa: 17 años Antecedentes: Antecedentes personales - Sin interés. Quirúrgicos - Amigdalectomía. Sinus pilonidas. SIN ALERGIAS Médico atención primaria: Antonio De la Toerre Centro de salud: C.S CASTROS Motivo baja: PATOLG CERVICAL DEGR DE AÑOS Evolución: Refiere que: se encuetrna igual o pero que antes. le van a infiltrar la UDD Exploración actual: Mismo dolor en la misma situación. cervical. Otras patologías: Alergia a acaros, polvo, Anisakis; AP artrosis, Qca amigdalas, sinus pilonidal Traumatologicas Esguince tobillos derecho e Izq. Fuma Niega OL social y /o ocasional. CONSULTA Atención primaria - Últ. cta: 10/09/2025 - Próx. cta: 07/10/2025 Otros: UME - Próx. cta: 20/10/2025 TRATAMIENTO Atención primaria : Dalingo 1-0-0 Unidad del dolor - Programada: 20/10/2025 - PTE Tratamiento SIN PRUEBAS Plan actuación próximo control: Proxima cita para ver evolución. Fecha prevista alta: 30/12/2025 | mismo estado-continua con dolor cabeza CTA U.DOLOR-OCT/25 TTO-Dalingo cta map-2/9 | no contesta- 615287410**********citado p.240d | Motivo: Consulta CC Realización: 16/07/2025 12:36:10 Tareas profesionales: BOMBERO Antigüedad en la empresa: 17 años Antecedentes: Antecedentes personales - Sin interés. Quirúrgicos - Amigdalectomía. Sinus pilonidas. SIN ALERGIAS Médico atención primaria: Antonio De la Toerre Centro de salud: C.S CASTROS Motivo baja: PATOLG CERVICAL DEGR DE AÑOS Evolución: misma situación cta u.dolor***************pte para infiltrar mismo tto-dalingo Otras patologías: Alergia a acaros, polvo, Anisakis; AP artrosis, Qca amigdalas, sinus pilonidal Traumatologicas Esguince tobillos derecho e Izq. Fuma Niega OL social y /o ocasional. CONSULTA Atención primaria - Próx. cta: 16/07/2025 Otros: UME - Próx. cta: 20/10/2025 TRATAMIENTO Unidad del dolor - Programada: 20/10/2025 - PTE Tratamiento SIN PRUEBAS Plan actuación próximo control: cito control telf+p.204d Fecha prevista alta: 31/08/2025 | misma situación cta u.dolor***************pte para infiltrar mismo tto-dalingo | NO contesta************** cta nrc- pauta infiltrac-NO iq pte u.dolor?? cito control telf y cito p.240d | Dejamos controles telefónicos para ver evolución hasta algun avance . | Ha visto por privado a un Neurocirujano Dgco Dsicopatía de columna multisegmentaria, más acentuada en nivel C5-C6 posible Neuralgia de Arnold. Le da recomendaciones : No tto qco. Pendietne vasloración por UD HUMV para infiltración del nervio de Arnold.. limitación los esfuerzos, cargar peso y que son incompatibles con su actividad laboral, y que puede evolucionar aumentando el deterioro funcional con el tiempo. | Motivo: Consulta CC Realización: 11/06/2025 17:37:38 Tareas profesionales: BOMBERO Antigüedad en la empresa: 17 años Antecedentes: Antecedentes personales - Sin interés. Quirúrgicos - Amigdalectomía. Sinus pilonidas. SIN ALERGIAS Médico atención primaria: Antonio De la Toerre Centro de salud: C.S CASTROS Motivo baja: PATOLG CERVICAL DEGR DE AÑOS Evolución: Refiere que: sufre cervicalgia y en los últimos 2 años loe han hecho 2 infiltraciones , por U Musc esqueletica , ya como se le va el efecto de la última le derivan a Udel Dolor HUMV.Le dan medicación Paracetamol, Enantyum. (sin muy buenos resultados). CITA 11/06/2025:: Refiere qeu le han disminuído los dolores de cabeza, pero el resto sin grandes cambios. Le ha visto el Med de AP pero que no le puede derivar a Neurocirujano porque solo sería acomodar la medicación para el dolor . Le queda pendiente la unidad del Dolor que lo tiene para despues del verano. no hay ejercicios ni nad que le pueda ayudar. Exploración actual: idem Otras patologías: Alergia a acaros, polvo, Anisakis; AP artrosis, Qca amigdalas, sinus pilonidal Traumatologicas Esguince tobillos derecho e Izq. Fuma Niega OL social y /o ocasional. CONSULTA Atención primaria - Próx. cta: 15/07/2025 Otros: UME - Próx. cta: 20/10/2025 TRATAMIENTO Unidad del dolor - Programada: 20/10/2025 - PTE Tratamiento SIN PRUEBAS Plan actuación próximo control: Próxima cita para ver evolución. Fecha prevista alta: 31/08/2025 | Acude a cita Refiere que: El med de Ap , decide darle otro medicamento Dalingo ( 1/2 dosis del 165) y luego de 1 mes le auemnta al doble o sea toma la dosis entera. Dice que tiene dias mejores y peores , pero no mejoría del cuadro. Y ayer que empezó con la nueva dosis no ha sentido cambios. Consultó con el Neurocirujano privado : le ha dicho que desaconseja la intervención quirúrgica, y que la neuralgia de Arnold la tiene que llevar , no levantar pesos. Sólo le comentó que le puede fiuncionar la infiltración. No tiene chance sde cambio de puesto por la plaza que tiene. El refiere que este dolor no lo ha tenido antes. el reposo solo le disminuye el gran dolor pero sigue teneindo dolor. constante mantenido. Cita para ver evolución. | Acude a cita Refiere que: cita con U Musculo esquelético, derivan A unidad del dolor., que ya no le puden realizar en Reumatología. Visto el MAP le recetan Triptizol. Y su médiuco el dice que probará condiferentes medicaciones. le sugiero que le consulte por una cita con neurólogo mientras tanto , lo consultará con su MAP. Cita prox 01/04/2025. Continúa con los mismos malestares. y la cita con la unidad del dolor será para 20/10/2025. Proxima cita para ver evolución. | Motivo: Consulta CC Realización: 03/03/2025 16:08:37 Tareas profesionales: BOMBERO Antigüedad en la empresa: 17 años --------------SPS-------------- Médico atención primaria: Antonio De la Toerre Centro de salud: C.S CASTROS --------------SEGUIMIENTO MM-------------- Motivo baja: PATOLG CERVICAL DEGR DE AÑOS Evolución: Refiere que: sufre cervicalgia y en los últimos 2 años loe han hecho 2 infiltraciones , por U Musc esqueletica , ya como se le va el efecto de la última le derivan a Udel Dolor HUMV.Le dan medicación Paracetamol, Enantyum. (sin muy buenos resultados) , se encuenta a la espera de la cita con Unidad del Dolro para una infiltración más profunda. Tiene un RNM 2023 (Cervicoartrosis, sobre todo en el segmento C4-C6 con potencial compormiso foraminal bilaterawl, de predominio derecho, a nivel de C5-C6 y en menor medida de C6-C7).y evolutivos actuales.Le derivan a U del Dolor por ser compatible con Neuralgia de Arnold. Exploración actual: Dolor en la base cervical , de toda la cabeza , tiene limitada la movilida por dolor. Otras patologías: Alergia a acaros, polvo, Anisakis; AP artrosis, Qca amigdalas, sinus pilonidal Traumatologicas Esguince tobillos derecho e Izq. Fuma Niega OL social y /o ocasional. --------------CONSULTA-------------- ATENCIÓN PRIMARIA - Últ. CTA: 18/02/2025 - Próx. CTA: 11/02/2025 ESPECIALISTA * OTROS: UME - Próx. CTA: 11/02/2025 --------------TRATAMIENTO-------------- * UNIDAD DEL DOLOR - PTE Tratamiento --------------PRUEBAS-------------- SIN PRUEBAS --------------OTRAS CUESTIONES-------------- --------------PLAN CONTIGENCIA-------------- Plan actuación próximo control: Plan a espera de que le citren de Unidad del dolor. Fecha prevista alta: 30/04/2025 | Motivo: Consulta CC Realización: 03/03/2025 16:08:37 Tareas profesionales: BOMBERO Antigüedad en la empresa: 17 años --------------SPS-------------- Médico atención primaria: Antonio De la Toerre Centro de salud: C.S CASTROS --------------SEGUIMIENTO MM-------------- Motivo baja: PATOLG CERVICAL DEGR DE AÑOS Evolución: Refiere que: sufre cervicalgia y en los últimos 2 años loe han hecho 2 infiltraciones , por U Musc esqueletica , ya como se le va el efecto de la última le derivan a Udel Dolor HUMV.Le dan medicaciòn Paracetamol, Enantyum. (sin muy buenos resultados) , está esperando que nlo llamen de la misma para hacerle otra infiltración. . Exploración actual: Dolor en la base cervical , de toda la cabeza , tiene limitada la movilida por dolor. Otras patologías: Alergia a acaros, polvo, Anisakis; AP artrosis, Qca amigdalas, sinus pilonidal Traumatologicas Esguince tobillos derecho e Izq. Fuma Niega OL social y /o ocasional. --------------CONSULTA-------------- ATENCIÓN PRIMARIA - Últ. CTA: 18/02/2025 - Próx. CTA: 11/02/2025 ESPECIALISTA * OTROS: UME - Próx. CTA: 11/02/2025 --------------TRATAMIENTO-------------- * UNIDAD DEL DOLOR - PTE Tratamiento --------------PRUEBAS-------------- SIN PRUEBAS --------------OTRAS CUESTIONES-------------- --------------PLAN CONTIGENCIA-------------- Plan actuación próximo control: Plan a espera de que le citren de Unidad del dolor. Fecha prevista alta: 30/04/2025 | Motivo: Consulta CC Realización: 06/02/2025 13:16:50 Tareas profesionales: BOMBERO --------------SPS-------------- Centro de salud: C.S CASTROS --------------SEGUIMIENTO MM-------------- Motivo baja: PATOLG CERVICAL DEGR DE AÑOS DE EVOLUCIÓN Exploración actual: tto-S/P Realizadas RN pte de infiltrac por UME-11/2/25*********** CTA MAP-17/2 --------------CONSULTA-------------- ATENCIÓN PRIMARIA - Próx. CTA: 17/02/2025 ESPECIALISTA * OTROS: UME - Próx. CTA: 11/02/2025 --------------TRATAMIENTO-------------- SIN TRATAMIENTO --------------PRUEBAS-------------- SIN PRUEBAS --------------OTRAS CUESTIONES-------------- --------------PLAN CONTIGENCIA-------------- Plan actuación próximo control: CITO EN MM-CARTA Fecha prevista alta: 07/03/2025 
  
    """
    
    resultado = codificar_paciente(historial)

    print("\n" + Fore.WHITE + Style.BRIGHT + "="*40)
    print(Fore.WHITE + Style.BRIGHT + "        DIAGNÓSTICO CIE-10 FINAL        ")
    print(Fore.WHITE + Style.BRIGHT + "="*40)
    
    print(f"{Fore.GREEN + Style.NORMAL}Código CIE10: {Style.BRIGHT}{extraer_codigo_cie10(resultado['resultado'])}")
    print(Fore.YELLOW + Style.NORMAL + "\nJustificación del LLM")
    print(Fore.YELLOW + "-" * 21)
    print(resultado["resultado"])

    print(Fore.CYAN + "\n--- Informe de rendimiento ---")
    t = resultado["tiempos"]
    print(f" -{Fore.CYAN}Gemini (extracción diagnóstico): {Style.BRIGHT}{t['gemini_extraccion_s']}s")
    print(f" -{Fore.CYAN}Búsqueda vectorial (Qdrant):     {Style.BRIGHT}{t['busqueda_vectorial_s']}s")
    print(f" -{Fore.CYAN}Gemini (decisión final):         {Style.BRIGHT}{t['gemini_decision_s']}s")
    print(f" -{Fore.YELLOW + Style.BRIGHT}TOTAL:                           {t['total_s']}s")
    

    
    