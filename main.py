import os
import pandas as pd
from colorama import init, Fore, Style

from pipeline import codificar_paciente, leer_excel_de_diagnosticos

init(autoreset=True)

if __name__ == "__main__":
    # Para vectorializar el catálogo CIE-10 (solo la primera vez, o si quieres actualizarlo)
    # preparar_base_datos_vectorial('data/CIE10ES_2026_Finales.xlsx')

    listado = leer_excel_de_diagnosticos('data/Pruebas CIE-10_v1.xlsx')

    os.makedirs("output", exist_ok=True)
    filas = []

    print("\n" + Fore.GREEN + Style.BRIGHT + "Listado de diagnósticos extraídos del Excel")
    print(Fore.GREEN + Style.BRIGHT +        "-------------------------------------------\n")

    for i, historial in enumerate(listado, 1):
        print(Fore.CYAN + f"\n[{i}/{len(listado)}] Procesando...")
        resultado = codificar_paciente(historial)

        CIE10  = resultado['resultado'].get('CIE10') or "NO DETECTADO"
        tiempos = resultado["tiempos"]

        filas.append({
            "CIE10":                CIE10,
            "gemini_extraccion_s":  tiempos["gemini_extraccion_s"],
            "busqueda_vectorial_s": tiempos["busqueda_vectorial_s"],
            "gemini_decision_s":    tiempos["gemini_decision_s"],
            "total_s":              tiempos["total_s"],
            "resultado":            resultado["resultado"],
        })
        print(Fore.GREEN + f"  → {CIE10}  ({tiempos['total_s']}s)")

    csv_path = os.path.join("output", "resultados.csv")
    pd.DataFrame(filas).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(Fore.GREEN + Style.BRIGHT + f"\nCSV guardado en: {csv_path}")
