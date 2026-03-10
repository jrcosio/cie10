import os
import pandas as pd
from colorama import init, Fore, Style

from pipeline import codificar_paciente, leer_excel_de_diagnosticos

init(autoreset=True) # Para limpiar los colorinchis después de cada print

if __name__ == "__main__":
    # Para vectorializar el catálogo CIE-10 (solo la primera vez, o si quieres actualizarlo)
    # preparar_base_datos_vectorial('data/CIE10ES_2026_Finales.xlsx')

    listado = leer_excel_de_diagnosticos('data/Pruebas CIE-10_v1.xlsx')

    os.makedirs("output", exist_ok=True)
    csv_path = os.path.join("output", "resultados.csv")
    filas = []
    procesados = set()

    if os.path.exists(csv_path):
        df_prev = pd.read_csv(csv_path, encoding="utf-8-sig")
        filas = df_prev.to_dict("records")
        procesados = set(df_prev["indice"].tolist())
        print(Fore.YELLOW + f"  Reanudando: {len(procesados)} filas ya procesadas.")

    print("\n" + Fore.GREEN + Style.BRIGHT + "Listado de diagnósticos extraídos del Excel")
    print(Fore.GREEN + Style.BRIGHT +        "-------------------------------------------\n")

    try:
        for i, historial in enumerate(listado, 1):
            if i in procesados:
                print(Fore.YELLOW + f"\n[{i}/{len(listado)}] Ya procesado, omitiendo.")
                continue

            print(Fore.CYAN + f"\n[{i}/{len(listado)}] Procesando...")
            resultado = codificar_paciente(historial)

            CIE10  = resultado['resultado'].get('CIE10') or "NO DETECTADO"
            tiempos = resultado["tiempos"]

            filas.append({
                "indice":               i,
                "CIE10":                CIE10,
                "gemini_extraccion_s":  tiempos["gemini_extraccion_s"],
                "busqueda_vectorial_s": tiempos["busqueda_vectorial_s"],
                "gemini_decision_s":    tiempos["gemini_decision_s"],
                "total_s":              tiempos["total_s"],
                "resultado":            resultado["resultado"],
            })
            pd.DataFrame(filas).to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(Fore.GREEN + f"  → {CIE10}  ({tiempos['total_s']}s)")

    except KeyboardInterrupt:
        print(Fore.RED + Style.BRIGHT + "\n\nInterrumpido por el usuario.")

    finally:
        if filas:
            print(Fore.GREEN + Style.BRIGHT + f"\nCSV guardado en: {csv_path}  ({len(filas)} filas)")
            print(Fore.CYAN + Style.BRIGHT + "\nPrimeras 10 filas:\n")
            print(pd.DataFrame(filas)[["indice", "CIE10", "total_s"]].head(10).to_string(index=False))
