import pandas as pd
import os
import numpy as np
from functools import reduce




class EndesProcessor:
    """
    Clase para procesar todos los módulos de la encuesta ENDES.
    """
    def __init__(self, data_path: str, anio: str):
        self.data_path = data_path
        self.anio = anio

    def procesar_modulo_1(self, data_path: str, anio: str) -> pd.DataFrame:
        """
        Crea el DataFrame base del Hogar y sus Miembros para la encuesta ENDES.

        Arregla el KeyError al usar el nombre de variable
        correcto ('HHO') para el ID de miembro en el módulo RECH4.

        Args:
            data_path (str): La ruta a la carpeta que contiene los archivos CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame consolidado con la información de cada persona y su hogar.
        """
        print(f"Iniciando el procesamiento del Módulo 1 (RECH0, RECH1, RECH4) para el año {anio}...")

        paths = {
            'rech0': os.path.join(data_path, f'RECH0_{anio}.csv'),
            'rech1': os.path.join(data_path, f'RECH1_{anio}.csv'),
            'rech4': os.path.join(data_path, f'RECH4_{anio}.csv')
        }

        id_dtypes = {
            'HHID': str, 'UBIGEO': str, 'HV001': str, 'HV002': str,
            'HVIDX': str, 'HHO': str 
        }

        rech0_vars = {
            'HHID': 'id_hogar', 'HV001': 'id_conglomerado', 'HV002': 'id_vivienda',
            'HV005': 'factor_ponderacion_hogar', 'HV024': 'region', 'HV025': 'area_residencia',
            'HV040': 'altitud_metros', 'UBIGEO': 'ubigeo', 'LATITUDY': 'latitud', 'LONGITUDX': 'longitud'
        }
        rech1_vars = {
            'HHID': 'id_hogar', 'HVIDX': 'id_miembro_hogar', 'HV101': 'parentesco_jefe_hogar',
            'HV104': 'sexo', 'HV105': 'edad_anios', 'HV109': 'nivel_educativo'
        }

        rech4_vars = {
            'HHID': 'id_hogar',
            'HHO': 'id_miembro_hogar', 
            'SH11A': 'seguro_essalud',
            'SH11C': 'seguro_sis'
        }

    
        try:
            df_hogar = pd.read_csv(paths['rech0'], usecols=rech0_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_hogar = df_hogar.rename(columns=rech0_vars)
            df_hogar = df_hogar.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        except Exception as e:
            print(f"Error cargando Módulo RECH0: {e}. Proceso detenido.")
            return pd.DataFrame()

        # Cargar RECH1 (Miembros del Hogar)
        try:
            df_miembros = pd.read_csv(paths['rech1'], usecols=rech1_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_miembros = df_miembros.rename(columns=rech1_vars)
            df_miembros = df_miembros.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        except Exception as e:
            print(f"Error cargando Módulo RECH1: {e}. Proceso detenido.")
            return pd.DataFrame()
            
        # Cargar RECH4 (Seguros de Salud)
        try:
            df_salud = pd.read_csv(paths['rech4'], usecols=rech4_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_salud = df_salud.rename(columns=rech4_vars)
            df_salud = df_salud.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        except Exception as e:
            print(f"Advertencia: No se pudo procesar RECH4. Razón: {e}. Se continuará sin datos de seguros.")
            df_salud = pd.DataFrame()

        # --- Unir los DataFrames ---
        
        print("Uniendo dataframes del Módulo 1...")
        # 1. Unir la lista de miembros con los datos de su hogar
        df_consolidado = pd.merge(
            left=df_miembros,
            right=df_hogar,
            on='id_hogar',
            how='left'
        )

        # 2. Unir los datos de seguros de salud si el dataframe df_salud no está vacío
        if not df_salud.empty:
            df_consolidado = pd.merge(
                left=df_consolidado,
                right=df_salud,
                on=['id_hogar', 'id_miembro_hogar'],
                how='left'
            )

        print(f"Proceso del Módulo 1 completado. DataFrame con {df_consolidado.shape[0]} filas y {df_consolidado.shape[1]} columnas.")
        return df_consolidado


    def procesar_modulo_2(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa datos del módulo RECH23 de la encuesta ENDES.

        Este módulo contiene las características detalladas de la vivienda, servicios
        básicos, activos del hogar y el índice de riqueza. La información es a
        nivel de hogar.

        Args:
            data_path (str): La ruta a la carpeta que contiene el archivo CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame con las características del hogar.
        """
        print(f"Iniciando el procesamiento del Módulo 2 (RECH23) para el año {anio}...")

        # --- Paso 1: Definir ruta, llaves y mapeo de variables ---
        
        file_path = os.path.join(data_path, f'RECH23_{anio}.csv')

        # Diccionario de IDs para forzar su lectura como string.
        id_dtypes = {
            'HHID': str
        }

        # Mapeo para RECH23 (Características de la Vivienda y Riqueza)
        rech23_vars = {
            'HHID': 'id_hogar',
            'HV201': 'fuente_agua_beber',          # Fuente principal de abastecimiento de agua
            'HV205': 'tipo_servicio_higienico',     # Tipo de servicio higiénico
            'HV206': 'tiene_electricidad',          # ¿Tiene electricidad?
            'HV208': 'tiene_tv',                    # ¿Su hogar tiene televisor?
            'HV209': 'tiene_refrigeradora',         # ¿Su hogar tiene refrigeradora/congeladora?
            'HV213': 'material_piso',               # Material predominante del piso
            'HV214': 'material_paredes',            # Material predominante de las paredes
            'HV215': 'material_techo',              # Material predominante del techo
            'HV216': 'num_habitaciones_dormir',     # Número de habitaciones utilizadas para dormir
            'HV220': 'edad_jefe_hogar',             # Edad del jefe de hogar
            'HV226': 'combustible_cocina',          # Combustible que utilizan para cocinar
            'HV244': 'tiene_terreno_agricola',      # ¿Algún miembro es dueño de tierras agrícolas?
            'HV246': 'tiene_ganado',                # ¿Algún miembro es dueño de ganadería?
            'HV270': 'quintil_riqueza',             # Índice de riqueza en quintiles (1:Más pobre - 5:Más rico)
            'HV271': 'puntaje_riqueza'              # Factor de puntuación del índice de riqueza
        }


        # --- Paso 2: Cargar y procesar el archivo CSV ---
        print(f"Cargando {file_path}...")
        try:
            # Leer solo las columnas disponibles en el archivo para evitar errores
            available_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
            cols_to_load = [col for col in rech23_vars.keys() if col in available_cols]
            
            df_hogar_caracteristicas = pd.read_csv(
                file_path,
                usecols=cols_to_load,
                dtype=id_dtypes,
                low_memory=False
            )
            
            rename_map = {k: v for k, v in rech23_vars.items() if k in cols_to_load}
            df_hogar_caracteristicas = df_hogar_caracteristicas.rename(columns=rename_map)
            df_hogar_caracteristicas = df_hogar_caracteristicas.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            print(f"Proceso del Módulo 2 completado. DataFrame con {df_hogar_caracteristicas.shape[0]} filas y {df_hogar_caracteristicas.shape[1]} columnas.")
            return df_hogar_caracteristicas

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {file_path}. Por favor, verifica la ruta.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Ocurrió un error al procesar el módulo RECH23: {e}")
            return pd.DataFrame()


    def procesar_modulo_3(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa los datos de los módulos REC0111 y REC91 de la encuesta ENDES.

        Aplica la limpieza de espacios (strip) de forma correcta
        y eficiente en ambos DataFrames antes de la unión para evitar errores de merge.
        """
        print(f"Iniciando el procesamiento del Módulo 3 (REC0111, REC91) para el año {anio}...")

        paths = {
            'rec0111': os.path.join(data_path, f'REC0111_{anio}.csv'),
            'rec91': os.path.join(data_path, f'REC91_{anio}.csv')
        }
        id_dtypes = {'CASEID': str, 'HHID': str}

        # Mapeos de variables corregidos
        rec0111_vars = {
            'CASEID': 'id_cuestionario_mujer', 'HHID': 'id_hogar', 'V012': 'edad_mujer',
            'V102': 'area_residencia_mujer', 'V106': 'nivel_educativo_mujer', 'V130': 'religion_mujer',
            'V157': 'frecuencia_lee_periodico', 'V158': 'frecuencia_escucha_radio',
            'V159': 'frecuencia_ve_tv', 'V190': 'quintil_riqueza_mujer'
            # Se eliminó V131 para evitar confusión.
        }
        rec91_vars = {
            'CASEID': 'id_cuestionario_mujer', 
            'S119': 'idioma_materno',
            'S119D': 'etnicidad_mujer', # <-- Se añadió la variable correcta de etnicidad aquí
            'S239C': 'tiene_dni',
            # Nombres de variables de anemia corregidos para no tener espacios
            'Q1479E A': 'conoce_sintoma_anemia_cansancio',
            'Q1479F A': 'conoce_alimento_anemia_carnes_rojas'
        }

        # Cargar y limpiar REC0111
        try:
            df_rec0111 = pd.read_csv(paths['rec0111'], usecols=rec0111_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_rec0111 = df_rec0111.rename(columns=rec0111_vars)
            for col in df_rec0111.select_dtypes(include=['object']).columns:
                df_rec0111[col] = df_rec0111[col].str.strip()
        except Exception as e:
            print(f"Error cargando Módulo REC0111: {e}. Proceso detenido.")
            return pd.DataFrame()

        # Cargar y limpiar REC91
        df_rec91 = pd.DataFrame()
        try:
            # Lógica para manejar nombres de columna con espacios
            temp_df = pd.read_csv(paths['rec91'], nrows=0)
            temp_df.columns = temp_df.columns.str.replace(' ', '_')
            rec91_vars_adapted = {k.replace(' ', '_'): v for k, v in rec91_vars.items()}
            cols_to_load_original = [col for col in rec91_vars.keys() if col.replace(' ', '_') in temp_df.columns]
            
            df_rec91 = pd.read_csv(paths['rec91'], usecols=cols_to_load_original, dtype=id_dtypes, low_memory=False)
            df_rec91.columns = df_rec91.columns.str.replace(' ', '_')
            df_rec91 = df_rec91.rename(columns=rec91_vars_adapted)

            for col in df_rec91.select_dtypes(include=['object']).columns:
                df_rec91[col] = df_rec91[col].str.strip()
        except Exception as e:
            print(f"Advertencia: No se pudo procesar REC91. Razón: {e}.")

        # Unir los DataFrames
        print("Uniendo dataframes del Módulo 3...")
        if not df_rec91.empty:
            df_consolidado = pd.merge(df_rec0111, df_rec91, on='id_cuestionario_mujer', how='left')
        else:
            df_consolidado = df_rec0111

        print(f"Proceso del Módulo 3 completado. DataFrame con {df_consolidado.shape[0]} filas y {df_consolidado.shape[1]} columnas.")
        return df_consolidado


    def procesar_modulo_4(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa los datos de historia reproductiva de los módulos RE223132 y REC21.

        Limpia los espacios en blanco de forma eficiente y
        prepara el DataFrame para ser unido con una tabla de enlace externa, en lugar
        de derivar el 'id_hogar'.

        Args:
            data_path (str): La ruta a la carpeta que contiene los archivos CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame consolidado con la información de cada nacimiento.
        """
        print(f"Iniciando el reprocesamiento del Módulo 4 (RE223132, REC21) para el año {anio}...")

        # --- Paso 1: Definir rutas, llaves y mapeo de variables ---

        paths = {
            're223132': os.path.join(data_path, f'RE223132_{anio}.csv'),
            'rec21': os.path.join(data_path, f'REC21_{anio}.csv')
        }
        id_dtypes = {'CASEID': str, 'B16': str}

        re223132_vars = {
            'CASEID': 'id_cuestionario_mujer',
            'V201': 'total_hijos_nacidos',
            'V212': 'edad_mujer_primer_parto',
            'V213': 'mujer_actualmente_embarazada',
            'V302': 'mujer_uso_anticonceptivo_alguna_vez'
        }
        rec21_vars = {
            'CASEID': 'id_cuestionario_mujer',
            'BIDX': 'id_nacimiento',
            'B4': 'sexo_bebe',
            'B5': 'bebe_esta_vivo',
            'B11': 'intervalo_nacimiento_anterior_meses',
            'B16': 'id_miembro_hogar'
        }

        # --- Paso 2: Cargar y procesar cada módulo ---

        # Cargar RE223132 
        try:
            df_repro_summary = pd.read_csv(paths['re223132'], usecols=re223132_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_repro_summary = df_repro_summary.rename(columns=re223132_vars)
            

            for col in df_repro_summary.select_dtypes(include=['object']).columns:
                df_repro_summary[col] = df_repro_summary[col].str.strip()
                
        except Exception as e:
            print(f"Advertencia: No se pudo procesar RE223132. Razón: {e}. Se continuará sin estos datos.")
            df_repro_summary = pd.DataFrame()

        # Cargar REC21 (Listado de Nacimientos)
        try:
            df_birth_roster = pd.read_csv(paths['rec21'], usecols=rec21_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_birth_roster = df_birth_roster.rename(columns=rec21_vars)


            for col in df_birth_roster.select_dtypes(include=['object']).columns:
                df_birth_roster[col] = df_birth_roster[col].str.strip()
                
        except Exception as e:
            print(f"Error cargando Módulo REC21: {e}. Este módulo es esencial, el proceso no puede continuar.")
            return pd.DataFrame()

        # --- Paso 3: Unir los DataFrames del Módulo ---
        # La unión se hace solo por la llave de la mujer. El id_hogar se añadirá después.

        print("Uniendo dataframes del Módulo 4...")
        if not df_repro_summary.empty:
            df_consolidado = pd.merge(
                left=df_birth_roster,
                right=df_repro_summary,
                on='id_cuestionario_mujer',
                how='left'
            )
        else:
            df_consolidado = df_birth_roster
        df_consolidado['id_hogar'] = df_consolidado['id_cuestionario_mujer'].str[:9]
        print(f"Proceso del Módulo 4 completado. DataFrame con {df_consolidado.shape[0]} filas y {df_consolidado.shape[1]} columnas.")
        return df_consolidado


    def procesar_modulo_5(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa los datos de los módulos REC41 y REC94 (Salud Prenatal y Parto).

        No deriva 'id_hogar'. Limpia los espacios en blanco
        de forma eficiente y prepara el DataFrame para ser unido con una tabla de enlace.

        Args:
            data_path (str): La ruta a la carpeta que contiene los archivos CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame consolidado a nivel de nacimiento.
        """
        print(f"Iniciando el reprocesamiento del Módulo 5 (REC41, REC94) para el año {anio}...")

        # --- Paso 1: Definir rutas, llaves y mapeo ---
        paths = {
            'rec41': os.path.join(data_path, f'REC41_{anio}.csv'),
            'rec94': os.path.join(data_path, f'REC94_{anio}.csv')
        }
        id_dtypes = {'CASEID': str}

        # Mapeos de variables (sin cambios)
        rec41_vars = {
            'CASEID': 'id_cuestionario_mujer', 'MIDX': 'id_nacimiento', 'M14': 'controles_prenatales_num',
            'M15': 'lugar_parto', 'M17': 'parto_fue_cesarea', 'M19': 'peso_bebe_nacimiento_kg',
            'M45': 'consumio_suplemento_hierro_embarazo', 'M70': 'bebe_tuvo_control_medico_1er_mes'
        }
        rec94_vars = {
            'CASEID': 'id_cuestionario_mujer', 'IDX94': 'id_nacimiento', 'S413': 'madre_afiliada_sis_embarazo',
            'S426GB': 'complicacion_parto_sangrado_excesivo', 'S427DA': 'complicacion_postparto_sangrado_intenso',
            'Q1422A A': 'se_hizo_prueba_anemia_embarazo', 'Q1422A B': 'dx_anemia_embarazo',
            'Q1422A C': 'recibio_tratamiento_hierro_embarazo', 'Q1422A D': 'cumplio_tratamiento_hierro_embarazo'
        }

        # --- Paso 2: Cargar y procesar cada módulo ---

        # Procesar REC41
        try:
            df_rec41 = pd.read_csv(paths['rec41'], usecols=rec41_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_rec41 = df_rec41.rename(columns=rec41_vars)
            
            # Limpieza de espacios (forma correcta y eficiente)
            for col in df_rec41.select_dtypes(include=['object']).columns:
                df_rec41[col] = df_rec41[col].str.strip()
                
            if 'peso_bebe_nacimiento_kg' in df_rec41.columns:
                df_rec41['peso_bebe_nacimiento_gr'] = df_rec41['peso_bebe_nacimiento_kg'] * 1000
                df_rec41 = df_rec41.drop(columns=['peso_bebe_nacimiento_kg'])
                
        except Exception as e:
            print(f"Error cargando Módulo REC41: {e}. Proceso detenido.")
            return pd.DataFrame()

        # Procesar REC94
        df_rec94 = pd.DataFrame()
        try:
            # Lógica para manejar nombres de columna con espacios
            temp_df = pd.read_csv(paths['rec94'], nrows=0)
            temp_df.columns = temp_df.columns.str.replace(' ', '_')
            rec94_vars_adapted = {k.replace(' ', '_'): v for k, v in rec94_vars.items()}
            
            cols_to_load_original = [col for col in rec94_vars.keys() if col.replace(' ', '_') in temp_df.columns]
            
            df_rec94_raw = pd.read_csv(paths['rec94'], usecols=cols_to_load_original, dtype=id_dtypes, low_memory=False)
            df_rec94_raw.columns = df_rec94_raw.columns.str.replace(' ', '_')
            
            rename_map = {k.replace(' ', '_'): v for k,v in rec94_vars.items()}
            df_rec94 = df_rec94_raw.rename(columns=rename_map)
            
            # Limpieza de espacios
            for col in df_rec94.select_dtypes(include=['object']).columns:
                df_rec94[col] = df_rec94[col].str.strip()

        except Exception as e:
            print(f"Advertencia: No se pudo procesar REC94. Razón: {e}. Se continuará sin estos datos.")

        # --- Paso 3: Unir los DataFrames del Módulo ---
        

        if not df_rec94.empty:
            df_final = pd.merge(df_rec41, df_rec94, on=['id_cuestionario_mujer', 'id_nacimiento'], how='left')
        else:
            df_final = df_rec41
        df_final['id_hogar'] = df_final['id_cuestionario_mujer'].str[:9]
        print(f"Módulo 5 reprocesado. DataFrame con {df_final.shape[0]} filas y {df_final.shape[1]} columnas.")
        return df_final


    def procesar_modulo_6(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa datos de salud, vacunación, desarrollo y nutrición.

        No deriva 'id_hogar'. Limpia los datos en cada
        módulo ANTES de unirlos. Prepara el DataFrame para la unión final con
        una tabla de enlace.

        Args:
            data_path (str): La ruta a la carpeta que contiene los archivos CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame consolidado con información de salud materno-infantil.
        """
        print(f"Iniciando el reprocesamiento del Módulo 6 para el año {anio}...")


        paths = {
            'rec43': os.path.join(data_path, f'REC43_{anio}.csv'),
            'rec95': os.path.join(data_path, f'REC95_{anio}.csv'),
            'dit': os.path.join(data_path, f'DIT_{anio}.csv'),
            'rec42': os.path.join(data_path, f'REC42_{anio}.csv')
        }
        id_dtypes = {'CASEID': str}

        rec43_vars = {
            'CASEID': 'id_cuestionario_mujer', 'HIDX': 'id_nacimiento', 'H1': 'tiene_carnet_salud',
            'H9': 'vacuna_sarampion', 'H11': 'tuvo_diarrea_ult_2_semanas',
            'H22': 'tuvo_fiebre_ult_2_semanas', 'H31': 'tuvo_tos_ult_2_semanas', 'H42': 'tomo_suplemento_hierro'
        }
        rec95_vars = {
            'CASEID': 'id_cuestionario_mujer', 'IDX95': 'id_nacimiento', 'S466': 'tuvo_control_cred',
            'S465EA': 'consumio_hierro_jarabe_ult_7_dias', 'S465EB': 'consumio_micronutrientes_ult_7_dias',
            'Q1465ED CC B': 'dx_anemia_bebe' 
        }
        dit_vars = {
            'CASEID': 'id_cuestionario_mujer', 'BIDX': 'id_nacimiento',
            'Q1478': 'edad_meses_eval_desarrollo' 
        }
        rec42_vars = {
            'CASEID': 'id_cuestionario_mujer', 'V437': 'madre_peso_kg', 'V438': 'madre_talla_cm',
            'V445': 'madre_imc', 'V456': 'madre_hemoglobina_ajustada', 'V457': 'madre_nivel_anemia',
            'V414H': 'dieta_nino_carnes_ayer', 'V414G': 'dieta_nino_huevos_ayer'
        }

        # --- Paso 2: Procesar módulos a nivel de niño ---
        dfs_nino = []
        child_modules = [
            ('rec43', paths['rec43'], rec43_vars),
            ('rec95', paths['rec95'], rec95_vars),
            ('dit', paths['dit'], dit_vars)
        ]

        for name, path, var_map in child_modules:
            try:

                temp_df = pd.read_csv(path, nrows=0)
                temp_df.columns = temp_df.columns.str.replace(' ', '_').str.replace('.', '')
                var_map_adapted = {k.replace(' ', '_').replace('.', ''): v for k, v in var_map.items()}
                
                cols_to_load_original = [col for col in var_map.keys() if col.replace(' ', '_').replace('.', '') in temp_df.columns]
                
                df = pd.read_csv(path, usecols=cols_to_load_original, dtype=id_dtypes, low_memory=False)
                df.columns = df.columns.str.replace(' ', '_').str.replace('.', '')
                
                df = df.rename(columns={k.replace(' ', '_').replace('.', ''): v for k,v in var_map.items()})


                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].str.strip()
                
                dfs_nino.append(df)
            except Exception as e:
                print(f"Advertencia: No se pudo procesar el módulo de niño '{name}'. Razón: {e}.")

        # Unir DataFrames de niño
        if not dfs_nino:
            print("Error: No se pudo cargar ningún módulo de datos de niño. Proceso detenido.")
            return pd.DataFrame()
        

        df_nino_consolidado = reduce(lambda left, right: pd.merge(left, right, on=['id_cuestionario_mujer', 'id_nacimiento'], how='outer'), dfs_nino)

        # --- Paso 3: Procesar módulo a nivel de mujer ---
        df_mujer = pd.DataFrame()
        try:
            df_mujer = pd.read_csv(paths['rec42'], usecols=rec42_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_mujer = df_mujer.rename(columns=rec42_vars)
            # Limpiar espacios ANTES de la unión
            for col in df_mujer.select_dtypes(include=['object']).columns:
                df_mujer[col] = df_mujer[col].str.strip()
        except Exception as e:
            print(f"Advertencia: No se pudo procesar el módulo de mujer 'rec42'. Razón: {e}.")

        # --- Paso 4: Unión Final del Módulo ---
        if not df_mujer.empty:
        
            df_final = pd.merge(
                df_nino_consolidado,
                df_mujer,
                on=['id_cuestionario_mujer'],
                how='left'
            )
        else:
            df_final = df_nino_consolidado
        df_final['id_hogar'] = df_final['id_cuestionario_mujer'].str[:9]
        print(f"Módulo 6 reprocesado. DataFrame con {df_final.shape[0]} filas y {df_final.shape[1]} columnas.")
        return df_final


    def procesar_modulo_7(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa datos del módulo RE516171 (Empoderamiento de la Mujer).

        VERSIÓN DEFINITIVA: Aplica la limpieza de espacios en blanco y crea la
        columna 'id_hogar' derivándola de los primeros 9 caracteres del CASEID
        limpio, según el método validado.

        Args:
            data_path (str): La ruta a la carpeta que contiene el archivo CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame con el perfil de empoderamiento de cada mujer.
        """
        print(f"Iniciando el procesamiento del Módulo 7 (RE516171) para el año {anio}...")

        # --- Paso 1: Definir ruta, llaves y mapeo de variables ---

        file_path = os.path.join(data_path, f'RE516171_{anio}.csv')
        id_dtypes = {'CASEID': str}

        re516171_vars = {
            'CASEID': 'id_cuestionario_mujer',
            'V501': 'estado_civil',
            'V511': 'edad_primera_union',
            'V602': 'deseo_mas_hijos',
            'V701': 'educacion_pareja',
            'V714': 'mujer_trabaja_actualmente',
            'V739': 'quien_decide_gastos_mujer',
            'V743A': 'decision_sobre_su_salud',
            'V743B': 'decision_sobre_compras_grandes',
            'V744A': 'justifica_golpear_si_sale_sin_avisar'
        }

        # --- Paso 2: Cargar, limpiar y procesar el archivo CSV ---
        print(f"Cargando {file_path}...")
        try:
            df_modulo_7 = pd.read_csv(
                file_path,
                usecols=re516171_vars.keys(),
                dtype=id_dtypes,
                low_memory=False
            )
            df_modulo_7 = df_modulo_7.rename(columns=re516171_vars)

            # Limpieza de espacios en todas las columnas de texto
            for col in df_modulo_7.select_dtypes(include=['object']).columns:
                df_modulo_7[col] = df_modulo_7[col].str.strip()
            
            # Creación de 'id_hogar' con el método validado
            df_modulo_7['id_hogar'] = df_modulo_7['id_cuestionario_mujer'].str[:9]

            print(f"Módulo 7 procesado. DataFrame con {df_modulo_7.shape[0]} filas y {df_modulo_7.shape[1]} columnas.")
            return df_modulo_7

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {file_path}. Por favor, verifica la ruta.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Ocurrió un error al procesar RE516171: {e}")
            return pd.DataFrame()
        
        
    def procesar_modulo_8(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa datos del módulo RE758081 (VIH/SIDA) y REC82 (Calendario).

        VERSIÓN DEFINITIVA: Aplica la limpieza de espacios en blanco y crea la
        columna 'id_hogar' derivándola de los primeros 9 caracteres del CASEID
        limpio, según el método validado.

        Args:
            data_path (str): La ruta a la carpeta que contiene los archivos CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame consolidado con información sobre VIH y el calendario.
        """
        print(f"Iniciando el procesamiento final del Módulo 8 (RE758081, REC82) para el año {anio}...")

        # --- Paso 1: Definir rutas, llaves y mapeo de variables ---
        paths = {
            're758081': os.path.join(data_path, f'RE758081_{anio}.csv'),
            'rec82': os.path.join(data_path, f'REC82_{anio}.csv')
        }
        id_dtypes = {'CASEID': str}

        # Mapeo para RE758081 (VIH/SIDA y ETS)
        re758081_vars = {
            'CASEID': 'id_cuestionario_mujer',
            'V751': 'ha_oido_hablar_sida',
            'V754BP': 'conoce_riesgo_relaciones_sexuales',
            'V754CP': 'conoce_uso_condon_previene_sida',
            'V781': 'se_hizo_prueba_vih',
            'V825': 'compraria_alimentos_vendedor_con_vih'
        }

        # Mapeo para REC82 (Calendario Reproductivo)
        rec82_vars = {
            'CASEID': 'id_cuestionario_mujer',
            'VCOL': 'calendario_num_columna',
            'VCAL': 'calendario_data_string' # String de 80 caracteres con la historia
        }

        # --- Paso 2: Cargar, limpiar y procesar cada módulo ---
        
        # Procesar RE758081
        try:
            df_re758081 = pd.read_csv(paths['re758081'], usecols=re758081_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_re758081 = df_re758081.rename(columns=re758081_vars)
            for col in df_re758081.select_dtypes(include=['object']).columns:
                df_re758081[col] = df_re758081[col].str.strip()
            df_re758081['id_hogar'] = df_re758081['id_cuestionario_mujer'].str[:9]
        except Exception as e:
            print(f"Advertencia: No se pudo procesar RE758081. Razón: {e}. Se continuará sin estos datos.")
            df_re758081 = pd.DataFrame()

        # Procesar REC82
        try:
            df_rec82 = pd.read_csv(paths['rec82'], usecols=rec82_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_rec82 = df_rec82.rename(columns=rec82_vars)
            for col in df_rec82.select_dtypes(include=['object']).columns:
                df_rec82[col] = df_rec82[col].str.strip()
            df_rec82['id_hogar'] = df_rec82['id_cuestionario_mujer'].str[:9]
        except Exception as e:
            print(f"Advertencia: No se pudo procesar REC82. Razón: {e}. Se continuará sin estos datos.")
            df_rec82 = pd.DataFrame()

        # --- Paso 3: Unir los DataFrames del Módulo ---
        
        # Este módulo es una excepción, ya que REC82 puede tener varias filas por mujer (una por cada columna del calendario)
        # Por lo general, es más útil mantenerlos separados o pivotar REC82 antes de unir.
        # Para este caso, realizaremos una unión simple que puede replicar los datos de RE758081
        # si hay varias filas de calendario por mujer.
        print("Uniendo dataframes del Módulo 8...")
        if not df_re758081.empty and not df_rec82.empty:
            df_final = pd.merge(
                left=df_re758081,
                right=df_rec82,
                on=['id_hogar', 'id_cuestionario_mujer'],
                how='left'
            )
        elif not df_re758081.empty:
            df_final = df_re758081
        elif not df_rec82.empty:
            df_final = df_rec82
        else:
            df_final = pd.DataFrame()

        print(f"Módulo 8 procesado. DataFrame con {df_final.shape[0]} filas y {df_final.shape[1]} columnas.")
        return df_final


    def procesar_modulo_9(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa datos de los módulos REC84DV (Violencia Doméstica) y REC83 (Mortalidad de Hermanas).

        VERSIÓN DEFINITIVA: Aplica la limpieza de espacios en blanco y crea la
        columna 'id_hogar' derivándola de los primeros 9 caracteres del CASEID
        limpio, según el método validado.

        Args:
            data_path (str): La ruta a la carpeta que contiene los archivos CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame consolidado con información sobre violencia y mortalidad de hermanas.
        """
        print(f"Iniciando el procesamiento del Módulo 9 (REC84DV, REC83) para el año {anio}...")

        # --- Paso 1: Definir rutas, llaves y mapeo de variables ---
        paths = {
            'rec84dv': os.path.join(data_path, f'REC84DV_{anio}.csv'),
            'rec83': os.path.join(data_path, f'REC83_{anio}.csv')
        }
        id_dtypes = {'CASEID': str}

        # Mapeo para REC84DV (Violencia Doméstica)
        rec84dv_vars = {
            'CASEID': 'id_cuestionario_mujer',
            'D101A': 'pareja_celoso_si_habla_con_otro_hombre',
            'D101E': 'pareja_insiste_en_saber_donde_va',
            'D103A': 'pareja_la_humilla',
            'D104': 'sufrio_violencia_emocional',
            'D105A': 'sufrio_violencia_fisica_empujon',
            'D105H': 'sufrio_violencia_sexual_forzada',
            'D121': 'padre_golpeo_a_madre'
        }

        # Mapeo para REC83 (Mortalidad de Hermanas).
        # Se seleccionan pocas variables, ya que es a nivel de hermana y no de la entrevistada.
        rec83_vars = {
            'CASEID': 'id_cuestionario_mujer',
            # 'MM2': 'estado_supervivencia_hermana', # Podría usarse para contar hermanas fallecidas.
            # Por simplicidad, solo usamos el ID para verificar si la mujer tiene datos en este módulo.
        }

        # --- Paso 2: Cargar, limpiar y procesar cada módulo ---

        # Procesar REC84DV
        try:
            df_rec84dv = pd.read_csv(paths['rec84dv'], usecols=rec84dv_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_rec84dv = df_rec84dv.rename(columns=rec84dv_vars)
            for col in df_rec84dv.select_dtypes(include=['object']).columns:
                df_rec84dv[col] = df_rec84dv[col].str.strip()
            df_rec84dv['id_hogar'] = df_rec84dv['id_cuestionario_mujer'].str[:9]
        except Exception as e:
            print(f"Advertencia: No se pudo procesar REC84DV. Razón: {e}. Se continuará sin estos datos.")
            df_rec84dv = pd.DataFrame()

        # Procesar REC83
        # Este módulo puede tener varias filas por mujer, una por cada hermana.
        # Lo usaremos para crear un indicador de si la mujer reportó datos en este módulo.
        df_rec83_indicator = pd.DataFrame()
        try:
            df_rec83 = pd.read_csv(paths['rec83'], usecols=['CASEID'], dtype=id_dtypes, low_memory=False)
            df_rec83 = df_rec83.rename(columns={'CASEID': 'id_cuestionario_mujer'})
            for col in df_rec83.select_dtypes(include=['object']).columns:
                df_rec83[col] = df_rec83[col].str.strip()
                
            # Creamos un indicador simple y eliminamos duplicados para tener una fila por mujer
            df_rec83['reporto_mortalidad_hermanas'] = 1
            df_rec83_indicator = df_rec83.drop_duplicates(subset=['id_cuestionario_mujer'])
            
        except Exception as e:
            print(f"Advertencia: No se pudo procesar REC83. Razón: {e}.")

        # --- Paso 3: Unir los DataFrames del Módulo ---
        # Unimos el indicador de REC83 a los datos de violencia de REC84DV
        
        print("Uniendo dataframes del Módulo 9...")
        if not df_rec84dv.empty and not df_rec83_indicator.empty:
            df_final = pd.merge(
                left=df_rec84dv,
                right=df_rec83_indicator,
                on='id_cuestionario_mujer',
                how='left'
            )
            # Llenar con 0 para las mujeres que no reportaron en REC83
            df_final['reporto_mortalidad_hermanas'] = df_final['reporto_mortalidad_hermanas'].fillna(0)
            
        elif not df_rec84dv.empty:
            df_final = df_rec84dv
            df_final['reporto_mortalidad_hermanas'] = 0 # Si no hay datos de REC83, nadie reportó
        else:
            df_final = pd.DataFrame()


        print(f"Módulo 9 procesado. DataFrame con {df_final.shape[0]} filas y {df_final.shape[1]} columnas.")
        return df_final


    def procesar_modulo_10(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa datos de los módulos de antropometría y hemoglobina.

        VERSIÓN DEFINITIVA: Consolida los datos de mujeres (RECH5) y niños (RECH6)
        en un único DataFrame. Estandariza las columnas de ID y de resultados
        para su fácil uso.

        Args:
            data_path (str): La ruta a la carpeta que contiene los archivos CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame consolidado con mediciones antropométricas.
        """
        print(f"Iniciando el procesamiento final del Módulo 10 (RECH5, RECH6) para el año {anio}...")

        # --- Paso 1: Definir rutas, llaves y mapeos ---
        paths = {
            'rech5': os.path.join(data_path, f'RECH5_{anio}.csv'),
            'rech6': os.path.join(data_path, f'RECH6_{anio}.csv')
        }
        id_dtypes = {'HHID': str, 'HAD': str, 'HC0': str}

        # Mapeo para RECH5 (Mujeres)
        rech5_vars = {
            'HHID': 'id_hogar', 'HAD': 'id_miembro_hogar',
            'HA1': 'edad_mujer_medicion', 'HA2': 'peso_kg_mujer', 'HA3': 'talla_cm_mujer',
            'HA40': 'imc_mujer', 'HA53': 'hemoglobina_mujer', 'HA57': 'anemia_nivel_mujer'
        }

        # Mapeo para RECH6 (Niños)
        rech6_vars = {
            'HHID': 'id_hogar', 'HC0': 'id_miembro_hogar',
            'HC1': 'edad_meses_nino', 'HC2': 'peso_kg_nino', 'HC3': 'talla_cm_nino',
            'HC53': 'hemoglobina_nino', 'HC57': 'anemia_nivel_nino'
        }

        # --- Paso 2: Cargar y procesar cada módulo ---

        # Procesar RECH5 (Mujeres)
        try:
            df_mujer = pd.read_csv(paths['rech5'], usecols=rech5_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_mujer = df_mujer.rename(columns=rech5_vars)
            for col in df_mujer.select_dtypes(include=['object']).columns:
                df_mujer[col] = df_mujer[col].str.strip()
        except Exception as e:
            print(f"Advertencia: No se pudo procesar RECH5. Razón: {e}.")
            df_mujer = pd.DataFrame()

        # Procesar RECH6 (Niños)
        try:
            df_nino = pd.read_csv(paths['rech6'], usecols=rech6_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_nino = df_nino.rename(columns=rech6_vars)
            for col in df_nino.select_dtypes(include=['object']).columns:
                df_nino[col] = df_nino[col].str.strip()
        except Exception as e:
            print(f"Advertencia: No se pudo procesar RECH6. Razón: {e}.")
            df_nino = pd.DataFrame()

        # --- Paso 3: Unir los DataFrames del Módulo ---
        print("Uniendo dataframes de antropometría (mujer y niño)...")
        if not df_mujer.empty and not df_nino.empty:
            # La unión externa conserva todas las filas de ambos dataframes
            df_final = pd.merge(
                left=df_mujer,
                right=df_nino,
                on=['id_hogar', 'id_miembro_hogar'],
                how='outer'
            )
        elif not df_mujer.empty:
            df_final = df_mujer
        elif not df_nino.empty:
            df_final = df_nino
        else:
            df_final = pd.DataFrame()

        print(f"Módulo 10 procesado. DataFrame con {df_final.shape[0]} filas y {df_final.shape[1]} columnas.")
        return df_final


    def procesar_modulo_11(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa datos del módulo REC93DVdisciplina (Disciplina Infantil).

        VERSIÓN DEFINITIVA: Extrae variables sobre los métodos de disciplina y la
        exposición del niño a la violencia. Aplica la limpieza de espacios en
        blanco y crea la columna 'id_hogar' derivándola de los primeros 9
        caracteres del CASEID limpio, según el método validado.

        Args:
            data_path (str): La ruta a la carpeta que contiene el archivo CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame con información sobre disciplina por cada niño.
        """
        print(f"Iniciando el procesamiento del Módulo 11 (REC93DVdisciplina) para el año {anio}...")

        # --- Paso 1: Definir ruta, llaves y mapeo de variables ---

        file_path = os.path.join(data_path, f'REC93DVdisciplina_{anio}.csv')
        id_dtypes = {'CASEID': str}

        # Mapeo de variables para REC93DVdisciplina
        rec93_vars = {
            'CASEID': 'id_cuestionario_mujer',
            'Q1035NO': 'id_nacimiento', # N° de orden del niño (historia de nacimiento)
            'Q1036N': 'quien_corrige_al_nino',
            'Q1037M': 'disciplina_madre_explica_conducta', # Se extrae la opción 'K' de esta variable múltiple
            'Q1037P': 'disciplina_padre_castigo_fisico', # Se extrae la opción 'E'
            'Q1040A': 'nino_presente_violencia_empujon',
            'Q1040C': 'nino_presente_violencia_golpe_puno'
        }


        # --- Paso 2: Cargar, limpiar y procesar el archivo CSV ---
        print(f"Cargando {file_path}...")
        try:
            df_modulo_11 = pd.read_csv(
                file_path,
                usecols=rec93_vars.keys(),
                dtype=id_dtypes,
                low_memory=False
            )
            df_modulo_11 = df_modulo_11.rename(columns=rec93_vars)

            # Limpieza de espacios en todas las columnas de texto
            for col in df_modulo_11.select_dtypes(include=['object']).columns:
                df_modulo_11[col] = df_modulo_11[col].str.strip()

            # Las columnas de disciplina son de respuesta múltiple (ej: 'ABC').
            # Creamos columnas binarias para las acciones específicas que nos interesan.
            if 'disciplina_madre_explica_conducta' in df_modulo_11.columns:
                df_modulo_11['disciplina_madre_explica_conducta'] = df_modulo_11['disciplina_madre_explica_conducta'].str.contains('K', na=False).astype(int)
            
            if 'disciplina_padre_castigo_fisico' in df_modulo_11.columns:
                df_modulo_11['disciplina_padre_castigo_fisico'] = df_modulo_11['disciplina_padre_castigo_fisico'].str.contains('E', na=False).astype(int)

            # Creación de 'id_hogar' con el método validado
            df_modulo_11['id_hogar'] = df_modulo_11['id_cuestionario_mujer'].str[:9]

            print(f"Módulo 11 procesado. DataFrame con {df_modulo_11.shape[0]} filas y {df_modulo_11.shape[1]} columnas.")
            return df_modulo_11

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {file_path}. Por favor, verifica la ruta.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Ocurrió un error al procesar REC93DVdisciplina: {e}")
            return pd.DataFrame()
        

    def procesar_modulo_12(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa datos del Cuestionario de Salud (CSALUD01 y CSALUD08).

        VERSIÓN DEFINITIVA: Consolida los datos de salud de adultos (15+ años) y
        niños (0-11 años). Aplica la limpieza de espacios y estandariza las llaves
        para su posterior unión con el resto de los módulos.

        Args:
            data_path (str): La ruta a la carpeta que contiene los archivos CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame consolidado con información de salud.
        """
        print(f"Iniciando el procesamiento del Módulo 12 (CSALUD01, CSALUD08) para el año {anio}...")

        # --- Paso 1: Definir rutas, llaves y mapeos ---
        paths = {
            'csalud01': os.path.join(data_path, f'CSALUD01_{anio}.csv'),
            'csalud08': os.path.join(data_path, f'CSALUD08_{anio}.csv')
        }
        id_dtypes = {'HHID': str, 'QSNUMERO': str, 'QS801': str}

        # Mapeo para CSALUD01 (Salud 15+ años)
        csalud01_vars = {
            'HHID': 'id_hogar', 'QSNUMERO': 'id_miembro_hogar',
            'QS102': 'dx_hipertension',
            'QS109': 'dx_diabetes',
            'QS201': 'fumo_ultimos_30_dias',
            'QS210': 'consumio_alcohol_ultimos_30_dias',
            'QS700A': 'salud_mental_poco_interes', # PHQ-9: Poco interés o placer
            'QS700B': 'salud_mental_deprimido'     # PHQ-9: Sentirse deprimido/triste
        }

        # Mapeo para CSALUD08 (Salud 0-11 años)
        csalud08_vars = {
            'HHID': 'id_hogar', 'QS801': 'id_miembro_hogar',
            'QS803': 'nino_atendido_por_odontologo',
            'QS811': 'nino_cepilla_dientes_veces_dia',
            'QS817': 'nino_evaluado_de_la_vista',
            'QS820': 'nino_dx_problema_vision',
            'QS835': 'nino_golpeado_por_estudiante'
        }

        # --- Paso 2: Cargar, limpiar y procesar cada módulo ---

        # Procesar CSALUD01 (Adultos 15+)
        try:
            df_adultos = pd.read_csv(paths['csalud01'], usecols=csalud01_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_adultos = df_adultos.rename(columns=csalud01_vars)
            for col in df_adultos.select_dtypes(include=['object']).columns:
                df_adultos[col] = df_adultos[col].str.strip()
        except Exception as e:
            print(f"Advertencia: No se pudo procesar CSALUD01. Razón: {e}.")
            df_adultos = pd.DataFrame()

        # Procesar CSALUD08 (Niños 0-11)
        try:
            df_ninos = pd.read_csv(paths['csalud08'], usecols=csalud08_vars.keys(), dtype=id_dtypes, low_memory=False)
            df_ninos = df_ninos.rename(columns=csalud08_vars)
            for col in df_ninos.select_dtypes(include=['object']).columns:
                df_ninos[col] = df_ninos[col].str.strip()
        except Exception as e:
            print(f"Advertencia: No se pudo procesar CSALUD08. Razón: {e}.")
            df_ninos = pd.DataFrame()
            
        # --- Paso 3: Unir los DataFrames del Módulo ---
        print("Uniendo dataframes del Cuestionario de Salud...")
        if not df_adultos.empty and not df_ninos.empty:
            df_final = pd.merge(
                left=df_adultos,
                right=df_ninos,
                on=['id_hogar', 'id_miembro_hogar'],
                how='outer'
            )
        elif not df_adultos.empty:
            df_final = df_adultos
        elif not df_ninos.empty:
            df_final = df_ninos
        else:
            df_final = pd.DataFrame()

        print(f"Módulo 12 procesado. DataFrame con {df_final.shape[0]} filas y {df_final.shape[1]} columnas.")
        return df_final


    def procesar_modulo_13(self,data_path: str, anio: str) -> pd.DataFrame:
        """
        Carga y procesa datos del módulo resumen de Programas Sociales a nivel de hogar.

        VERSIÓN DEFINITIVA: Utiliza el archivo resumen 'Programas Sociales x Hogar'
        para crear variables indicadoras (flags) de la participación de un hogar en
        diferentes programas sociales, lo cual es ideal para modelos de Machine Learning.

        Args:
            data_path (str): La ruta a la carpeta que contiene el archivo CSV.
            anio (str): El año de la encuesta a procesar.

        Returns:
            pd.DataFrame: Un DataFrame a nivel de hogar con indicadores de programas sociales.
        """
        print(f"Iniciando el procesamiento del Módulo 13 (Programas Sociales x Hogar) para el año {anio}...")

        # --- Paso 1: Definir ruta, llaves y mapeo de variables ---
        
        # El nombre del archivo puede variar, usamos un nombre genérico.
        # El usuario debe asegurar que el archivo correcto esté en la carpeta.
        file_path = os.path.join(data_path, f'Programas Sociales x Hogar_{anio}.csv')
        
        id_dtypes = {'HHID': str}

        # Mapeo para el módulo de Programas Sociales a nivel de Hogar
        ps_hogar_vars = {
            'HHID': 'id_hogar',
            'QH91': 'hogar_beneficiario_beca18',
            'QH93': 'hogar_beneficiario_trabaja_peru',
            'QH95': 'hogar_beneficiario_juntos',
            'QH99': 'hogar_beneficiario_pension65',
            'QH101': 'hogar_recibe_vaso_de_leche',
            'QH103': 'hogar_recibe_comedor_popular',
            'QH106': 'hogar_recibe_cuna_mas'
        }

        # --- Paso 2: Cargar, limpiar y procesar el archivo CSV ---
        print(f"Cargando {file_path}...")
        try:
            # Leer solo las columnas disponibles en el archivo para evitar errores
            available_cols = pd.read_csv(file_path, nrows=0, encoding='latin1').columns.tolist()
            cols_to_load = [col for col in ps_hogar_vars.keys() if col in available_cols]
            
            df_programas = pd.read_csv(
                file_path,
                usecols=cols_to_load,
                dtype=id_dtypes,
                low_memory=False,
                encoding='latin1' # Usamos latin1 por si hay caracteres especiales
            )
            
            rename_map = {k: v for k, v in ps_hogar_vars.items() if k in cols_to_load}
            df_programas = df_programas.rename(columns=rename_map)

            # Limpieza de espacios en todas las columnas de texto
            for col in df_programas.select_dtypes(include=['object']).columns:
                df_programas[col] = df_programas[col].str.strip()

            # Convertir las respuestas a un formato numérico (1 para 'Si', 0 para 'No')
            # La encuesta usa código '1' para 'Si' y '2' para 'No'. Lo estandarizamos.
            for col in df_programas.columns:
                if col != 'id_hogar':
                    # Usamos .loc para evitar SettingWithCopyWarning
                    df_programas.loc[:, col] = df_programas[col].apply(lambda x: 1 if x == 1 or x == '1' else 0)

            print(f"Módulo 13 procesado. DataFrame con {df_programas.shape[0]} filas y {df_programas.shape[1]} columnas.")
            return df_programas

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {file_path}. Por favor, verifica la ruta y el nombre del archivo.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Ocurrió un error al procesar el Módulo de Programas Sociales: {e}")
            return pd.DataFrame()
    

