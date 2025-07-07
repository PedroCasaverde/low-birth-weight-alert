import pandas as pd


class JoinProcessor:
    """
    Clase para Unir todos los módulos necesarios
    """

    def __init__(self, dfs: dict[str, "pd.DataFrame"], anio: str):
        self.dfs = dfs
        self.anio = anio

    def combinar_modulos(self, dfs: dict[str, "pd.DataFrame"], anio: str) -> "pd.DataFrame":
        """
        Une y enriquece los distintos módulos de la ENDES en un único DataFrame.

        Parámetros
        ----------
        dfs : dict[str, pd.DataFrame]
            Diccionario con las tablas procesadas.  Se esperan las claves:
            {'mod_1', 'mod_2', 'mod_3', 'mod_4', 'mod_5',
            'mod_6', 'mod_7', 'mod_8', 'mod_9', 'mod_13'}

        Devuelve
        --------
        pd.DataFrame
            DataFrame final con todos los merges aplicados.
        """

        # ── Copias locales ─────────────────────────────────────────────────────────
        df_mod_1 = dfs["mod_1"].copy()
        df_mod_2 = dfs["mod_2"].copy()
        df_mod_3 = dfs["mod_3"].copy()
        df_mod_4 = dfs["mod_4"].copy()
        df_mod_5 = dfs["mod_5"].copy()
        df_mod_6 = dfs["mod_6"].copy()
        df_mod_7 = dfs["mod_7"].copy()
        df_mod_9 = dfs["mod_9"].copy()
        df_mod_13 = dfs["mod_13"].copy()

        # ── 1. Datos de padre y madre ──────────────────────────────────────────────
        df_padre = df_mod_1.loc[
            df_mod_1["parentesco_jefe_hogar"] == 1,
            [
                "id_hogar",
                "edad_anios",
                "nivel_educativo",
                "ubigeo",
                "latitud",
                "longitud",
                "altitud_metros",
            ],
        ].rename(columns={"edad_anios": "edad_padre", "nivel_educativo": "educacion_padre"})

        df_madre = df_mod_1.loc[
            df_mod_1["parentesco_jefe_hogar"] == 2,
            ["id_hogar", "edad_anios", "nivel_educativo"],
        ].rename(columns={"edad_anios": "edad_madre", "nivel_educativo": "educacion_madre"})

        # ── 2. Nacimiento + niño ──────────────────────────────────────────────────
        df_unido = df_mod_4.merge(
            df_mod_1[["id_hogar", "id_miembro_hogar", "edad_anios", "nivel_educativo"]],
            on=["id_hogar", "id_miembro_hogar"],
            how="left",
        ).rename(columns={"edad_anios": "edad_nino", "nivel_educativo": "educacion_nino"})

        df_unido = df_unido.merge(df_padre, on="id_hogar", how="left")
        df_unido = df_unido.merge(df_madre, on="id_hogar", how="left")

        mapa_educacion = {
            0: "Sin educación",
            1: "Primaria",
            2: "Secundaria",
            3: "Superior",
            4: "Superior",
            5: "Superior",
        }
        for col in ("educacion_padre", "educacion_madre", "educacion_nino"):
            if col in df_unido.columns:
                df_unido[col] = df_unido[col].map(mapa_educacion).fillna(df_unido[col])

        # ── 3. Características del hogar (mód. 2) ─────────────────────────────────
        print("Uniendo con datos de características del hogar (Módulo 2)...")
        df_unido = df_unido.merge(
            df_mod_2[[c for c in df_mod_2.columns if c != "edad_jefe_hogar"]],
            on="id_hogar",
            how="left",
        )

        mapas_hogar = {
            "fuente_agua_beber": {
                11: "Red pública dentro de la vivienda",
                12: "Red pública fuera de la vivienda",
                13: "Pilón/Grifo público",
                21: "Pozo en la vivienda",
                22: "Pozo público",
                41: "Manantial (puquio)",
                43: "Río/acequia/laguna",
                61: "Camión cisterna",
                71: "Agua embotellada",
                96: "Otro",
            },
            "tipo_servicio_higienico": {
                11: "Conectado a red pública",
                12: "Conectado a red pública (fuera)",
                21: "Pozo séptico",
                22: "Letrina ventilada",
                23: "Letrina pozo ciego/negro",
                31: "Río, acequia o canal",
                32: "No hay servicio (campo)",
                96: "Otro",
            },
            "material_piso": {
                11: "Tierra/arena",
                21: "Madera (entablados)",
                31: "Parquet o madera pulida",
                32: "Láminas asfálticas/vinílicos",
                33: "Losetas/terrazos",
                34: "Cemento/ladrillo",
                96: "Otro",
            },
            "combustible_cocina": {
                1: "Electricidad",
                2: "GLP",
                3: "Gas natural",
                4: "Kerosene",
                5: "Carbón vegetal",
                6: "Carbón mineral",
                7: "Leña",
                10: "Residuos agrícolas",
                11: "Bosta/estiércol",
                95: "No cocina",
                96: "Otro",
            },
            "material_paredes": {
                21: "Adobe o tapia",
                23: "Quincha (caña con barro)",
                31: "Ladrillo/bloques de cemento",
                32: "Piedra con cemento",
                41: "Estera",
                96: "Otro",
            },
            "material_techo": {
                31: "Concreto armado",
                33: "Tejas",
                34: "Plancha de calamina/fibra",
                11: "Paja/hojas de palmera",
                21: "Caña o estera con barro",
                96: "Otro",
            },
        }

        print("Traduciendo códigos a valores descriptivos...")
        for col, mapa in mapas_hogar.items():
            if col in df_unido.columns:
                df_unido[col] = (
                    pd.to_numeric(df_unido[col], errors="coerce").map(mapa).fillna(df_unido[col])
                )

        # ── 4. Módulo 3 (caract. mujer) ───────────────────────────────────────────
        cols_drop_mod3 = {
            "edad_mujer",
            "area_residencia_mujer",
            "nivel_educativo_mujer",
            "religion_mujer",
        }
        df_unido = df_unido.merge(
            df_mod_3.drop(columns=[c for c in cols_drop_mod3 if c in df_mod_3.columns]),
            on=["id_cuestionario_mujer", "id_hogar"],
            how="inner",
        )

        mapa_etnicidad = {
            "1": "Quechua",
            "2": "Aimara",
            "3": "Nativo o indígena de la Amazonía",
            "4": "Parte de otro pueblo indígena u originario",
            "5": "Negro/Moreno/Zambo/Mulato/Pueblo afroperuano o afrodescendiente",
            "6": "Blanco",
            "7": "Mestizo",
            "8": "Otro",
            "98": "No sabe / No responde",
        }
        mapa_idioma = {
            "1": "Quechua",
            "2": "Aimara",
            "3": "Ashaninka",
            "4": "Awajún/Aguaruna",
            "5": "Shipibo/Konibo",
            "6": "Shawi/Chayahuita",
            "7": "Matsigenka/Machiguenga",
            "8": "Achuar",
            "9": "Otra lengua nativa u originaria",
            "10": "Castellano",
            "11": "Portugués",
            "12": "Otra lengua extranjera",
        }

        if "etnicidad_mujer" in df_unido.columns:
            df_unido["etnicidad_mujer"] = (
                df_unido["etnicidad_mujer"].map(mapa_etnicidad).fillna(df_unido["etnicidad_mujer"])
            )
        if "idioma_materno" in df_unido.columns:
            df_unido["idioma_materno"] = (
                df_unido["idioma_materno"].map(mapa_idioma).fillna(df_unido["idioma_materno"])
            )

        df_unido.rename(
            columns={
                "etnicidad_mujer": "etnicidad_madre",
                "idioma_materno": "idioma_madre",
            },
            inplace=True,
        )

        # ── 5. Módulo 5 (prenatal y parto) ────────────────────────────────────────
        columnas_mod_5 = [
            "id_hogar",
            "id_cuestionario_mujer",
            "id_nacimiento",
            "controles_prenatales_num",
            "lugar_parto",
            "parto_fue_cesarea",
            "consumio_suplemento_hierro_embarazo",
            "bebe_tuvo_control_medico_1er_mes",
            "peso_bebe_nacimiento_gr",
            "madre_afiliada_sis_embarazo",
            "complicacion_parto_sangrado_excesivo",
            "complicacion_postparto_sangrado_intenso",
        ]
        df_mod_5_sel = df_mod_5[[c for c in columnas_mod_5 if c in df_mod_5.columns]]

        print("Uniendo con datos prenatales y de parto (Módulo 5)...")
        df_unido = df_unido.merge(
            df_mod_5_sel,
            on=["id_hogar", "id_cuestionario_mujer", "id_nacimiento"],
            how="inner",
        )

        mapa_lugar_parto = {
            11: "Su domicilio",
            12: "Casa de la partera",
            21: "Hospital MINSA",
            22: "Hospital ESSALUD",
            23: "Hospital FF.AA./PNP",
            24: "Centro de salud MINSA",
            25: "Puesto de salud MINSA",
            26: "Centro/Posta ESSALUD",
            27: "Hospital/Otro Municipal",
            31: "Clínica privada",
            32: "Consultorio médico privado",
            41: "Clínica/Posta ONG",
            42: "Hospital/Otro de la Iglesia",
            96: "Otro",
        }
        if "lugar_parto" in df_unido.columns:
            df_unido["lugar_parto"] = (
                df_unido["lugar_parto"].map(mapa_lugar_parto).fillna(df_unido["lugar_parto"])
            )

        print("\nUnión con Módulo 5 y mapeo completados.")

        # ── 6. Resto de módulos ───────────────────────────────────────────────────
        df_unido = df_unido.merge(
            df_mod_6,
            on=["id_cuestionario_mujer", "id_hogar", "id_nacimiento"],
            how="inner",
        )
        df_unido = df_unido.merge(df_mod_7, on=["id_cuestionario_mujer", "id_hogar"], how="inner")
        df_unido = df_unido.merge(df_mod_9, on=["id_cuestionario_mujer", "id_hogar"], how="inner")
        df_unido = df_unido.merge(df_mod_13, on="id_hogar", how="inner")
        df_unido["anio"] = anio
        df_unido.peso_bebe_nacimiento_gr = df_unido.peso_bebe_nacimiento_gr / 1000

        print("shape antes de eliminar duplicados: ", df_unido.shape)
        id_cols = {
            "id_hogar",
            "id_miembro_hogar",
            "id_cuestionario_mujer",
            "id_nacimiento",
            "ubigeo",
        }
        # Nos aseguramos de que existan en el DF
        id_cols = id_cols & set(df_unido.columns)

        df_sin_ids = df_unido.drop(columns=id_cols, errors="ignore")
        df_unido = df_unido.loc[~df_sin_ids.duplicated(keep="first")].reset_index(drop=True)
        print("shape después  de eliminar duplicados: ", df_unido.shape)
        return df_unido
