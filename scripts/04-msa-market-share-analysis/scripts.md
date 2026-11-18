# 4. SCRIPTS MSA (MARKET SHARE ANALYSIS)

**Nota:** *Los siguientes scripts están diseñados para el análisis de datos orientado a la estimación de market share, la generación de visualizaciones analíticas y la identificación de insights estratégicos a partir de datos históricos obtenidos de TradeMap. Se sugiere implementar este flujo de trabajo en entornos de ejecución Python tales como:*

- [Google Colab](https://colab.research.google.com/)
- [Python-Fiddle](https://python-fiddle.com/)
- [Kaggle Notebooks](https://www.kaggle.com/code)
- [Deepnote](https://deepnote.com/)

## 4.1. Importar e instalar librerías

```
# Instalar librerías si es necesario
!pip install openpyxl

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## 4.2. Cargar archivos fuente (Trademap)
# ============================================================
# 4.2. CARGAR ARCHIVOS FUENTE (TRADEMAP) CON DETECCIÓN FLEXIBLE
# ============================================================
# Funciona con archivos de Trade Map en español o inglés (.xls / .xlsx).
# Si la detección automática no es concluyente, el script pide
# seleccionar las columnas escribiendo su número (índice).
# Para elegir siempre las columnas a mano, cambie a: MODO_MANUAL = True

!pip install -q xlrd   # lector de archivos .xls antiguos

import re
import io

MODO_MANUAL = False

# ---------- 1. Vocabulario para reconocer cabeceras (español / inglés) ----------
PALABRAS_PAIS = ['exportador', 'importador', 'exporter', 'importer', 'socio', 'partner',
                 'pais', 'country', 'countries', 'mercado', 'market', 'proveedor', 'supplier',
                 'origen', 'origin', 'destino', 'destination', 'reporter', 'region']
PALABRAS_VALOR = ['valor', 'value', 'importad', 'exportad', 'imported', 'exported',
                  'comercio', 'trade', 'usd', 'dolar', 'dollar']
PALABRAS_EXCLUIR = ['cantidad', 'quantity', 'unidad', 'unit', 'crecimiento', 'growth',
                    'participacion', 'share', 'tasa', 'rate', 'balanza', 'balance',
                    'precio', 'price', '%', 'ranking', 'distancia', 'distance', 'arancel', 'tariff']
FILAS_TOTAL = {'mundo', 'world', 'total', 'total mundial', 'world total'}
PATRON_ANIO = re.compile(r'(?<!\d)(19[5-9]\d|20\d{2})(?!\d)')


def normalizar(texto):
    """Minúsculas y sin tildes, para comparar cabeceras sin depender de la escritura exacta."""
    t = str(texto).lower().strip()
    for a, b in zip('áéíóúüñ', 'aeiouun'):
        t = t.replace(a, b)
    return re.sub(r'\s+', ' ', t)


def anio_de(cabecera):
    m = PATRON_ANIO.search(str(cabecera))
    return int(m.group(1)) if m else None


def proporcion_numerica(serie):
    s = serie.dropna()
    if len(s) == 0:
        return 0.0
    return pd.to_numeric(limpiar_numeros(s), errors='coerce').notna().mean()


def limpiar_numeros(serie):
    """Quita espacios y separadores de miles (Trade Map exporta enteros en miles USD)."""
    if not pd.api.types.is_numeric_dtype(serie):
        serie = (serie.astype(str).str.replace(' ', '', regex=False)
                 .str.replace(' ', '', regex=False).str.replace(',', '', regex=False))
    return serie


# ---------- 2. Lectura del archivo (.xls, .xlsx o .xls que en realidad es texto/HTML) ----------
def leer_trademap(nombre, contenido=None):
    """Lee el archivo sin asumir cabeceras y luego ubica la fila de encabezado."""
    fuente = io.BytesIO(contenido) if contenido is not None else nombre
    crudo, errores = None, []
    lectores = [
        ('Excel', lambda f: pd.read_excel(f, sheet_name=0, header=None)),
        ('texto tabulado', lambda f: pd.read_csv(f, sep='\t', header=None, encoding='latin-1', dtype=str)),
        ('HTML', lambda f: pd.read_html(f, header=None)[0]),
    ]
    for nombre_lector, lector in lectores:
        try:
            if hasattr(fuente, 'seek'):
                fuente.seek(0)
            crudo = lector(fuente)
            if crudo is not None and crudo.shape[1] >= 2:
                break
            crudo = None
        except Exception as e:
            errores.append(f'{nombre_lector}: {e}')
    if crudo is None:
        raise ValueError('❌ No fue posible leer el archivo. Verifique que sea un .xls o .xlsx '
                         'descargado de Trade Map.\nDetalle: ' + ' | '.join(errores))

    # Ubicar la fila de encabezado: la primera (de las 20 iniciales) con 2 o más años
    fila_cab = 0
    for i in range(min(20, len(crudo))):
        if sum(anio_de(v) is not None for v in crudo.iloc[i].tolist()) >= 2:
            fila_cab = i
            break
    cabeceras, vistos = [], {}
    for j, v in enumerate(crudo.iloc[fila_cab].tolist()):
        nombre_col = str(v).strip() if pd.notna(v) and str(v).strip() else f'Columna_{j}'
        if nombre_col in vistos:                      # evitar nombres repetidos
            vistos[nombre_col] += 1
            nombre_col = f'{nombre_col} ({vistos[nombre_col]})'
        else:
            vistos[nombre_col] = 0
        cabeceras.append(nombre_col)
    datos = crudo.iloc[fila_cab + 1:].reset_index(drop=True)
    datos.columns = cabeceras
    datos = datos.dropna(how='all').reset_index(drop=True)
    return datos


# ---------- 3. Detección automática ----------
def detectar_columna_pais(datos):
    """Puntúa cada columna por su cabecera y por su contenido (texto, valores distintos)."""
    puntajes = {}
    for col in datos.columns:
        if anio_de(col) is not None:
            continue
        cab = normalizar(col)
        serie = datos[col].dropna()
        if len(serie) == 0:
            continue
        texto = 1 - proporcion_numerica(serie)
        unicos = serie.astype(str).nunique() / len(serie)
        puntaje = texto + unicos
        if any(p in cab for p in PALABRAS_PAIS):
            puntaje += 2
        if 'codigo' in cab or 'code' in cab:
            puntaje -= 2
        puntajes[col] = puntaje
    if not puntajes:
        return None, 'no se encontró ninguna columna de texto con nombres de países'
    orden = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)
    mejor, p1 = orden[0]
    if p1 < 2.5:
        return None, f'la mejor candidata ("{mejor}") no tiene una cabecera reconocible de país/socio'
    if len(orden) > 1 and orden[1][1] >= p1 - 0.25:
        return None, f'hay varias candidatas para la columna de países: "{mejor}" y "{orden[1][0]}"'
    return mejor, None


def detectar_columnas_valor(datos):
    """Columnas con año en la cabecera y contenido numérico, excluyendo cantidades, tasas, etc."""
    con_anio = [c for c in datos.columns if anio_de(c) is not None
                and not any(p in normalizar(c) for p in PALABRAS_EXCLUIR)
                and proporcion_numerica(datos[c]) >= 0.5]
    if not con_anio:
        return None, 'no se encontraron columnas con años en la cabecera y valores numéricos'
    con_palabra = [c for c in con_anio if any(p in normalizar(c) for p in PALABRAS_VALOR)]
    candidatas = con_palabra if con_palabra else con_anio
    anios = [anio_de(c) for c in candidatas]
    repetidos = sorted({a for a in anios if anios.count(a) > 1})
    if repetidos:
        return None, (f'hay más de una columna para el/los año(s) {repetidos} '
                      '(por ejemplo, valores importados y exportados en el mismo archivo)')
    return candidatas, None


# ---------- 4. Validación de correspondencias ----------
def validar_columnas(datos, col_pais, cols_valor, anios):
    problemas = []
    if col_pais in cols_valor:
        problemas.append('la columna de países no puede ser también una columna de valores')
    if proporcion_numerica(datos[col_pais]) > 0.5:
        problemas.append(f'"{col_pais}" parece contener números, no nombres de países')
    for c, a in zip(cols_valor, anios):
        if a is None:
            problemas.append(f'no se identificó el año de la columna "{c}"')
        if proporcion_numerica(datos[c]) < 0.5:
            problemas.append(f'"{c}" no contiene valores numéricos suficientes')
    if len(set(anios)) != len(anios):
        problemas.append('hay años repetidos entre las columnas de valores seleccionadas')
    return problemas


# ---------- 5. Asignación manual (por número de columna) ----------
def interpretar_indices(texto, total):
    """Admite '3', '2,3,4' o rangos '2-6'."""
    indices = []
    for parte in str(texto).replace(' ', '').split(','):
        if not parte:
            continue
        if '-' in parte:
            a, b = parte.split('-', 1)
            indices.extend(range(int(a), int(b) + 1))
        else:
            indices.append(int(parte))
    if not indices or any(i < 0 or i >= total for i in indices):
        raise ValueError(f'use números entre 0 y {total - 1}')
    return list(dict.fromkeys(indices))


def seleccion_manual(datos, entrada=input, intentos=3):
    columnas = list(datos.columns)
    print('\n📋 COLUMNAS DISPONIBLES EN EL ARCHIVO')
    print('=' * 60)
    for i, c in enumerate(columnas):
        ejemplo = datos[c].dropna().astype(str).head(2).tolist()
        print(f'  [{i}] {c}   → ej.: {", ".join(ejemplo)}')
    for intento in range(1, intentos + 1):
        try:
            i_pais = interpretar_indices(entrada('\n👉 Número de la columna de PAÍSES / socios: '), len(columnas))
            if len(i_pais) != 1:
                raise ValueError('seleccione una sola columna de países')
            i_val = interpretar_indices(entrada('👉 Números de las columnas de VALORES por año '
                                                '(ej.: 2,3,4 o 2-6): '), len(columnas))
            col_pais = columnas[i_pais[0]]
            cols_valor = [columnas[i] for i in i_val]
            previos = validar_columnas(datos, col_pais, cols_valor, list(range(len(cols_valor))))
            if previos:
                raise ValueError('; '.join(previos))
            anios = [anio_de(c) for c in cols_valor]
            if any(a is None for a in anios):
                faltan = [c for c, a in zip(cols_valor, anios) if a is None]
                txt = entrada(f'👉 No se identificó el año en {faltan}. Escriba los años en el '
                              f'mismo orden (ej.: 2022,2023): ')
                try:
                    nuevos = [int(x) for x in txt.replace(' ', '').split(',') if x]
                except ValueError:
                    raise ValueError('escriba los años como números separados por comas, ej.: 2022,2023')
                if len(nuevos) != len(faltan):
                    raise ValueError(f'se esperaban {len(faltan)} año(s)')
                it = iter(nuevos)
                anios = [a if a is not None else next(it) for a in anios]
            problemas = validar_columnas(datos, col_pais, cols_valor, anios)
            if problemas:
                raise ValueError('; '.join(problemas))
            return col_pais, cols_valor, anios
        except ValueError as e:
            msg = str(e)
            if 'invalid literal' in msg:
                msg = 'escriba solo números de columna, ej.: 3 o 2,3,4 o 2-6'
            print(f'⚠️ Selección no válida ({intento}/{intentos}): {msg}')
    raise ValueError('❌ No se completó una selección válida de columnas. Vuelva a ejecutar la celda.')


# ---------- 6. Proceso completo: detección → (manual) → melt → limpieza ----------
def procesar_trademap(datos, modo_manual=False, entrada=input):
    col_pais = cols_valor = anios = None
    if not modo_manual:
        col_pais, motivo_pais = detectar_columna_pais(datos)
        cols_valor, motivo_valor = detectar_columnas_valor(datos)
        if col_pais and cols_valor:
            anios = [anio_de(c) for c in cols_valor]
            problemas = validar_columnas(datos, col_pais, cols_valor, anios)
            if problemas:
                print('⚠️ La detección automática encontró inconsistencias: ' + '; '.join(problemas))
                col_pais = None
            else:
                print('✅ Detección automática exitosa')
        else:
            print('⚠️ La detección automática no fue concluyente:')
            for m in (motivo_pais, motivo_valor):
                if m:
                    print(f'   - {m}')
    if modo_manual or not (col_pais and cols_valor):
        col_pais, cols_valor, anios = seleccion_manual(datos, entrada)
        print('✅ Selección manual registrada')

    print(f'   Columna de países: "{col_pais}"')
    print('   Columnas de valores por año:')
    for c, a in zip(cols_valor, anios):
        print(f'     {a} ← "{c}"')

    # Transformar de formato ancho a formato largo (melt)
    df_melted = datos.melt(id_vars=[col_pais], value_vars=cols_valor,
                           var_name='year_column', value_name='trade_value_usd')
    df_melted['year'] = df_melted['year_column'].map(dict(zip(cols_valor, anios))).astype(int)
    df_melted = df_melted.rename(columns={col_pais: 'partner_country'})

    # Limpiar datos
    df_melted['partner_country'] = df_melted['partner_country'].astype(str).str.strip()
    df_melted['trade_value_usd'] = pd.to_numeric(limpiar_numeros(df_melted['trade_value_usd']), errors='coerce')
    df = df_melted[['year', 'partner_country', 'trade_value_usd']].copy()

    # Eliminar filas con datos faltantes y fila "Mundo" / "World"
    df = df.dropna(subset=['trade_value_usd'])
    df = df[~df['partner_country'].map(normalizar).isin(FILAS_TOTAL)].reset_index(drop=True)
    if df.empty:
        raise ValueError('❌ Después de la limpieza no quedaron datos. Revise las columnas seleccionadas.')
    return df, df_melted, cols_valor


# ---------- 7. Ejecución ----------
# Subir archivo Excel directamente desde TradeMap
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Leer datos tal como vienen de TradeMap
df_raw = leer_trademap(filename, uploaded[filename])

# Verificar estructura original
print("Columnas disponibles:", df_raw.columns.tolist())
print("\nPrimeras 5 filas:")
print(df_raw.head())
print()

df, df_melted, year_columns = procesar_trademap(df_raw, modo_manual=MODO_MANUAL)

print(f"\nDatos procesados: {len(df)} filas")
print(f"Países únicos: {df['partner_country'].nunique()}")
print(f"Años disponibles: {sorted(df['year'].unique())}")
print("\nPrimeros datos procesados:")
print(df.head(10))

## 4.3. Calcular Market Share, Ranking y aplicar filtros

```
# Calcular market share por año y país
market_evolution = df.groupby(['year', 'partner_country'])['trade_value_usd'].sum().reset_index()

# Calcular market share total por año
yearly_totals = market_evolution.groupby('year')['trade_value_usd'].sum()
market_evolution['market_share'] = market_evolution.apply(
    lambda x: (x['trade_value_usd'] / yearly_totals[x['year']]) * 100, axis=1
)

# Identificar top países
top_countries = df.groupby('partner_country')['trade_value_usd'].sum().nlargest(8).index

# Filtrar solo top países
market_evolution_top = market_evolution[market_evolution['partner_country'].isin(top_countries)]
```

## 4.4.  Generar Tablas Pivot para análisis

```
# TABLA 1: Market Share por año y país
print("📊 TABLA 1: MARKET SHARE POR AÑO (%)")
print("="*60)
pivot_share = market_evolution_top.pivot(index='year', columns='partner_country', values='market_share')
pivot_share_rounded = pivot_share.round(1)
print(pivot_share_rounded.to_string())

# TABLA 2: Valores de comercio (Millones USD)
print("\n\n💰 TABLA 2: VALORES DE COMERCIO (Millones USD)")
print("="*60)
pivot_value = market_evolution_top.pivot(index='year', columns='partner_country', values='trade_value_usd')
pivot_value_millions = (pivot_value / 1000).round(1)  # Convertir a millones
print(pivot_value_millions.to_string())

# TABLA 3: Crecimiento anual por país
print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
print("="*60)
growth_data = market_evolution_top.groupby('partner_country').apply(
    lambda x: ((x['trade_value_usd'].iloc[-1] / x['trade_value_usd'].iloc[0]) ** (1/(len(x)-1)) - 1) * 100
).reset_index(name='cagr')
growth_summary = growth_data.set_index('partner_country')['cagr'].round(1).sort_values(ascending=False)
print(growth_summary.to_string())

# TABLA 4: Ranking por año
print("\n\n🏆 TABLA 4: RANKING DE PAÍSES POR AÑO (Top 5)")
print("="*60)
for year in sorted(df['year'].unique()):
    year_data = market_evolution_top[market_evolution_top['year'] == year].nlargest(5, 'market_share')
    print(f"\n{year}:")
    for i, (_, row) in enumerate(year_data.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% ({row['trade_value_usd']/1000:.0f}M USD)")
```

## 4.5. Generar Gráficos analíticos con selector de paleta de colores

# ============================================================
# 4.5. GRÁFICOS ANALÍTICOS CON SELECCIÓN DE PALETA DE COLORES
# ============================================================
# Ejecute esta celda, elija una paleta en el menú y presione
# "Generar gráficos". No es necesario modificar el código.

import re
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.colors import LinearSegmentedColormap, to_hex

# ---------- 1. Paletas predeterminadas (configuración central) ----------
PALETAS_PREDETERMINADAS = {
    'Ejecutivo Clásico': [
        '#1F3A5F', '#4A6FA5', '#9DB4D3', '#5C6B73',
        '#A3ADB5', '#C9A227', '#2E8B8B', '#7A4E8C'
    ],
    'Comercio Global': [
        '#0B4F6C', '#01BAEF', '#20A39E', '#2A9D8F',
        '#8AB17D', '#E9C46A', '#F4A261', '#264653'
    ],
    'Corporativo Contemporáneo': [
        '#3D405B', '#81B29A', '#E07A5F', '#F2CC8F',
        '#8D6E63', '#B5838D', '#6D6875', '#D4A373'
    ],
}
OPCION_PERSONALIZADA = 'Personalizada (pegar código CSS)'
MIN_COLORES_PERSONALIZADOS = 2

# Paleta activa (se actualiza con la selección del usuario)
PALETA_ACTIVA = list(PALETAS_PREDETERMINADAS['Ejecutivo Clásico'])


# ---------- 2. Procesamiento de la paleta personalizada (CSS) ----------
def extraer_colores_css(texto_css):
    """Extrae los códigos HEX de un texto CSS. El CSS NO se ejecuta:
    solo se leen los valores de color con una expresión regular."""
    if not isinstance(texto_css, str) or not texto_css.strip():
        return []
    # Quitar comentarios /* ... */ para no tomar colores comentados
    texto = re.sub(r'/\*.*?\*/', ' ', texto_css, flags=re.DOTALL)
    hallados = re.findall(r'#([0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})\b', texto)
    colores = []
    for h in hallados:
        if len(h) in (3, 4):            # #abc -> #aabbcc (se ignora el canal alfa)
            h = ''.join(c * 2 for c in h[:3])
        h = '#' + h[:6].upper()          # #rrggbbaa -> #rrggbb
        if h not in colores:            # eliminar duplicados conservando el orden
            colores.append(h)
    return colores


def validar_paleta_css(texto_css):
    """Devuelve (colores, mensaje_error). mensaje_error es None si es válida."""
    if not texto_css or not texto_css.strip():
        return [], '⚠️ El campo está vacío. Pegue el código CSS de su paleta (por ejemplo, exportado desde Coolors).'
    colores = extraer_colores_css(texto_css)
    if len(colores) == 0:
        return [], ('⚠️ No se encontraron colores HEX válidos (formato #RRGGBB o #RGB). '
                    'Revise que copió el código CSS completo de la paleta.')
    if len(colores) < MIN_COLORES_PERSONALIZADOS:
        return colores, (f'⚠️ Solo se encontró {len(colores)} color ({colores[0]}). '
                         f'Se necesitan al menos {MIN_COLORES_PERSONALIZADOS} colores distintos.')
    return colores, None


def ajustar_paleta(colores, n):
    """Garantiza n colores. Si la paleta tiene menos, interpola entre sus
    colores para no repetir tonos entre países."""
    if len(colores) >= n:
        return list(colores[:n])
    cmap = LinearSegmentedColormap.from_list('paleta_msa', colores)
    return [to_hex(cmap(i / (n - 1))).upper() for i in range(n)]


def colores_por_pais(paleta, paises):
    """Asigna un color fijo a cada país para usarlo igual en los 4 gráficos."""
    paises = list(paises)
    return dict(zip(paises, ajustar_paleta(paleta, len(paises))))


# ---------- 3. Generación de los gráficos (cálculos sin cambios) ----------
def generar_graficos(paleta):
    global fig, axes, latest_year, latest_data, bars, bars2, PALETA_ACTIVA
    PALETA_ACTIVA = list(paleta)
    mapa = colores_por_pais(PALETA_ACTIVA, pivot_share.columns)
    color_texto = '#333333'

    # Gráficos mejorados con valores
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Market Share Trends con valores en puntos clave
    pivot_share.plot(kind='line', marker='o', ax=axes[0,0], linewidth=2, markersize=6,
                     color=[mapa[c] for c in pivot_share.columns])
    axes[0,0].set_title('Market Share Evolution (%)', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Market Share (%)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Añadir valores para Brasil y Colombia (principales)
    for country in ['Brasil', 'Colombia']:
        if country in pivot_share.columns:
            for year in pivot_share.index:
                value = pivot_share.loc[year, country]
                if not pd.isna(value):
                    axes[0,0].annotate(f'{value:.1f}%',
                                       (year, value),
                                       textcoords="offset points",
                                       xytext=(0,10),
                                       ha='center', fontsize=8, color=mapa[country])

    # Trade Value Evolution
    (pivot_value/1000).plot(kind='line', marker='s', ax=axes[0,1], linewidth=2, markersize=6,
                            color=[mapa.get(c, PALETA_ACTIVA[0]) for c in pivot_value.columns])
    axes[0,1].set_title('Trade Value Evolution (Million USD)', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Million USD')
    axes[0,1].grid(True, alpha=0.3)

    # Market Share actual con valores
    latest_year = df['year'].max()
    latest_data = market_evolution_top[market_evolution_top['year'] == latest_year].nlargest(8, 'market_share')
    serie_ms = latest_data.set_index('partner_country')['market_share']
    bars = serie_ms.plot(kind='bar', ax=axes[1,0],
                         color=[mapa.get(c, PALETA_ACTIVA[0]) for c in serie_ms.index])
    axes[1,0].set_title(f'Market Share {latest_year} (%)', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Market Share (%)')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Añadir valores encima de las barras
    for i, (country, value) in enumerate(serie_ms.items()):
        axes[1,0].text(i, value + 0.5, f'{value:.1f}%', ha='center', va='bottom',
                       fontsize=9, color=color_texto)

    # CAGR con valores
    serie_cagr = growth_data.set_index('partner_country')['cagr']
    bars2 = serie_cagr.plot(kind='bar', ax=axes[1,1],
                            color=[mapa.get(c, PALETA_ACTIVA[0]) for c in serie_cagr.index])
    axes[1,1].set_title('CAGR % (Compound Annual Growth Rate)', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('CAGR (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].axhline(y=0, color=color_texto, linestyle='-', alpha=0.3)

    # Añadir valores encima de las barras
    for i, (country, value) in enumerate(serie_cagr.items()):
        if pd.isna(value) or np.isinf(value):
            continue
        axes[1,1].text(i, value + 0.5 if value > 0 else value - 1, f'{value:.1f}%',
                       ha='center', va='bottom' if value > 0 else 'top',
                       fontsize=9, color=color_texto)

    plt.tight_layout()
    plt.show()


# ---------- 4. Interfaz interactiva (Google Colab / Jupyter) ----------
def muestra_paleta_html(colores):
    cuadros = ''.join(
        f'<span title="{c}" style="display:inline-block;width:26px;height:18px;'
        f'background:{c};margin-right:3px;border:1px solid #ccc;border-radius:3px"></span>'
        for c in colores)
    return f'<div style="margin:2px 0 6px 0">{cuadros}</div>'

selector_paleta = widgets.RadioButtons(
    options=list(PALETAS_PREDETERMINADAS.keys()) + [OPCION_PERSONALIZADA],
    value='Ejecutivo Clásico',
    description='Paleta:',
    layout=widgets.Layout(width='max-content'),
)
vista_previa = widgets.HTML(
    '<b>Vista previa de las paletas:</b><br>' + ''.join(
        f'{nombre}{muestra_paleta_html(cols)}' for nombre, cols in PALETAS_PREDETERMINADAS.items()))
campo_css = widgets.Textarea(
    placeholder='/* Ejemplo exportado desde Coolors */\n:root {\n  --color-1: #264653ff;\n  --color-2: #2a9d8fff;\n  --color-3: #e9c46aff;\n}',
    description='CSS:',
    layout=widgets.Layout(width='620px', height='160px', display='none'),
)
mensaje = widgets.HTML()
boton = widgets.Button(description='Generar gráficos', button_style='primary', icon='bar-chart')
salida = widgets.Output()

def al_cambiar_opcion(cambio):
    # El campo CSS solo aparece cuando se elige la opción personalizada
    campo_css.layout.display = 'flex' if cambio['new'] == OPCION_PERSONALIZADA else 'none'
    mensaje.value = ''

def al_presionar_boton(_):
    with salida:
        clear_output(wait=True)
        opcion = selector_paleta.value
        if opcion == OPCION_PERSONALIZADA:
            colores, error = validar_paleta_css(campo_css.value)
            if error:
                mensaje.value = f'<span style="color:#B00020">{error}</span>'
                return
            mensaje.value = (f'<span style="color:#1B5E20">✅ Paleta personalizada válida: '
                             f'{len(colores)} colores detectados.</span>{muestra_paleta_html(colores)}')
        else:
            colores = PALETAS_PREDETERMINADAS[opcion]
            mensaje.value = f'<span style="color:#1B5E20">✅ Paleta seleccionada: {opcion}</span>'
        try:
            generar_graficos(colores)
        except Exception as e:
            mensaje.value = (f'<span style="color:#B00020">⚠️ No se pudieron generar los gráficos: {e}. '
                             f'Verifique que ejecutó las secciones 4.1 a 4.4.</span>')

selector_paleta.observe(al_cambiar_opcion, names='value')
boton.on_click(al_presionar_boton)

display(widgets.VBox([
    widgets.HTML('<h3 style="margin:0">🎨 Seleccione la paleta de colores para los gráficos</h3>'),
    vista_previa, selector_paleta, campo_css, boton, mensaje, salida,
]))

## 4.6. Generar Insights estratégicos o "Descubrimientos relevantes" 

**Nota:** *Reemplazar "producto analizado" (Line 2) y "Producto" (Line 35) por el número de partida y nombre del producto analizado antes de proceder con la ejecución del script.*

```
# Generar insights automáticos
def generate_insights(df, market_evolution_top, product_name="producto analizado"):
    latest_year = df['year'].max()
    first_year = df['year'].min()

    print(f"📊 INSIGHTS ESTRATÉGICOS - {product_name.upper()}")
    print("="*60)

    # Top 3 países
    top3 = market_evolution_top[market_evolution_top['year'] == latest_year].nlargest(3, 'market_share')
    print(f"\n🏆 TOP 3 PAÍSES ({latest_year}):")
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% market share")

    # Mayor crecimiento
    growth_analysis = market_evolution_top.groupby('partner_country').apply(
        lambda x: x['trade_value_usd'].iloc[-1] / x['trade_value_usd'].iloc[0] - 1
    ).sort_values(ascending=False)

    print(f"\n📈 MAYOR CRECIMIENTO ({first_year}-{latest_year}):")
    print(f"  {growth_analysis.index[0]}: +{growth_analysis.iloc[0]:.1%}")

    # Oportunidades
    declining = growth_analysis[growth_analysis < 0]
    if len(declining) > 0:
        print(f"\n⚠️  MERCADOS EN DECLIVE:")
        for country, decline in declining.head(3).items():
            print(f"  {country}: {decline:.1%}")

    # Tamaño total del mercado
    total_market = df[df['year'] == latest_year]['trade_value_usd'].sum()
    print(f"\n💰 TAMAÑO TOTAL DEL MERCADO ({latest_year}): ${total_market/1e9:.1f}B USD")

# Define the product_name variable
product_name = "Producto"

# Aplicar análisis con nombre de producto dinámico
generate_insights(df, market_evolution_top, product_name)
```

## 4.7. Generación de Promts para análisis posteriores

**Nota:** *Reemplazar "market_name" (Line 7) por el mercado objetivo (país-destino) antes de proceder con la ejecución del script.*

```
# GENERAR PROMPT AUTOMÁTICO PARA ANÁLISIS
def generate_analysis_prompt(df, market_evolution_top, product_name):
    latest_year = df['year'].max()
    first_year = df['year'].min()

    # Detectar mercado objetivo (desde columnas o contexto)
    market_name = "Estados Unidos"  # Por defecto, se podría dinamizar

    # Generar estadísticas clave
    top3 = market_evolution_top[market_evolution_top['year'] == latest_year].nlargest(3, 'market_share')
    total_market = df[df['year'] == latest_year]['trade_value_usd'].sum()

    prompt = f"""
    Actúa como consultor en inteligencia de mercados y analiza estos datos de market share de {product_name.lower()} en {market_name} ({first_year}-{latest_year}). Proporciona insights estratégicos específicos:

    CONTEXTO DEL MERCADO:
    - Producto: {product_name}
    - Mercado objetivo: {market_name}
    - Período: {first_year}-{latest_year}
    - Tamaño total del mercado ({latest_year}): ${total_market/1000:.0f} millones USD

    TOP 3 LÍDERES ACTUALES ({latest_year}):
    """

    for i, (_, row) in enumerate(top3.iterrows(), 1):
        prompt += f"\n    {i}. {row['partner_country']}: {row['market_share']:.1f}% market share"

    prompt += f"""

    DATOS PARA ANÁLISIS:
    [Pegar aquí las 4 tablas generadas por el script anterior]

    PREGUNTAS ESPECÍFICAS A RESPONDER:

    1. ANÁLISIS COMPETITIVO:
       - ¿Cuáles son las tendencias de los líderes del mercado?
       - ¿Qué países están ganando/perdiendo participación?
       - ¿Hay oportunidades por concentración/fragmentación del mercado?

    2. OPORTUNIDADES DE ENTRADA:
       - ¿Qué nichos o segmentos presentan menos competencia?
       - ¿Existen países que han reducido su participación creando vacíos?
       - ¿Cuál sería el mejor timing para entrar al mercado?

    3. BENCHMARKING REGIONAL:
       - ¿Cómo se comportan países con perfil similar al de tu empresa?
       - ¿Qué países de América Latina tienen mejor performance?
       - ¿Cuáles son las mejores prácticas observables?

    4. ESTRATEGIA DE POSICIONAMIENTO:
       - ¿Competir directamente con líderes o buscar nichos?
       - ¿Qué ventajas competitivas serían más relevantes?
       - ¿Cuál sería un market share objetivo realista en 2-3 años?

    5. RIESGOS Y AMENAZAS:
       - ¿Qué competidores emergentes representan amenazas?
       - ¿Hay signos de saturación o madurez del mercado?
       - ¿Qué factores externos podrían afectar la industria?

    FORMATO DE RESPUESTA:
    - Insights concisos y accionables
    - Recomendaciones estratégicas específicas
    - Identificación de 2-3 oportunidades prioritarias
    - Métricas clave a monitorear
    """

    return prompt

# Generar y mostrar el prompt
analysis_prompt = generate_analysis_prompt(df, market_evolution_top, product_name)
print("="*80)
print("📋 PROMPT PARA ANÁLISIS ESTRATÉGICO:")
print("="*80)
print(analysis_prompt)
print("\n" + "="*80)
print("💡 INSTRUCCIONES:")
print("1. Copia este prompt")
print("2. Pega las 4 tablas de datos generadas arriba")
print("3. Úsalo con cualquier chatbot para obtener insights estratégicos")
print("="*80)
```

## 4.8. Helpers (opcionales) para depuración y personalización
### 4.8.1. Promt para análisis de oportunidad: Origen (Proveedor)-Destino

```
Con base en lo anterior, redacta un texto argumentativo, en tono formal y académico en tercera persona, máximo una cuartilla (250 palabras) donde evidencies los resultados de este estudio de mercados. Enfatiza en el nivel de oportunidad que tendría [COLOMBIA] para ingresar a este mercado desde el punto de vista comercial basándose únicamente en los datos suministrados.
```

### 4.8.2. Promt para personalizar colores en los gráficos

```
Con base en la siguiente paleta de colores:

[CSS.CODE]

Ajusta el siguiente trozo de código en consecuencia, da por hecho que las librerías ya se encuentran instaladas:

[PY.CODE.4.5]
```

### 4.8.3. Ajuste para datos faltantes o sin datos históricos (No calcula CAGR)

```
# TABLA 1: Market Share por año y país
print("📊 TABLA 1: MARKET SHARE POR AÑO (%)")
print("="*60)
pivot_share = market_evolution_top.pivot(index='year', columns='partner_country', values='market_share')
pivot_share_rounded = pivot_share.round(1)
print(pivot_share_rounded.to_string())

# TABLA 2: Valores de comercio (Millones USD)
print("\n\n💰 TABLA 2: VALORES DE COMERCIO (Millones USD)")
print("="*60)
pivot_value = market_evolution_top.pivot(index='year', columns='partner_country', values='trade_value_usd')
pivot_value_millions = (pivot_value / 1000).round(1)  # Convertir a millones
print(pivot_value_millions.to_string())

# TABLA 3: Crecimiento anual por país
# El cálculo del CAGR requiere al menos dos años de datos. Como solo se tiene un año (2024),
# esta tabla no se puede generar con los datos actuales.
# print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
# print("="*60)
# growth_data = market_evolution_top.groupby('partner_country').apply(
#     lambda x: ((x['trade_value_usd']. iloc[-1] / x['trade_value_usd'].iloc[0]) ** (1/(len(x)-1)) - 1) * 100
# ).reset_index(name='cagr')
# growth_summary = growth_data.set_index('partner_country')['cagr'].round(1).sort_values(ascending=False)
# print(growth_summary.to_string())
print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
print("="*60)
print("No se puede calcular el CAGR. Se requiere al menos dos años de datos para este cálculo.")

# TABLA 4: Ranking por año
print("\n\n🏆 TABLA 4: RANKING DE PAÍSES POR AÑO (Top 5)")
print("="*60)
for year in sorted(df['year'].unique()):
    year_data = market_evolution_top[market_evolution_top['year'] == year].nlargest(5, 'market_share')
    print(f"\n{year}:")
    for i, (_, row) in enumerate(year_data.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% ({row['trade_value_usd']/1000:.0f}M USD)")
```

### 4.8.4. Ajuste para datos faltantes (Calcula CAGR omitiendo datos vacíos o nulos)

```
# TABLA 1: Market Share por año y país
print("📊 TABLA 1: MARKET SHARE POR AÑO (%)")
print("="*60)
pivot_share = market_evolution_top.pivot(index='year', columns='partner_country', values='market_share')
pivot_share_rounded = pivot_share.round(1)
print(pivot_share_rounded.to_string())

# TABLA 2: Valores de comercio (Millones USD)
print("\n\n💰 TABLA 2: VALORES DE COMERCIO (Millones USD)")
print("="*60)
pivot_value = market_evolution_top.pivot(index='year', columns='partner_country', values='trade_value_usd')
pivot_value_millions = (pivot_value / 1000).round(1)  # Convertir a millones
print(pivot_value_millions.to_string())

# TABLA 3: Crecimiento anual por país
print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
print("="*60)
growth_data = market_evolution_top.groupby('partner_country').apply(
    lambda x: ((x['trade_value_usd'].iloc[-1] / x['trade_value_usd'].iloc[0]) ** (1/(len(x)-1)) - 1) * 100 if len(x) > 1 else np.nan
).reset_index(name='cagr')
growth_summary = growth_data.set_index('partner_country')['cagr'].round(1).sort_values(ascending=False)
print(growth_summary.to_string())

# TABLA 4: Ranking por año
print("\n\n🏆 TABLA 4: RANKING DE PAÍSES POR AÑO (Top 5)")
print("="*60)
for year in sorted(df['year'].unique()):
    year_data = market_evolution_top[market_evolution_top['year'] == year].nlargest(5, 'market_share')
    print(f"\n{year}:")
    for i, (_, row) in enumerate(year_data.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% ({row['trade_value_usd']/1000:.0f}M USD)")
```
