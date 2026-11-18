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

**Nota:** *El bloque reconoce automáticamente la columna de países y las columnas de valores por año, con cabeceras en español o inglés (p. ej. "Exportadores"/"Exporters", "Valor importado en 2024"/"Imported value in 2024"), omite filas de título y admite .xlsx, .xls y los .xls de Trade Map en formato HTML. Si la detección no es concluyente, o si se marca la casilla `asignacion_manual`, el script lista las columnas con su número y solicita elegirlas. La salida conserva el DataFrame `df` con `year`, `partner_country` y `trade_value_usd`.*

```
#@title 4.2. Cargar archivos fuente (Trademap) { display-mode: "form" }
#@markdown Marque la casilla solo si desea elegir las columnas manualmente (por ejemplo, si la detección automática falla).
asignacion_manual = False  #@param {type:"boolean"}

import os
import re
import unicodedata

# ---------------------------------------------------------------------------
# Diccionarios de reconocimiento (español / inglés). Ampliar aquí si Trade Map
# cambia sus encabezados; no es necesario tocar el resto del código.
# ---------------------------------------------------------------------------
PALABRAS_PAIS = ['exportador', 'importador', 'socio', 'pais', 'mercado', 'proveedor',
                 'origen', 'destino', 'exporter', 'importer', 'partner', 'country',
                 'market', 'supplier', 'reporter']
PALABRAS_VALOR = ['valor', 'value', 'importado', 'exportado', 'importacion', 'exportacion',
                  'imported', 'exported', 'import', 'export', 'comercio', 'trade', 'usd']
PALABRAS_EXCLUIR = ['cantidad', 'quantity', 'unidad', 'unit', 'tasa', 'rate', 'crecimiento',
                    'growth', 'participacion', 'share', 'balanza', 'balance', 'precio', 'price',
                    'distancia', 'distance', 'arancel', 'tariff', 'concentracion', 'concentration',
                    'codigo', 'code', 'posicion', 'ranking']
FILAS_AGREGADAS = {'mundo', 'world'}          # fila "Mundo" / "World" que se excluye
PATRON_ANIO = re.compile(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)')


def _normalizar(texto):
    """Minúsculas, sin tildes y con espacios simples (para comparar encabezados)."""
    texto = unicodedata.normalize('NFKD', str(texto))
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', texto).strip().lower()


def _texto_celda(valor):
    """Convierte una celda de encabezado en texto (2020.0 -> '2020')."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ''
    if isinstance(valor, float) and valor.is_integer():
        valor = int(valor)
    return re.sub(r'\s+', ' ', str(valor)).strip()


def _leer_tabla_cruda(nombre):
    """Lee el archivo sin asumir encabezados. Admite .xlsx, .xls y los .xls de
    Trade Map que en realidad son HTML o texto tabulado."""
    errores = []
    try:
        return pd.read_excel(nombre, sheet_name=0, header=None)
    except ImportError as e:  # falta xlrd para .xls binarios
        errores.append(f'read_excel: {e}')
        try:
            import subprocess, sys
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xlrd'], check=True)
            return pd.read_excel(nombre, sheet_name=0, header=None)
        except Exception as e2:
            errores.append(f'xlrd: {e2}')
    except Exception as e:
        errores.append(f'read_excel: {e}')
    try:  # Trade Map exporta a menudo un "Excel" que es una tabla HTML
        tablas = pd.read_html(nombre)
        tabla = max(tablas, key=lambda t: t.shape[0] * t.shape[1])
        if isinstance(tabla.columns, pd.MultiIndex):
            cab = [' '.join(dict.fromkeys(_texto_celda(x) for x in col if 'Unnamed' not in str(x)))
                   for col in tabla.columns]
        else:
            cab = [_texto_celda(c) for c in tabla.columns]
        tabla = tabla.astype(object)
        tabla.columns = range(tabla.shape[1])
        return pd.concat([pd.DataFrame([cab]), tabla], ignore_index=True)
    except Exception as e:
        errores.append(f'read_html: {e}')
    for codificacion in ('utf-8', 'latin-1', 'utf-16'):
        try:
            tabla = pd.read_csv(nombre, sep='\t', header=None, encoding=codificacion)
            if tabla.shape[1] > 1:
                return tabla
        except Exception as e:
            errores.append(f'read_csv ({codificacion}): {e}')
    raise ValueError('❌ No fue posible leer el archivo. Verifique que sea una exportación de Trade Map '
                     '(.xlsx, .xls o .txt).\nDetalle técnico:\n  - ' + '\n  - '.join(errores))


def _detectar_fila_encabezado(tabla, max_filas=30):
    """Trade Map puede incluir filas de título antes de los encabezados. Se elige
    la fila con más años y palabras clave (o la primera si ninguna destaca)."""
    mejor, mejor_puntaje = 0, -1
    for i in range(min(max_filas, len(tabla))):
        celdas = [_normalizar(_texto_celda(v)) for v in tabla.iloc[i].tolist()]
        anios = sum(bool(PATRON_ANIO.search(c)) for c in celdas)
        pais = sum(any(p in c for p in PALABRAS_PAIS) for c in celdas)
        no_vacias = sum(bool(c) for c in celdas)
        if anios == 0 or no_vacias < 2:
            continue
        puntaje = anios * 2 + pais * 3 + no_vacias * 0.1
        if puntaje > mejor_puntaje:
            mejor, mejor_puntaje = i, puntaje
    return mejor


def _construir_df_raw(tabla):
    fila = _detectar_fila_encabezado(tabla)
    encabezados, vistos = [], {}
    for j, v in enumerate(tabla.iloc[fila].tolist()):
        nombre = _texto_celda(v) or f'Columna_{j}'
        if nombre in vistos:                      # encabezados repetidos -> sufijo
            vistos[nombre] += 1
            nombre = f'{nombre} ({vistos[nombre]})'
        else:
            vistos[nombre] = 0
        encabezados.append(nombre)
    datos = tabla.iloc[fila + 1:].copy()
    datos.columns = encabezados
    datos = datos.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    return datos, fila


def _proporcion_numerica(serie):
    serie = serie.dropna()
    if len(serie) == 0:
        return 0.0
    return pd.to_numeric(serie, errors='coerce').notna().mean()


def _proporcion_texto(serie):
    serie = serie.dropna()
    if len(serie) == 0:
        return 0.0
    return (pd.to_numeric(serie, errors='coerce').isna() & serie.astype(str).str.strip().ne('')).mean()


def _anio_de(nombre_columna):
    m = PATRON_ANIO.findall(str(nombre_columna))
    return int(m[-1]) if m else None


def detectar_columnas(df_raw):
    """Devuelve (columna_pais, {columna: año}, avisos, concluyente)."""
    avisos = []
    columnas = list(df_raw.columns)

    # 1) Columnas de valores comerciales por año
    con_anio = [c for c in columnas
                if _anio_de(c) and not any(p in _normalizar(c) for p in PALABRAS_EXCLUIR)]
    con_valor = [c for c in con_anio if any(p in _normalizar(c) for p in PALABRAS_VALOR)]
    candidatas = con_valor or [c for c in con_anio if re.fullmatch(r'\s*(19|20)\d{2}\s*', str(c))]
    candidatas = [c for c in candidatas if _proporcion_numerica(df_raw[c]) >= 0.5]
    columnas_anio = {}
    for c in candidatas:
        anio = _anio_de(c)
        if anio in columnas_anio.values():
            avisos.append(f'El año {anio} aparece en más de una columna; se usa la primera.')
            continue
        columnas_anio[c] = anio

    # 2) Columna de países / socios comerciales
    restantes = [c for c in columnas if c not in columnas_anio and not _anio_de(c)]
    por_nombre = [c for c in restantes
                  if any(p in _normalizar(c) for p in PALABRAS_PAIS)
                  and not any(p in _normalizar(c) for p in ('codigo', 'code'))
                  and _proporcion_texto(df_raw[c]) >= 0.6]
    if len(por_nombre) == 1:
        columna_pais = por_nombre[0]
    elif len(por_nombre) > 1:
        columna_pais = por_nombre[0]
        avisos.append(f'Varias columnas parecen contener países: {por_nombre}.')
    else:  # sin palabra clave: columna de texto con más valores distintos
        texto = [c for c in restantes if _proporcion_texto(df_raw[c]) >= 0.8]
        columna_pais = max(texto, key=lambda c: df_raw[c].nunique()) if texto else None
        if columna_pais is not None:
            avisos.append(f'La columna de países se dedujo por su contenido: "{columna_pais}".')

    concluyente = (columna_pais is not None and len(columnas_anio) > 0
                   and len(por_nombre) <= 1)
    return columna_pais, columnas_anio, avisos, concluyente


def _mostrar_columnas(df_raw):
    print('\nColumnas disponibles en el archivo:')
    print(f"  {'Nº':>3}  {'Encabezado':<50} {'Tipo':<9} Ejemplo")
    for i, c in enumerate(df_raw.columns):
        tipo = 'número' if _proporcion_numerica(df_raw[c]) >= 0.5 else 'texto'
        ejemplo = df_raw[c].dropna()
        ejemplo = str(ejemplo.iloc[0])[:25] if len(ejemplo) else '(vacía)'
        print(f'  {i:>3}  {str(c)[:50]:<50} {tipo:<9} {ejemplo}')


def _interpretar_indices(texto, total):
    indices = []
    for parte in re.split(r'[,\s;]+', texto.strip()):
        if not parte:
            continue
        if not re.fullmatch(r'\d+(-\d+)?', parte):
            raise ValueError(f'"{parte}" no es un número de columna válido. Use números como 3, 1,2,4 o 1-5.')
        if '-' in parte:
            a, b = parte.split('-', 1)
            indices.extend(range(int(a), int(b) + 1))
        else:
            indices.append(int(parte))
    fuera = [i for i in indices if i < 0 or i >= total]
    if fuera:
        raise ValueError(f'Los números {fuera} no corresponden a ninguna columna (0 a {total - 1}).')
    return list(dict.fromkeys(indices))


def asignar_columnas_manualmente(df_raw, intentos=3):
    """Interfaz de selección: lista columnas con nombre e índice y valida la elección."""
    _mostrar_columnas(df_raw)
    columnas = list(df_raw.columns)
    for intento in range(1, intentos + 1):
        try:
            i_pais = _interpretar_indices(input('\n👉 Nº de la columna con los PAÍSES / socios comerciales: '), len(columnas))
            if len(i_pais) != 1:
                raise ValueError('Indique un único número para la columna de países.')
            columna_pais = columnas[i_pais[0]]
            if _proporcion_texto(df_raw[columna_pais]) < 0.6:
                raise ValueError(f'La columna "{columna_pais}" no parece contener nombres de países (es mayormente numérica).')

            i_val = _interpretar_indices(input('👉 Nº de las columnas con VALORES comerciales por año (ej.: 1,2,3 o 1-5): '), len(columnas))
            if not i_val:
                raise ValueError('Debe indicar al menos una columna de valores.')
            columnas_anio = {}
            for i in i_val:
                c = columnas[i]
                if c == columna_pais:
                    raise ValueError('La columna de países no puede ser también una columna de valores.')
                if _proporcion_numerica(df_raw[c]) < 0.5:
                    raise ValueError(f'La columna "{c}" no contiene valores numéricos suficientes.')
                anio = _anio_de(c)
                while anio is None:
                    respuesta = input(f'   No se encontró el año en "{c}". Escriba el año (AAAA): ').strip()
                    anio = int(respuesta) if re.fullmatch(r'(19|20)\d{2}', respuesta) else None
                    if anio is None:
                        print('   ⚠️ Año no válido. Use cuatro dígitos, por ejemplo 2023.')
                if anio in columnas_anio.values():
                    raise ValueError(f'El año {anio} quedó asignado a dos columnas distintas.')
                columnas_anio[c] = anio
            return columna_pais, columnas_anio
        except ValueError as e:
            print(f'⚠️ {e} (intento {intento} de {intentos})')
    raise ValueError('❌ No se completó la asignación manual de columnas. Vuelva a ejecutar la celda.')


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
# Subir archivo Excel directamente desde TradeMap
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Leer datos tal como vienen de TradeMap (detectando la fila de encabezados)
tabla_cruda = _leer_tabla_cruda(filename)
df_raw, fila_encabezado = _construir_df_raw(tabla_cruda)

# Verificar estructura original
if fila_encabezado:
    print(f'ℹ️ Se omitieron {fila_encabezado} fila(s) de título antes de los encabezados.')
print("Columnas disponibles:", df_raw.columns.tolist())
print("\nPrimeras 5 filas:")
print(df_raw.head())

# Identificar columnas: automático (ES/EN) o manual
columna_pais, columnas_anio, avisos, concluyente = detectar_columnas(df_raw)
for aviso in avisos:
    print(f'⚠️ {aviso}')

if asignacion_manual or not concluyente:
    if not asignacion_manual:
        print('\n⚠️ La detección automática no fue concluyente. Seleccione las columnas manualmente.')
    columna_pais, columnas_anio = asignar_columnas_manualmente(df_raw)
else:
    print(f'\n✅ Columna de países detectada: "{columna_pais}"')

year_columns = list(columnas_anio.keys())
print(f"\nColumnas de años encontradas: {year_columns}")
print(f"Años reconocidos: {sorted(columnas_anio.values())}")

# Transformar de formato ancho a formato largo (melt)
df_melted = df_raw.melt(
    id_vars=[columna_pais],
    value_vars=year_columns,
    var_name='year_column',
    value_name='trade_value_usd'
)

# Asignar el año reconocido a cada columna
df_melted['year'] = df_melted['year_column'].map(columnas_anio).astype(int)
df_melted = df_melted.rename(columns={columna_pais: 'partner_country'})
df_melted['partner_country'] = df_melted['partner_country'].astype(str).str.strip()

# Limpiar datos
df_melted['trade_value_usd'] = pd.to_numeric(df_melted['trade_value_usd'], errors='coerce')
df = df_melted[['year', 'partner_country', 'trade_value_usd']].copy()

# Eliminar filas con datos faltantes y fila "Mundo" / "World"
df = df.dropna(subset=['trade_value_usd'])
df = df[~df['partner_country'].map(_normalizar).isin(FILAS_AGREGADAS | {'', 'nan'})].reset_index(drop=True)

# Validación final de la estructura de salida
if df.empty:
    raise ValueError('❌ No quedaron registros con valores comerciales. Revise las columnas seleccionadas '
                     '(active "asignacion_manual" y vuelva a ejecutar).')

print(f"\nDatos procesados: {len(df)} filas")
print(f"Países únicos: {df['partner_country'].nunique()}")
print(f"Años disponibles: {sorted(df['year'].unique())}")
print("\nPrimeros datos procesados:")
print(df.head(10))
```

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

## 4.5. Generar Gráficos analíticos

**Nota:** *Al ejecutar la celda aparece un selector con tres paletas predeterminadas (Ejecutivo Clásico, Comercio Global y Corporativo Contemporáneo) y una opción personalizada. Para la personalizada, pegue el código CSS exportado desde Coolors (u otra fuente con códigos HEX) y pulse "Aplicar paleta y generar gráficos". El CSS no se ejecuta: solo se extraen y validan los colores. Cada país conserva el mismo color en los cuatro gráficos.*

```
#@title 4.5. Generar Gráficos analíticos (con selector de paleta de colores)
import re
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# 1. PALETAS PREDETERMINADAS (definición centralizada de colores)
#    'series': un color por país, en orden fijo (país 1 = color 1, etc.).
#    'tinta' : color neutro para textos, etiquetas y línea de referencia.
# ---------------------------------------------------------------------------
PALETAS_PREDETERMINADAS = {
    'Ejecutivo Clásico': {
        'descripcion': 'Azules sobrios, gris pizarra y acentos ámbar/burdeos para informes ejecutivos.',
        'series': ['#1F5FAD', '#E08E2B', '#3D9BD6', '#8E3B46', '#11907F', '#6C5BAF', '#C0A12B', '#6B7785'],
        'tinta': '#1E2A38',
    },
    'Comercio Global': {
        'descripcion': 'Azules oceánicos y verdes que evocan mercados internacionales y competitividad.',
        'series': ['#1C6DB5', '#6BAF3A', '#4FA8E0', '#C4572B', '#3E4FA8', '#1E9E7A', '#E39B2D', '#2E7D4F'],
        'tinta': '#14303F',
    },
    'Corporativo Contemporáneo': {
        'descripcion': 'Neutros, tierras y acentos cálidos para reportes analíticos modernos.',
        'series': ['#C2572F', '#1A7F99', '#D27D6A', '#8E4E7E', '#6E9A4E', '#9A5B2E', '#D39A2C', '#7D7468'],
        'tinta': '#2F2A26',
    },
}
OPCION_PERSONALIZADA = 'Personalizada (pegar código CSS)'
MARCADORES = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']      # forma distinta por país
ESTILOS_LINEA = ['-', '--', ':', '-.']
TRAMAS_BARRA = ['', '//', '..', 'xx']
MAX_CARACTERES_CSS = 20000
MIN_COLORES = 2

# ---------------------------------------------------------------------------
# 2. PALETA PERSONALIZADA: extracción y validación (el CSS NUNCA se ejecuta,
#    solo se leen los códigos HEX con expresiones regulares)
# ---------------------------------------------------------------------------
_PATRON_HEX = re.compile(r'#([0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})(?![0-9a-zA-Z_-])')
_PATRON_VARIABLE = re.compile(r'--([\w-]+)\s*:\s*#([0-9a-fA-F]{3,8})\b')


def _hex_a_6(codigo):
    """Normaliza #RGB, #RGBA, #RRGGBB y #RRGGBBAA a #RRGGBB (se descarta la transparencia)."""
    codigo = codigo.lstrip('#')
    if len(codigo) in (3, 4):
        codigo = ''.join(c * 2 for c in codigo[:3])
    return '#' + codigo[:6].upper()


def _luminancia(color):
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
               for c in mcolors.to_rgb(color)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def extraer_paleta_css(texto_css):
    """Devuelve (colores_validos, mensajes). Lanza ValueError si no hay paleta utilizable."""
    mensajes = []
    if not texto_css or not texto_css.strip():
        raise ValueError('El campo está vacío. Pegue el código CSS de su paleta (por ejemplo, exportado desde Coolors).')
    if len(texto_css) > MAX_CARACTERES_CSS:
        raise ValueError(f'El texto supera {MAX_CARACTERES_CSS} caracteres. Pegue solo el bloque de la paleta.')

    # Si hay comentarios por secciones (Coolors), se priorizan variables de :root / CSS HEX
    sin_comentarios = re.sub(r'/\*.*?\*/', ' ', texto_css, flags=re.S)
    encontrados = [_hex_a_6(m.group(1)) for m in _PATRON_HEX.finditer(sin_comentarios)]
    nombres = {_hex_a_6(v): n for n, v in _PATRON_VARIABLE.findall(sin_comentarios)}
    if not encontrados:
        raise ValueError('No se encontraron códigos de color HEX (por ejemplo #1F5FAD). '
                         'Verifique que copió el formato "CSS" y no solo los nombres de los colores.')

    unicos = list(dict.fromkeys(encontrados))                 # sin duplicados, en orden
    if len(unicos) < len(encontrados):
        mensajes.append(f'Se omitieron {len(encontrados) - len(unicos)} color(es) repetido(s).')

    utilizables = []
    for c in unicos:
        lum = _luminancia(c)
        if lum > 0.65:
            mensajes.append(f'{c} es demasiado claro y no se distinguiría del fondo: se omitió.')
        elif lum < 0.004:
            mensajes.append(f'{c} es casi negro: se usará para textos, no para series.')
        else:
            utilizables.append(c)
    if len(utilizables) < MIN_COLORES:
        raise ValueError(f'La paleta solo tiene {len(utilizables)} color(es) utilizable(s); se necesitan al menos '
                         f'{MIN_COLORES} colores que no sean demasiado claros ni casi negros.'
                         + ''.join(f'\n   - {m}' for m in mensajes))
    if nombres:
        mensajes.append('Variables reconocidas: ' + ', '.join(f'--{nombres[c]}' for c in utilizables if c in nombres))
    return utilizables, mensajes


def _tinta_para(colores):
    """Color de texto derivado de la paleta: el más oscuro, llevado a casi negro."""
    oscuro = min(colores, key=_luminancia)
    r, g, b = mcolors.to_rgb(oscuro)
    return mcolors.to_hex((r * 0.35, g * 0.35, b * 0.35))


# ---------------------------------------------------------------------------
# 3. ASIGNACIÓN DE ESTILO POR PAÍS (el color sigue al país en los 4 gráficos)
#    Si hay más países que colores, se repiten los colores pero cambia la forma
#    del marcador, el tipo de línea y la trama de la barra (nunca se confunden).
# ---------------------------------------------------------------------------
def construir_estilos(colores, paises):
    estilos = {}
    for i, pais in enumerate(paises):
        vuelta = i // len(colores)
        estilos[pais] = {
            'color': colores[i % len(colores)],
            'marker': MARCADORES[i % len(MARCADORES)],
            'linestyle': ESTILOS_LINEA[vuelta % len(ESTILOS_LINEA)],
            'hatch': TRAMAS_BARRA[vuelta % len(TRAMAS_BARRA)],
        }
    return estilos


def _barras(ax, serie, estilos, tinta):
    paises = list(serie.index)
    barras = ax.bar(range(len(paises)), serie.values,
                    color=[estilos[p]['color'] for p in paises],
                    edgecolor='white', linewidth=1.5)
    for barra, p in zip(barras, paises):
        if estilos[p]['hatch']:
            barra.set_hatch(estilos[p]['hatch'])
            barra.set_edgecolor(tinta)
    ax.set_xticks(range(len(paises)))
    ax.set_xticklabels(paises, rotation=45, ha='right', color=tinta)
    return barras


# ---------------------------------------------------------------------------
# 4. GENERACIÓN DE LOS 4 GRÁFICOS (mismos datos y cálculos de 4.3 y 4.4)
# ---------------------------------------------------------------------------
def generar_graficos(nombre_paleta, colores, tinta):
    global fig, axes, latest_year, latest_data, bars, bars2
    orden_paises = [p for p in top_countries if p in set(pivot_share.columns) | set(growth_data['partner_country'])]
    estilos = construir_estilos(colores, orden_paises)
    rejilla = mcolors.to_rgba(tinta, 0.15)

    # Gráficos mejorados con valores
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Paleta: {nombre_paleta}', fontsize=11, color=tinta, x=0.01, ha='left')

    # Market Share Trends con valores en puntos clave
    ax = axes[0, 0]
    for pais in [p for p in orden_paises if p in pivot_share.columns]:
        e = estilos[pais]
        ax.plot(pivot_share.index, pivot_share[pais], label=pais, color=e['color'], marker=e['marker'],
                linestyle=e['linestyle'], linewidth=2, markersize=7)
    ax.set_title('Market Share Evolution (%)', fontsize=14, fontweight='bold', color=tinta)
    ax.set_ylabel('Market Share (%)', color=tinta)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', labelcolor=tinta)

    # Añadir valores para Brasil y Colombia (principales)
    for country in ['Brasil', 'Brazil', 'Colombia']:
        if country in pivot_share.columns:
            for year in pivot_share.index:
                value = pivot_share.loc[year, country]
                if not pd.isna(value):
                    ax.annotate(f'{value:.1f}%', (year, value), textcoords="offset points",
                                xytext=(0, 10), ha='center', fontsize=8, color=tinta)

    # Trade Value Evolution
    ax = axes[0, 1]
    for pais in [p for p in orden_paises if p in pivot_value.columns]:
        e = estilos[pais]
        ax.plot(pivot_value.index, pivot_value[pais] / 1000, label=pais, color=e['color'], marker=e['marker'],
                linestyle=e['linestyle'], linewidth=2, markersize=7)
    ax.set_title('Trade Value Evolution (Million USD)', fontsize=14, fontweight='bold', color=tinta)
    ax.set_ylabel('Million USD', color=tinta)
    ax.legend(labelcolor=tinta)

    for ax in axes[0]:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel('year', color=tinta)

    # Market Share actual con valores
    latest_year = df['year'].max()
    latest_data = market_evolution_top[market_evolution_top['year'] == latest_year].nlargest(8, 'market_share')
    serie_ms = latest_data.set_index('partner_country')['market_share']
    ax = axes[1, 0]
    bars = _barras(ax, serie_ms, estilos, tinta)
    ax.set_title(f'Market Share {latest_year} (%)', fontsize=14, fontweight='bold', color=tinta)
    ax.set_ylabel('Market Share (%)', color=tinta)
    for i, value in enumerate(serie_ms.values):
        ax.text(i, value + 0.5, f'{value:.1f}%', ha='center', va='bottom', fontsize=9, color=tinta)

    # CAGR con valores
    serie_cagr = growth_data.set_index('partner_country')['cagr']
    ax = axes[1, 1]
    bars2 = _barras(ax, serie_cagr, estilos, tinta)
    ax.set_title('CAGR % (Compound Annual Growth Rate)', fontsize=14, fontweight='bold', color=tinta)
    ax.set_ylabel('CAGR (%)', color=tinta)
    ax.axhline(y=0, color=tinta, linestyle='-', alpha=0.5)
    for i, value in enumerate(serie_cagr.values):
        if pd.isna(value):
            continue
        ax.text(i, value + 0.5 if value > 0 else value - 1, f'{value:.1f}%',
                ha='center', va='bottom' if value > 0 else 'top', fontsize=9, color=tinta)

    for ax in axes.flat:
        ax.grid(True, color=rejilla)
        ax.tick_params(colors=tinta)
        for borde in ax.spines.values():
            borde.set_color(rejilla)

    plt.tight_layout()
    plt.show()


def _muestra_html(colores):
    return ''.join(f'<span title="{c}" style="display:inline-block;width:26px;height:18px;'
                   f'background:{c};margin-right:3px;border-radius:3px"></span>' for c in colores)


def aplicar_seleccion(opcion, texto_css=''):
    """Resuelve la opción elegida en (nombre, colores, tinta) o informa el error."""
    global paleta_seleccionada
    if opcion == OPCION_PERSONALIZADA:
        colores, mensajes = extraer_paleta_css(texto_css)
        for m in mensajes:
            print(f'ℹ️ {m}')
        tinta = _tinta_para(colores)
        if len(colores) < len(top_countries):
            print(f'ℹ️ La paleta tiene {len(colores)} colores para {len(top_countries)} países: '
                  'los colores se repiten con otra forma de marcador, tipo de línea y trama.')
        nombre = 'Personalizada'
    else:
        p = PALETAS_PREDETERMINADAS[opcion]
        colores, tinta, nombre = p['series'], p['tinta'], opcion
    paleta_seleccionada = {'nombre': nombre, 'colores': colores, 'tinta': tinta}
    print(f'✅ Paleta aplicada: {nombre} ({len(colores)} colores)')
    generar_graficos(nombre, colores, tinta)


# ---------------------------------------------------------------------------
# 5. INTERFAZ INTERACTIVA (ipywidgets, incluido en Google Colab)
# ---------------------------------------------------------------------------
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML
    _HAY_WIDGETS = True
except ImportError:
    _HAY_WIDGETS = False

if _HAY_WIDGETS:
    opciones = list(PALETAS_PREDETERMINADAS) + [OPCION_PERSONALIZADA]
    selector = widgets.RadioButtons(options=opciones, value=opciones[0],
                                    description='Paleta:', layout=widgets.Layout(width='auto'))
    vista_previa = widgets.HTML()
    campo_css = widgets.Textarea(
        placeholder='Pegue aquí el código CSS de su paleta, por ejemplo:\n:root {\n  --azul: #1f5fadff;\n  --naranja: #e08e2bff;\n}',
        layout=widgets.Layout(width='620px', height='150px', display='none'))
    boton = widgets.Button(description='Aplicar paleta y generar gráficos', button_style='primary',
                           icon='bar-chart', layout=widgets.Layout(width='300px'))
    salida = widgets.Output()

    def _actualizar_vista(_=None):
        if selector.value == OPCION_PERSONALIZADA:
            campo_css.layout.display = 'block'
            vista_previa.value = ('<i>Pegue el código CSS (Coolors → Export → CSS) y pulse el botón. '
                                  'Solo se leen los códigos HEX; el texto no se ejecuta.</i>')
        else:
            campo_css.layout.display = 'none'
            p = PALETAS_PREDETERMINADAS[selector.value]
            vista_previa.value = f"{_muestra_html(p['series'])}<br><small>{p['descripcion']}</small>"

    def _al_pulsar(_=None):
        salida.clear_output(wait=True)
        with salida:
            try:
                aplicar_seleccion(selector.value, campo_css.value)
            except ValueError as e:
                print(f'❌ Paleta no válida: {e}\nCorrija el código CSS o elija una paleta predeterminada. '
                      'No se generaron gráficos.')

    selector.observe(_actualizar_vista, names='value')
    boton.on_click(_al_pulsar)
    _actualizar_vista()
    display(HTML('<b>🎨 Seleccione una paleta de colores para los gráficos</b>'),
            widgets.VBox([selector, vista_previa, campo_css, boton, salida]))
    _al_pulsar()   # primera ejecución con la paleta por defecto
else:
    # Alternativa sin widgets: selección por consola
    opciones = list(PALETAS_PREDETERMINADAS) + [OPCION_PERSONALIZADA]
    for i, o in enumerate(opciones, 1):
        print(f'  {i}. {o}')
    eleccion = input('Número de la paleta (Enter = 1): ').strip() or '1'
    opcion = opciones[int(eleccion) - 1] if eleccion.isdigit() and 1 <= int(eleccion) <= len(opciones) else opciones[0]
    texto = ''
    if opcion == OPCION_PERSONALIZADA:
        print('Pegue el CSS y escriba FIN en una línea aparte:')
        lineas = []
        while (linea := input()).strip().upper() != 'FIN':
            lineas.append(linea)
        texto = '\n'.join(lineas)
    try:
        aplicar_seleccion(opcion, texto)
    except ValueError as e:
        print(f'❌ Paleta no válida: {e}')
```

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
