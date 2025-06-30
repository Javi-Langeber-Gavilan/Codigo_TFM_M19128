# Código para el desarrollo del RoboAdvisor del TFM titulado "Desarrollo y análisis de un RoboAdvisor con Algoritmos Genéticos", por Javier
# Langeber Gavilán.

#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#                                              Código de la limpieza de datos
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------

def get_gross_data():
    """
    Descripción: Esta función permite obtener los datos en bruto scrapeados desde AWS S3 usando boto3.
    Inputs: Ninguno directamente (lee desde un archivo config.ini)
    Outputs: master_data: DataFrame con los NAVs brutos de fondos
    """
    config = configparser.ConfigParser()
    config.read("RUTA DE CONFIG")

    s3 = boto3.resource(
        service_name='s3',
        region_name=config.get('AWS-S3', 'region_name'),
        aws_access_key_id=config.get('AWS-S3', 'aws_access_key_id'),
        aws_secret_access_key=config.get('AWS-S3', 'aws_secret_access_key'),
    )

    master_data_data_obj = s3.Bucket(config.get('AWS-S3', 's3_bucket')).Object('master_data.csv').get()
    master_data = pd.read_csv(master_data_data_obj['Body'], index_col=1)
    return master_data


def get_cleaned_homogenized_normalised_data(av_gross_master_data=False, gross_master_data=None,
                                            max_upper_mov=110, max_lower_mov=180, initial_cap=1000.0, norm=False):
    """
    Descripción: Limpia, homogeneiza y normaliza datos NAV descargados desde AWS o recibidos como parámetro.
    Inputs:
        - av_gross_master_data: [bool] Si se proporciona un DataFrame en bruto.
        - gross_master_data: [DataFrame] NAVs brutos opcionales si no se quiere descargar.
        - max_upper_mov: [int] límite superior de variaciones diarias.
        - max_lower_mov: [int] límite inferior de variaciones diarias.
        - initial_cap: [float] capital base para normalización.
        - norm: [bool] indica si se desea normalizar o no los datos finales.
    Outputs:
        - master_data: DataFrame limpio y normalizado/homogeneizado listo para análisis.
    """

    config = configparser.ConfigParser()
    config.read("RUTA DE CONFIG")

    s3 = boto3.resource(
        service_name='s3',
        region_name=config.get('AWS-S3', 'region_name'),
        aws_access_key_id=config.get('AWS-S3', 'aws_access_key_id'),
        aws_secret_access_key=config.get('AWS-S3', 'aws_secret_access_key'),
    )

    # Carga de datos en bruto desde S3 si no se ha proporcionado manualmente
    if av_gross_master_data == False:
        master_data_data_obj = s3.Bucket(config.get('AWS-S3', 's3_bucket')).Object('master_data.csv').get()
        master_data = pd.read_csv(master_data_data_obj['Body'], index_col=0)
    else:
        master_data = gross_master_data.copy()

    # Establece índices de tipo fecha
    master_data.index = pd.DatetimeIndex(master_data.index, freq='B')

    # Eliminar fines de semana
    seleccion = master_data.index.dayofweek < 5
    master_data = master_data.loc[seleccion, :]

    # Limpieza de columnas/filas vacías
    master_data.dropna(axis=1, how="all", inplace=True)
    master_data.dropna(axis=0, how="any", inplace=True)

    # Relleno de NaNs por métodos de interpolación
    master_data = master_data.fillna(method="ffill")
    master_data = master_data.fillna(method="bfill", limit=3)

    # Cálculo de rentabilidades diarias
    master_returns = np.log(master_data).diff()
    master_returns.dropna(how="any", inplace=True)

    # Eliminación de outliers
    max_rent_diaria = np.log(max_upper_mov / 100)
    min_rent_diaria = np.log(100 / max_lower_mov)
    master_returns[(master_returns > max_rent_diaria) | (master_returns < min_rent_diaria)] = np.nan

    # Recompone NAV a partir de los retornos limpios
    master_data = np.exp(master_returns.cumsum()) * master_data.iloc[0, :]

    # Inserta primer valor original
    first_date = pd.DataFrame(master_data.iloc[0, :]).T
    first_date.index = [pd.to_datetime(master_data.index[0])]
    master_data = pd.concat([first_date, master_data], axis=0)
    master_data = master_data.loc[~master_data.index.duplicated(keep='last')]

    if norm:
        # Normaliza todos los NAVs a un capital inicial
        composed_master_date_returns = np.exp(np.log(master_data).diff().dropna().cumsum()) * initial_cap
        first_row = pd.DataFrame([[initial_cap] * master_data.shape[1]], columns=master_data.columns)
        first_row.index = [pd.to_datetime(master_data.index[0])]
        master_data = pd.concat([first_row, composed_master_date_returns])
        
    return master_data
