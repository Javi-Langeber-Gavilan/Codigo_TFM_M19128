# Código para el desarrollo del RoboAdvisor del TFM titulado "Desarrollo y análisis de un RoboAdvisor con Algoritmos Genéticos", por Javier
# Langeber Gavilán.

#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#                                              Código del Variational Autoencoder
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------

def preprocess_data(df):
    """
    Descripción: Calcula log-retornos y escala los datos usando StandardScaler.
    Inputs: df: DataFrame de precios (NAV)
    Outputs: array escalado, scaler entrenado
    """
    log_returns = np.log(df / df.shift(1)).dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_returns.T)
    return scaled, scaler


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        Descripción: Define una VAE sencilla con codificador y decodificador simétricos.
        """
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc21 = nn.Linear(16, latent_dim)  # media
        self.fc22 = nn.Linear(16, latent_dim)  # logvar
        self.fc3 = nn.Linear(latent_dim, 16)
        self.fc4 = nn.Linear(16, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(logvar)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    """
    Descripción: Calcula la pérdida combinada de reconstrucción (MSE) + divergencia KL.
    """
    recon_loss = nn.L1Loss()(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(1)
    return recon_loss + KLD


def train_vae(data, latent_dim=2, epochs=1000, batch_size=32):
    """
    Descripción: Entrena un VAE con los datos dados.
    Inputs: data escalada, dimensión latente, hiperparámetros
    Outputs: modelo VAE entrenado
    """
    input_dim = data.shape[1]
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch  # <- ¿falta indexado?
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
    return model


def generate_synthetic(model, scaler, num_samples=135, latent_dim=2):
    """
    Descripción: Genera muestras sintéticas a partir de un VAE entrenado y las desescala.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(latent_dim, num_samples)
        samples = model.decode(z).numpy()
    synthetic_log_returns = scaler.inverse_transform(samples)
    return synthetic_log_returns


def recomponer_precios(precios_finales_reales, log_retornos_sinteticos):
    """
    Descripción: Reconstruye precios desde retornos logarítmicos sintéticos.
    """
    precios = np.exp(log_retornos_sinteticos).cumprod(axis=1)
    precios = precios * precios_finales_reales[:, None]
    return precios


def generar_fechas_sinteticas(fechas_reales, num_fechas_sinteticas):
    """
    Descripción: Genera un rango de fechas hábiles a partir del último día real.
    """
    ultima_fecha = pd.to_datetime(fechas_reales[-1])
    nuevas_fechas = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1),
                                   periods=num_fechas_sinteticas, freq='B')
    return nuevas_fechas


def compose_new_data(real_data, new_data):
    """
    Descripción: Reconstruye un nuevo DataFrame con fechas sintéticas y precios sintéticos.
    """
    ultimos_precios = real_data.iloc[-1].values
    precios_sinteticos = recomponer_precios(ultimos_precios, new_data)
    fechas_sinteticas = generar_fechas_sinteticas(real_data.index, len(precios_sinteticos))
    df_sintetico = pd.DataFrame(precios_sinteticos, columns=real_data.columns, index=fechas_sinteticas)
    df_total = pd.concat([real_data, df_sintetico])
    return df_total, df_sintetico


def get_vae_synthetic_data(generation_dataset):
    """
    Descripción: Orquesta todo el pipeline: escalado, entrenamiento, generación y recomposición.
    """
    scaled_data, scaler = preprocess_data(generation_dataset)
    vae_model = train_vae(scaled_data)
    synthetic_data = generate_synthetic(vae_model, scaler)
    total_df, df_sintetico = compose_new_data(generation_dataset, synthetic_data)
    total_df, df_sintetico = compose_new_data(generation_dataset, synthetic_data)
    return total_df, df_sintetico
