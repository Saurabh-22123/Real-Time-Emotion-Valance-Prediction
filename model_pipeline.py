# =============================================================
#         Valence Regression — RF Baseline vs LSTM      
# =============================================================
# Compares a Random Forest baseline against a single-layer
# LSTM on pre-extracted visual features for valence prediction.
# Metrics reported: MSE, RMSE, MAE on held-out test set.
# =============================================================

# =========================
#  1. Importing Libraries
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ======================
#  2. Loading the Data
# ======================
train_df = pd.read_csv('visual_train_features.csv')
val_df   = pd.read_csv('visual_val_features.csv')
test_df  = pd.read_csv('visual_test_features.csv')

TARGET = 'unused_target_label'    # column name for the regression target variable


# ========================
#  3. Cleaning the Data
# ========================
for df in [train_df, val_df, test_df]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)   # replace infinite values with NaN so they can be imputed
    df.fillna(df.median(numeric_only=True), inplace=True)  # fill NaN with column median to avoid dropping rows

feature_cols = [c for c in train_df.columns if c != TARGET]  # all columns except the target

X_train = train_df[feature_cols].values
y_train = train_df[TARGET].values
X_val   = val_df[feature_cols].values
y_val   = val_df[TARGET].values
X_test  = test_df[feature_cols].values
y_test  = test_df[TARGET].values

X_trainval = np.concatenate([X_train, X_val], axis=0)  # merge train + val for Random Forest fitting
y_trainval = np.concatenate([y_train, y_val], axis=0)


# =============
#  4. Scaling
# =============
scaler = StandardScaler()
X_trainval_sc = scaler.fit_transform(X_trainval)  # fit on train+val, then transform
X_train_sc    = scaler.transform(X_train)
X_val_sc      = scaler.transform(X_val)
X_test_sc     = scaler.transform(X_test)


# ============================
#  5. Random Forest Baseline
# ============================
rf_model = RandomForestRegressor(
    n_estimators=200,  # number of trees
    max_depth=12,      # limit depth to control overfitting
    random_state=42,
    n_jobs=-1,         # use all available CPU cores
)
rf_model.fit(X_trainval_sc, y_trainval)


y_pred_rf = rf_model.predict(X_test_sc)
mse_rf    = mean_squared_error(y_test, y_pred_rf)
rmse_rf   = np.sqrt(mse_rf)
mae_rf    = mean_absolute_error(y_test, y_pred_rf)

print(f"\n{'Model':<20} {'MSE':>11} {'RMSE':>11} {'MAE':>11}")
print("-" * 56)
print(f"{'RF (Baseline)':<20} {mse_rf:>11.6f} {rmse_rf:>11.6f} {mae_rf:>11.6f}")


# =======================
#  6. LSTM Configuration
# =======================
SEQ_LEN = 1                    # each sample treated as a single time step
N_FEAT  = X_train_sc.shape[1]  # number of input features
HIDDEN  = 16                   # LSTM hidden units
LAYERS  = 1                    # stacked LSTM layers
DROPOUT = 0.5                  # dropout probability for regularisation
EPOCHS  = 10
BATCH   = 128
LR      = 0.01                 # Adam learning rate

def make_tensor_ds(X, y):
    """Wrap numpy arrays into a PyTorch TensorDataset for DataLoader."""
    Xt = torch.tensor(X.reshape(-1, SEQ_LEN, N_FEAT), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(Xt, yt)


train_loader = DataLoader(make_tensor_ds(X_train_sc, y_train), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(make_tensor_ds(X_val_sc,   y_val),   batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(make_tensor_ds(X_test_sc,  y_test),  batch_size=BATCH, shuffle=False)


class ValenceLSTM(nn.Module):
    """Single-layer LSTM with dropout and a linear output head."""

    def __init__(self, n_feat, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])  # take the last time step
        return self.fc(out)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = ValenceLSTM(N_FEAT, HIDDEN, LAYERS, DROPOUT).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ========================
#  7. Training the Model
# ========================
for epoch in range(1, EPOCHS + 1):
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()


# =================
#  8. Evaluation
# =================
model.eval()
lstm_preds = []
with torch.no_grad():
    for Xb, _ in test_loader:
        lstm_preds.append(model(Xb.to(device)).cpu().numpy())

y_pred_lstm = np.concatenate(lstm_preds).flatten()
mse_lstm    = mean_squared_error(y_test, y_pred_lstm)
rmse_lstm   = np.sqrt(mse_lstm)
mae_lstm    = mean_absolute_error(y_test, y_pred_lstm)

print(f"\n{'Model':<20} {'MSE':>11} {'RMSE':>11} {'MAE':>11}")
print("-" * 56)
print(f"{'RF (Baseline)':<20} {mse_rf:>11.6f} {rmse_rf:>11.6f} {mae_rf:>11.6f}")
print(f"{'LSTM':<20} {mse_lstm:>11.6f} {rmse_lstm:>11.6f} {mae_lstm:>11.6f}")


# ===================
#  9. Visualisation
# ===================
metrics     = ['MSE', 'RMSE', 'MAE']
rf_scores   = [mse_rf,   rmse_rf,   mae_rf]
lstm_scores = [mse_lstm, rmse_lstm, mae_lstm]

x     = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
bars_rf   = ax.bar(x - width / 2, rf_scores,   width, label='RF (Baseline)', color='#378ADD')
bars_lstm = ax.bar(x + width / 2, lstm_scores, width, label='LSTM',          color='#D85A30')

ax.bar_label(bars_rf,   fmt='%.4f', padding=3, fontsize=8)
ax.bar_label(bars_lstm, fmt='%.4f', padding=3, fontsize=8)

ax.set_title('RF Baseline vs LSTM — Test Set Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Error')
ax.legend()
plt.tight_layout()
plt.show()
