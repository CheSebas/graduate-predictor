# model_utils.py — Preprocesamiento, FT-Transformer propio, MLP, métricas
import numpy as np
import pandas as pd
import random, math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# PREPROCESAMIENTO
# ──────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["academic_year", "online_access_count", "final_grade", "semester"]
CONTINUOUS_COLS  = ["test_scores", "project_grades", "assignment_completion",
                    "final_points", "engagement_score"]

def prepare_data(df: pd.DataFrame):
    df = df.drop_duplicates().dropna()
    df = df.drop(columns=["student_id"], errors="ignore")

    # LabelEncode categóricas
    encoders = {}
    cat_vocab = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        cat_vocab[col] = int(df[col].max()) + 1   # num categorías

    X = df.drop("graduate", axis=1)
    y = df["graduate"].values.astype(np.float32)

    # Split 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    # Escalar continuas
    scaler = StandardScaler()
    for split in [X_train, X_val, X_test]:
        cont_present = [c for c in CONTINUOUS_COLS if c in split.columns]
        if split is X_train:
            split[cont_present] = scaler.fit_transform(split[cont_present])
        else:
            split[cont_present] = scaler.transform(split[cont_present])

    cat_cols_present  = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    cont_cols_present = [c for c in CONTINUOUS_COLS  if c in X_train.columns]

    def to_tensors(Xd, yd):
        Xcat  = torch.tensor(Xd[cat_cols_present].values.astype(np.int64),   dtype=torch.long)
        Xcont = torch.tensor(Xd[cont_cols_present].values.astype(np.float32), dtype=torch.float32)
        yt    = torch.tensor(yd, dtype=torch.float32)
        return Xcat, Xcont, yt

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "cat_cols": cat_cols_present,
        "cont_cols": cont_cols_present,
        "cat_vocab": cat_vocab,
        "to_tensors": to_tensors,
        "n_cont": len(cont_cols_present),
    }

# ──────────────────────────────────────────────────────────────
# FT-TRANSFORMER (PyTorch puro)
# ──────────────────────────────────────────────────────────────
class FTTransformer(nn.Module):
    def __init__(self, cat_vocab: dict, n_cont: int,
                 embed_dim=32, num_heads=8, num_blocks=4,
                 ff_dim=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding por cada columna categórica
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(vocab, embed_dim) for vocab in cat_vocab.values()
        ])

        # Proyección de continuas a embed_dim
        self.cont_proj = nn.Linear(n_cont, embed_dim) if n_cont > 0 else None

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        # Head de clasificación
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat, x_cont):
        tokens = []

        # Embed categóricas
        for i, emb in enumerate(self.cat_embeds):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))   # (B,1,D)

        # Proyectar continuas como un único token
        if self.cont_proj is not None and x_cont.shape[1] > 0:
            tokens.append(self.cont_proj(x_cont).unsqueeze(1))  # (B,1,D)

        # Agregar [CLS]
        B = x_cat.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = [cls] + tokens

        x = torch.cat(tokens, dim=1)   # (B, n_tokens, D)
        x = self.transformer(x)
        cls_out = x[:, 0]              # solo el [CLS]
        return self.head(cls_out).squeeze(-1)


def train_ft_transformer(data: dict, epochs=50, batch_size=256,
                         embed_dim=32, num_heads=8, num_blocks=4,
                         lr=1e-3, patience=10,
                         progress_cb=None):
    """
    progress_cb(epoch, train_loss, val_loss, val_acc) se llama cada época.
    Retorna (model, history_dict).
    """
    to_t = data["to_tensors"]
    Xcat_tr, Xcont_tr, y_tr = to_t(data["X_train"], data["y_train"])
    Xcat_vl, Xcont_vl, y_vl = to_t(data["X_val"],   data["y_val"])

    train_ds = TensorDataset(Xcat_tr, Xcont_tr, y_tr)
    val_ds   = TensorDataset(Xcat_vl, Xcont_vl, y_vl)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    model = FTTransformer(
        cat_vocab=data["cat_vocab"],
        n_cont=data["n_cont"],
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
    ).to(DEVICE)

    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, patience_cnt, best_state = float("inf"), 0, None

    for ep in range(1, epochs + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for xc, xn, yb in train_dl:
            xc, xn, yb = xc.to(DEVICE), xn.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xc, xn), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(train_ds)
        scheduler.step()

        # ── Val ──
        model.eval()
        vl_loss, correct = 0.0, 0
        with torch.no_grad():
            for xc, xn, yb in val_dl:
                xc, xn, yb = xc.to(DEVICE), xn.to(DEVICE), yb.to(DEVICE)
                logits = model(xc, xn)
                vl_loss += criterion(logits, yb).item() * len(yb)
                correct += ((logits.sigmoid() > 0.5).float() == yb).sum().item()
        vl_loss /= len(val_ds)
        vl_acc   = correct / len(val_ds)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if progress_cb:
            progress_cb(ep, tr_loss, vl_loss, vl_acc)

        # EarlyStopping
        if vl_loss < best_val - 1e-5:
            best_val, patience_cnt = vl_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history


def train_mlp(data: dict, epochs=100, batch_size=32,
              patience=10, progress_cb=None):
    to_t = data["to_tensors"]
    Xcat_tr, Xcont_tr, y_tr = to_t(data["X_train"], data["y_train"])
    Xcat_vl, Xcont_vl, y_vl = to_t(data["X_val"],   data["y_val"])

    # Para MLP concatenamos todo
    X_tr = torch.cat([Xcat_tr.float(), Xcont_tr], dim=1)
    X_vl = torch.cat([Xcat_vl.float(), Xcont_vl], dim=1)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_vl, y_vl), batch_size=batch_size)

    n_in = X_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32),   nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1),
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, patience_cnt, best_state = float("inf"), 0, None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(train_dl.dataset)

        model.eval()
        vl_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb).squeeze(-1)
                vl_loss += criterion(logits, yb).item() * len(yb)
                correct += ((logits.sigmoid() > 0.5).float() == yb).sum().item()
        vl_loss /= len(val_dl.dataset)
        vl_acc   = correct / len(val_dl.dataset)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if progress_cb:
            progress_cb(ep, tr_loss, vl_loss, vl_acc)

        if vl_loss < best_val - 1e-5:
            best_val, patience_cnt = vl_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ──────────────────────────────────────────────────────────────
# EVALUACIÓN
# ──────────────────────────────────────────────────────────────
def _predict_ft(model, data):
    to_t = data["to_tensors"]
    Xcat, Xcont, y = to_t(data["X_test"], data["y_test"])
    model.eval()
    with torch.no_grad():
        probs = model(Xcat.to(DEVICE), Xcont.to(DEVICE)).sigmoid().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return probs, preds, y.numpy().astype(int)


def _predict_mlp(model, data):
    to_t = data["to_tensors"]
    Xcat, Xcont, y = to_t(data["X_test"], data["y_test"])
    X = torch.cat([Xcat.float(), Xcont], dim=1)
    model.eval()
    with torch.no_grad():
        probs = model(X.to(DEVICE)).squeeze(-1).sigmoid().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return probs, preds, y.numpy().astype(int)


def evaluate(model, data, model_type="ft"):
    if model_type == "ft":
        probs, preds, y_true = _predict_ft(model, data)
    else:
        probs, preds, y_true = _predict_mlp(model, data)

    metrics = {
        "Accuracy":  accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall":    recall_score(y_true, preds, zero_division=0),
        "F1 Score":  f1_score(y_true, preds, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_true, probs),
    }
    cm = confusion_matrix(y_true, preds)

    # Curvas
    fpr, tpr, _       = roc_curve(y_true, probs)
    roc_auc_val       = auc(fpr, tpr)
    prec, rec, _      = precision_recall_curve(y_true, probs)
    avg_prec          = average_precision_score(y_true, probs)
    prevalence        = y_true.mean()

    curves = {
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_val,
        "prec": prec, "rec": rec, "avg_prec": avg_prec,
        "prevalence": prevalence,
    }
    return metrics, cm, curves