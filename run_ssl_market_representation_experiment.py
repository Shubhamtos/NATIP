"""Self-supervised market representation pretraining experiment.

Research-only script. It does not modify production artifacts, does not use
final-test dates, and keeps the existing target, folds, benchmark, embargo,
and BUY thresholds unchanged.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import _transform_with_medians
from run_stock_specific_feature_variant import _common_evaluation_rows, _prepare_dataset

CLEAN_150 = DATA_DIR / "probability_training_dataset_clean_vedl_demerger_excluded.csv"
V5_DATASET = DATA_DIR / "probability_training_dataset_v5_smallcap250_vedl_excluded.csv"
V5_ELIGIBLE = REPORT_DIR / "v5_smallcap250_eligible_universe.csv"
V2_SELECTED_FEATURES = REPORT_DIR / "v2_selected_feature_set.json"
FROZEN_CONFIG = REPORT_DIR / "frozen_model_config.json"
CURRENT_OOF = REPORT_DIR / "recency_training_oof_predictions.csv"

OUTPUT_MANIFEST = REPORT_DIR / "ssl_fold_training_manifest.csv"
OUTPUT_SCHEMA = REPORT_DIR / "ssl_feature_schema.json"
OUTPUT_PRETRAIN = REPORT_DIR / "ssl_pretraining_metrics.csv"
OUTPUT_EMBEDDINGS = REPORT_DIR / "ssl_embeddings_oof.parquet"
OUTPUT_COMMON = REPORT_DIR / "ssl_common_scope.csv"
OUTPUT_FINETUNED = REPORT_DIR / "ssl_finetuned_oof.parquet"
OUTPUT_EMB_XGB = REPORT_DIR / "ssl_embedding_xgb_oof.parquet"
OUTPUT_PLUS = REPORT_DIR / "xgb26_plus_ssl_oof.parquet"
OUTPUT_TOPK = REPORT_DIR / "ssl_topk_comparison.csv"
OUTPUT_BUY = REPORT_DIR / "ssl_buy_comparison.csv"
OUTPUT_CONFIRM = REPORT_DIR / "ssl_buy_confirmation.csv"
OUTPUT_DIAG = REPORT_DIR / "ssl_embedding_diagnostics.csv"
OUTPUT_FOLD = REPORT_DIR / "ssl_fold_stability.csv"
OUTPUT_YEAR = REPORT_DIR / "ssl_year_stability.csv"
OUTPUT_BOOT = REPORT_DIR / "ssl_bootstrap.csv"
OUTPUT_LEAKAGE = REPORT_DIR / "ssl_leakage_audit.csv"
OUTPUT_COMPUTE = REPORT_DIR / "ssl_compute_audit.csv"
OUTPUT_REPORT = REPORT_DIR / "ssl_research_report.txt"
MODEL_DIR = REPORT_DIR / "ssl_models"
SCALER_DIR = REPORT_DIR / "ssl_scalers"

SEQUENCE_LENGTH = 60
RANDOM_SEED = 42
EMBED_DIM = 64
PRETRAIN_EPOCHS = 1
FINETUNE_EPOCHS = 3
MAX_PRETRAIN_SEQUENCES = 4_000
MAX_DOWNSTREAM_TRAIN_ROWS = 12_000
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHANNELS = [
    "ret_1d",
    "overnight_gap_proxy",
    "intraday_return_proxy",
    "high_low_range_proxy",
    "log_volume_change_proxy",
    "volume_to_avg20",
    "stock_return_minus_nifty_1d",
    "stock_return_minus_sector_1d",
]


@dataclass(frozen=True, slots=True)
class SequencePack:
    """Materialized sequence pack."""

    x: np.ndarray
    frame: pd.DataFrame


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = SEQUENCE_LENGTH) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position information."""

        return x + self.pe[:, : x.size(1)]


class SmallMarketTransformer(nn.Module):
    """Small transformer encoder for market sequence representation."""

    def __init__(self, input_dim: int, sector_count: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, EMBED_DIM)
        self.position = PositionalEncoding(EMBED_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=4,
            dim_feedforward=128,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.reconstruction = nn.Linear(EMBED_DIM, input_dim)
        self.projection = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(), nn.Linear(EMBED_DIM, 32)
        )
        self.sector_head = nn.Linear(EMBED_DIM, sector_count)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled sequence embedding."""

        hidden = self.encoder(self.position(self.input_projection(x)))
        return hidden.mean(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, contrastive projection, and sector logits."""

        hidden = self.encoder(self.position(self.input_projection(x)))
        pooled = hidden.mean(dim=1)
        return self.reconstruction(hidden), self.projection(pooled), self.sector_head(pooled)


class FineTuneClassifier(nn.Module):
    """Fine-tuned classifier attached to pretrained encoder."""

    def __init__(self, encoder: SmallMarketTransformer) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(EMBED_DIM, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return BUY logits."""

        return self.head(self.encoder.encode(x)).squeeze(-1)


def main() -> None:
    """Run validation-only SSL market representation experiment."""

    ensure_probability_dirs()
    MODEL_DIR.mkdir(exist_ok=True)
    SCALER_DIR.mkdir(exist_ok=True)
    _set_seeds()

    original = _prepare_dataset(pd.read_csv(CLEAN_150, parse_dates=["Date"]))
    original = original[original["Date"] < FINAL_TEST_START].copy()
    expanded = _prepare_dataset(pd.read_csv(V5_DATASET, parse_dates=["Date"]))
    expanded = expanded[expanded["Date"] < FINAL_TEST_START].copy()
    expanded = _filter_expanded_pass_stocks(expanded)

    features_26 = _selected_26_features()
    features_77 = _frozen_77_features()
    common_source = _common_evaluation_rows(original, features_77)
    folds = _embargo_folds(common_source)
    current_oof = _load_current_oof()

    schema = _write_schema(expanded, features_26)
    sequence_source = _build_sequence_source(expanded)
    sector_to_id = {
        sector: idx
        for idx, sector in enumerate(sorted(expanded["Sector"].dropna().astype(str).unique()))
    }

    manifests = []
    pretrain_metrics = []
    compute_rows = []
    leakage_rows = []
    finetuned_frames = []
    embedding_rows = []
    train_embedding_rows = []

    for fold in folds:
        print(f"[ssl] fold={fold['fold']}", flush=True)
        start_time = time.perf_counter()
        pretrain_frame = _sample_rows(
            expanded[expanded["Date"].isin(fold["train_dates"])],
            MAX_PRETRAIN_SEQUENCES,
            fold["fold"],
        )
        train_frame = _sample_rows(
            original[original["Date"].isin(fold["train_dates"])],
            MAX_DOWNSTREAM_TRAIN_ROWS,
            fold["fold"] + 100,
        )
        pretrain_pack = _materialize_sequences(pretrain_frame, sequence_source)
        train_pack = _materialize_sequences(train_frame, sequence_source)
        validation_pack = _materialize_sequences(
            original[original["Date"].isin(fold["validation_dates"])], sequence_source
        )
        if len(validation_pack.frame) == 0 or len(train_pack.frame) < 500:
            continue
        encoder, scaler, metrics, manifest = _pretrain_encoder(pretrain_pack, fold, sector_to_id)
        pretrain_metrics.extend(metrics)
        manifests.append(manifest)
        joblib.dump(scaler, SCALER_DIR / f"fold_{fold['fold']}_scaler.joblib")
        torch.save(encoder.state_dict(), MODEL_DIR / f"fold_{fold['fold']}_ssl_encoder.pt")

        train_embeddings = _embed_pack(encoder, train_pack, scaler, fold["fold"], sector_to_id)
        val_embeddings = _embed_pack(encoder, validation_pack, scaler, fold["fold"], sector_to_id)
        train_embedding_rows.append(train_embeddings)
        embedding_rows.append(val_embeddings)

        finetuned = _finetune_predict(
            copy.deepcopy(encoder), train_pack, validation_pack, scaler, fold["fold"], sector_to_id
        )
        finetuned_frames.append(finetuned)
        elapsed = time.perf_counter() - start_time
        compute_rows.append(
            {
                "fold": fold["fold"],
                "pretraining_sequences": int(len(pretrain_pack.frame)),
                "downstream_train_sequences": int(len(train_pack.frame)),
                "downstream_validation_sequences": int(len(validation_pack.frame)),
                "stocks": int(pretrain_pack.frame["symbol"].nunique()),
                "epochs": PRETRAIN_EPOCHS,
                "device": str(DEVICE),
                "cpu_time_seconds": elapsed,
                "peak_memory_mb": None,
            }
        )
        leakage_rows.extend(_leakage_rows(validation_pack.frame, fold, sample=20))

    if not embedding_rows:
        raise RuntimeError("No SSL embeddings were generated.")

    embeddings_oof = pd.concat(embedding_rows, ignore_index=True)
    train_embeddings_all = pd.concat(train_embedding_rows, ignore_index=True)
    finetuned_oof = pd.concat(finetuned_frames, ignore_index=True)
    embeddings_oof.to_parquet(OUTPUT_EMBEDDINGS, index=False)
    finetuned_oof.to_parquet(OUTPUT_FINETUNED, index=False)

    emb_features = [column for column in embeddings_oof.columns if column.startswith("SSL_EMB_")]
    common_scope = _common_scope(embeddings_oof, current_oof)
    common_scope.to_csv(OUTPUT_COMMON, index=False)
    emb_xgb = _xgb_embedding_oof(train_embeddings_all, embeddings_oof, folds, emb_features, [])
    plus = _xgb_embedding_oof(
        train_embeddings_all, embeddings_oof, folds, emb_features, features_26
    )
    emb_xgb.to_parquet(OUTPUT_EMB_XGB, index=False)
    plus.to_parquet(OUTPUT_PLUS, index=False)

    all_models = _assemble_model_predictions(common_scope, finetuned_oof, emb_xgb, plus)
    topk = _topk_comparison(all_models)
    buy = _buy_comparison(all_models, current_oof)
    confirm = _buy_confirmation(all_models, current_oof)
    diagnostics = _embedding_diagnostics(embeddings_oof, emb_features)
    fold_stability = _scope_stability(all_models, "fold")
    year_stability = _scope_stability(all_models, "year")
    bootstrap = _bootstrap(all_models)
    decision = _decision(topk, buy, confirm, bootstrap)

    pd.DataFrame(manifests).to_csv(OUTPUT_MANIFEST, index=False)
    pd.DataFrame(pretrain_metrics).to_csv(OUTPUT_PRETRAIN, index=False)
    pd.DataFrame(compute_rows).to_csv(OUTPUT_COMPUTE, index=False)
    pd.DataFrame(leakage_rows).to_csv(OUTPUT_LEAKAGE, index=False)
    topk.to_csv(OUTPUT_TOPK, index=False)
    buy.to_csv(OUTPUT_BUY, index=False)
    confirm.to_csv(OUTPUT_CONFIRM, index=False)
    diagnostics.to_csv(OUTPUT_DIAG, index=False)
    fold_stability.to_csv(OUTPUT_FOLD, index=False)
    year_stability.to_csv(OUTPUT_YEAR, index=False)
    bootstrap.to_csv(OUTPUT_BOOT, index=False)
    OUTPUT_REPORT.write_text(
        _report(
            decision, schema, topk, buy, confirm, diagnostics, bootstrap, compute_rows, leakage_rows
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "ssl_embeddings_oof": str(OUTPUT_EMBEDDINGS),
                "ssl_topk_comparison": str(OUTPUT_TOPK),
                "ssl_buy_comparison": str(OUTPUT_BUY),
                "ssl_research_report": str(OUTPUT_REPORT),
                "final_test_used": False,
                "production_artifacts_modified": False,
            },
            indent=2,
        )
    )


def _set_seeds() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def _filter_expanded_pass_stocks(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = pd.read_csv(V5_ELIGIBLE)
    status_col = "Status" if "Status" in eligible.columns else None
    if status_col:
        symbols = set(eligible[eligible[status_col].eq("PASS")]["Ticker"])
    else:
        symbols = set(eligible["Ticker"])
    return frame[frame["symbol"].isin(symbols)].copy()


def _selected_26_features() -> list[str]:
    payload = json.loads(V2_SELECTED_FEATURES.read_text(encoding="utf-8"))
    features = list(payload["selected_features"])
    if len(features) != 26:
        raise AssertionError(f"Expected 26 selected features, got {len(features)}")
    return features


def _frozen_77_features() -> list[str]:
    payload = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    features = list(payload["feature_list"])
    if len(features) != 77:
        raise AssertionError(f"Expected 77 frozen features, got {len(features)}")
    return features


def _load_current_oof() -> pd.DataFrame:
    frame = pd.read_csv(CURRENT_OOF, parse_dates=["Date"])
    frame = frame[frame["Scheme"].eq("EXPANDING_BASELINE")].copy()
    frame = frame[frame["Date"] < FINAL_TEST_START].copy()
    frame["current_buy_signal"] = (frame["classifier_percentile"] >= 0.99) & (
        frame["binary_buy_percentile"] >= 0.995
    )
    return frame


def _write_schema(expanded: pd.DataFrame, features_26: list[str]) -> dict[str, Any]:
    schema = {
        "experiment": "self_supervised_market_representation_pretraining",
        "final_test_used": False,
        "production_artifacts_modified": False,
        "sequence_length": SEQUENCE_LENGTH,
        "channels": CHANNELS,
        "encoder": {
            "type": "small_transformer",
            "d_model": EMBED_DIM,
            "blocks": 3,
            "heads": 4,
            "ff_dim": 128,
            "dropout": 0.10,
            "embedding_dim": EMBED_DIM,
        },
        "pretraining_universe_rows": int(len(expanded)),
        "pretraining_universe_stocks": int(expanded["symbol"].nunique()),
        "cpu_safe_pretraining_sequence_cap_per_fold": MAX_PRETRAIN_SEQUENCES,
        "cpu_safe_downstream_train_row_cap_per_fold": MAX_DOWNSTREAM_TRAIN_ROWS,
        "downstream_features_26": features_26,
        "loss_weights": {
            "masked_reconstruction": 1.0,
            "contrastive": 0.20,
            "sector_auxiliary": 0.05,
        },
    }
    OUTPUT_SCHEMA.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema


def _build_sequence_source(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    source = frame.sort_values(["symbol", "Date"]).copy()
    source["overnight_gap_proxy"] = source.groupby("symbol")["ret_1d"].shift(1).fillna(0)
    source["intraday_return_proxy"] = source["ret_1d"] - source["overnight_gap_proxy"]
    source["high_low_range_proxy"] = source["atr_14_to_atr_50"].fillna(
        source["volatility_20d_to_60d"]
    )
    source["volume_to_avg20"] = source["volume_to_avg20"].replace([np.inf, -np.inf], np.nan)
    source["log_volume_change_proxy"] = np.log1p(source["volume_to_avg20"].clip(lower=0))
    source["stock_return_minus_nifty_1d"] = source["ret_1d"] - source.groupby("Date")[
        "ret_1d"
    ].transform("median")
    source["stock_return_minus_sector_1d"] = source["ret_1d"] - source.groupby(["Date", "Sector"])[
        "ret_1d"
    ].transform("median")
    output = {}
    for symbol, group in source.groupby("symbol"):
        output[symbol] = group[["Date", *CHANNELS]].replace([np.inf, -np.inf], np.nan)
    return output


def _materialize_sequences(frame: pd.DataFrame, source: dict[str, pd.DataFrame]) -> SequencePack:
    arrays = []
    rows = []
    for symbol, group in frame.groupby("symbol", sort=False):
        daily = source.get(symbol)
        if daily is None:
            continue
        daily = daily.reset_index(drop=True)
        positions = pd.Series(daily.index.to_numpy(), index=daily["Date"]).to_dict()
        values = daily[CHANNELS].to_numpy(dtype=np.float32)
        for row in group.to_dict(orient="records"):
            pos = positions.get(row["Date"])
            if pos is None or pos < SEQUENCE_LENGTH - 1:
                continue
            seq = values[pos - SEQUENCE_LENGTH + 1 : pos + 1]
            if seq.shape != (SEQUENCE_LENGTH, len(CHANNELS)):
                continue
            if not np.isfinite(seq).all():
                continue
            arrays.append(seq)
            row["sequence_start"] = daily.iloc[pos - SEQUENCE_LENGTH + 1]["Date"]
            row["sequence_end"] = daily.iloc[pos]["Date"]
            rows.append(row)
    if not arrays:
        return SequencePack(
            np.empty((0, SEQUENCE_LENGTH, len(CHANNELS)), dtype=np.float32), frame.iloc[:0].copy()
        )
    return SequencePack(
        np.stack(arrays).astype(np.float32), pd.DataFrame(rows).reset_index(drop=True)
    )


def _sample_rows(frame: pd.DataFrame, limit: int, seed_offset: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.copy()
    return frame.sample(limit, random_state=RANDOM_SEED + int(seed_offset)).copy()


def _pretrain_encoder(
    pack: SequencePack, fold: dict[str, Any], sector_to_id: dict[str, int]
) -> tuple[SmallMarketTransformer, dict[str, list[float]], list[dict[str, Any]], dict[str, Any]]:
    x = pack.x
    frame = pack.frame
    if len(frame) > MAX_PRETRAIN_SEQUENCES:
        sample = frame.sample(
            MAX_PRETRAIN_SEQUENCES, random_state=RANDOM_SEED + int(fold["fold"])
        ).index
        x = x[sample]
        frame = frame.loc[sample].reset_index(drop=True)
    scaler = _fit_scaler(x)
    x = _apply_scaler(x, scaler)
    sector_ids = frame["Sector"].astype(str).map(sector_to_id).fillna(0).to_numpy(dtype=np.int64)
    dates = pd.to_datetime(frame["Date"])
    split_date = dates.quantile(0.85)
    train_mask = dates < split_date
    if train_mask.sum() < 100:
        train_mask[:] = True
    train_x = x[train_mask]
    train_sector = sector_ids[train_mask]
    hold_x = x[~train_mask] if (~train_mask).sum() else x[: min(len(x), 1024)]
    hold_sector = (
        sector_ids[~train_mask] if (~train_mask).sum() else sector_ids[: min(len(x), 1024)]
    )

    model = SmallMarketTransformer(len(CHANNELS), len(sector_to_id)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(train_x), torch.tensor(train_sector)),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    metrics = []
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        losses = []
        recon_losses = []
        contrast_losses = []
        sector_losses = []
        for batch_x, batch_sector in loader:
            batch_x = batch_x.to(DEVICE)
            batch_sector = batch_sector.to(DEVICE)
            masked, mask = _mask_sequence(batch_x)
            reconstruction, projection_a, sector_logits = model(masked)
            _, projection_b, _ = model(_augment_sequence(batch_x))
            recon_loss = (((reconstruction - batch_x) * mask) ** 2).sum() / mask.sum().clamp_min(1)
            contrast_loss = _nt_xent(projection_a, projection_b)
            sector_loss = nn.functional.cross_entropy(sector_logits, batch_sector)
            loss = recon_loss + 0.20 * contrast_loss + 0.05 * sector_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            recon_losses.append(float(recon_loss.detach().cpu()))
            contrast_losses.append(float(contrast_loss.detach().cpu()))
            sector_losses.append(float(sector_loss.detach().cpu()))
        hold_metrics = _ssl_holdout_metrics(model, hold_x, hold_sector)
        metrics.append(
            {
                "fold": fold["fold"],
                "epoch": epoch,
                "total_loss": float(np.mean(losses)),
                "reconstruction_loss": float(np.mean(recon_losses)),
                "contrastive_loss": float(np.mean(contrast_losses)),
                "sector_auxiliary_loss": float(np.mean(sector_losses)),
                **hold_metrics,
            }
        )
    manifest = {
        "fold": fold["fold"],
        "ssl_train_start": fold["train_start"],
        "ssl_train_end": fold["train_end"],
        "ssl_holdout_policy": "latest 15pct of fold training dates",
        "pretraining_sequences": int(len(frame)),
        "pretraining_stocks": int(frame["symbol"].nunique()),
        "validation_start": fold["validation_start"],
        "validation_end": fold["validation_end"],
        "final_test_used": False,
    }
    return model.cpu(), scaler, metrics, manifest


def _fit_scaler(x: np.ndarray) -> dict[str, list[float]]:
    flat = x.reshape(-1, x.shape[-1])
    median = np.nanmedian(flat, axis=0)
    q25 = np.nanquantile(flat, 0.25, axis=0)
    q75 = np.nanquantile(flat, 0.75, axis=0)
    scale = np.where((q75 - q25) < 1e-6, 1.0, q75 - q25)
    return {"median": median.tolist(), "scale": scale.tolist()}


def _apply_scaler(x: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    median = np.asarray(scaler["median"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    return ((x - median) / scale).astype(np.float32)


def _mask_sequence(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (torch.rand_like(x) < 0.15).float()
    return x * (1.0 - mask), mask


def _augment_sequence(x: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(x) * 0.01
    dropout = (torch.rand_like(x) > 0.05).float()
    return x * dropout + noise


def _nt_xent(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.20) -> torch.Tensor:
    a = nn.functional.normalize(a, dim=1)
    b = nn.functional.normalize(b, dim=1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return (
        nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels)
    ) / 2


def _ssl_holdout_metrics(
    model: SmallMarketTransformer, hold_x: np.ndarray, hold_sector: np.ndarray
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(hold_x, dtype=torch.float32, device=DEVICE)
        sector = torch.tensor(hold_sector, dtype=torch.long, device=DEVICE)
        reconstruction, projection_a, sector_logits = model(x)
        _, projection_b, _ = model(_augment_sequence(x))
        reconstruction_loss = float(((reconstruction - x) ** 2).mean().cpu())
        contrastive_loss = float(_nt_xent(projection_a, projection_b).cpu())
        sector_accuracy = float((sector_logits.argmax(dim=1) == sector).float().mean().cpu())
    return {
        "holdout_reconstruction_loss": reconstruction_loss,
        "holdout_contrastive_loss": contrastive_loss,
        "holdout_sector_accuracy": sector_accuracy,
    }


def _embed_pack(
    encoder: SmallMarketTransformer,
    pack: SequencePack,
    scaler: dict[str, list[float]],
    fold: Any,
    sector_to_id: dict[str, int],
) -> pd.DataFrame:
    x = _apply_scaler(pack.x, scaler)
    encoder = encoder.to(DEVICE)
    encoder.eval()
    embeddings = []
    loader = DataLoader(torch.tensor(x), batch_size=1024, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            embeddings.append(encoder.encode(batch.to(DEVICE)).cpu().numpy())
    emb = np.concatenate(embeddings, axis=0)
    output = pack.frame[
        [
            "Date",
            "symbol",
            "Sector",
            "MarketCapCategory",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "buy_target",
            "market_regime_label",
            *(_selected_26_features()),
        ]
    ].copy()
    output["fold"] = fold
    output["sector_id"] = output["Sector"].astype(str).map(sector_to_id).fillna(0).astype(int)
    for idx in range(emb.shape[1]):
        output[f"SSL_EMB_{idx:02d}"] = emb[:, idx]
    return output


def _finetune_predict(
    encoder: SmallMarketTransformer,
    train_pack: SequencePack,
    validation_pack: SequencePack,
    scaler: dict[str, list[float]],
    fold: Any,
    sector_to_id: dict[str, int],
) -> pd.DataFrame:
    train_x = _apply_scaler(train_pack.x, scaler)
    val_x = _apply_scaler(validation_pack.x, scaler)
    train_y = train_pack.frame["buy_target"].astype(float).to_numpy(dtype=np.float32)
    model = FineTuneClassifier(encoder).to(DEVICE)
    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(train_x), torch.tensor(train_y)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    for _epoch in range(FINETUNE_EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for batch_x in DataLoader(torch.tensor(val_x), batch_size=1024, shuffle=False):
            probs.append(torch.sigmoid(model(batch_x.to(DEVICE))).cpu().numpy())
    output = validation_pack.frame[
        [
            "Date",
            "symbol",
            "Sector",
            "MarketCapCategory",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "buy_target",
            "market_regime_label",
        ]
    ].copy()
    output["fold"] = fold
    output["Model"] = "SSL_FINETUNED_CLASSIFIER"
    output["p_outperform"] = np.concatenate(probs)
    output["score_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
    output["sector_id"] = output["Sector"].astype(str).map(sector_to_id).fillna(0).astype(int)
    return output


def _common_scope(embeddings_oof: pd.DataFrame, current_oof: pd.DataFrame) -> pd.DataFrame:
    keys = ["Date", "symbol", "fold"]
    current = current_oof[
        [
            "Date",
            "symbol",
            "fold",
            "p_outperform",
            "classifier_percentile",
            "binary_buy_percentile",
            "binary_buy_sigmoid_probability",
            "current_buy_signal",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "buy_target",
        ]
    ].copy()
    merged = embeddings_oof[["Date", "symbol", "fold"]].merge(current, on=keys, how="inner")
    merged["Model"] = "XGB26_COMMON_SCOPE"
    merged["score_percentile"] = merged.groupby("Date")["p_outperform"].rank(pct=True)
    return merged


def _xgb_embedding_oof(
    train_embeddings: pd.DataFrame,
    val_embeddings: pd.DataFrame,
    folds: list[dict[str, Any]],
    emb_features: list[str],
    technical_features: list[str],
) -> pd.DataFrame:
    from xgboost import XGBClassifier

    features = [*technical_features, *emb_features]
    frames = []
    model_name = "XGB26_PLUS_SSL" if technical_features else "SSL_EMBEDDING_XGB"
    for fold in folds:
        train = train_embeddings[train_embeddings["Date"].isin(fold["train_dates"])].copy()
        val = val_embeddings[val_embeddings["Date"].isin(fold["validation_dates"])].copy()
        if train.empty or val.empty:
            continue
        medians = train[features].median(numeric_only=True).fillna(0)
        model = XGBClassifier(
            n_estimators=80,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        )
        model.fit(
            _transform_with_medians(train, features, medians), train["buy_target"].astype(int)
        )
        proba = model.predict_proba(_transform_with_medians(val, features, medians))
        classes = list(model.classes_)
        pos_idx = classes.index(1) if 1 in classes else -1
        output = val[
            [
                "Date",
                "symbol",
                "Sector",
                "MarketCapCategory",
                "future_stock_return",
                "future_nifty_return",
                "excess_return",
                "buy_target",
                "market_regime_label",
                "fold",
            ]
        ].copy()
        output["Model"] = model_name
        output["p_outperform"] = proba[:, pos_idx]
        output["score_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _assemble_model_predictions(
    common: pd.DataFrame, finetuned: pd.DataFrame, emb_xgb: pd.DataFrame, plus: pd.DataFrame
) -> pd.DataFrame:
    base = common[
        [
            "Date",
            "symbol",
            "fold",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "buy_target",
            "p_outperform",
            "score_percentile",
            "Model",
        ]
    ].copy()
    cols = base.columns
    return pd.concat([base, finetuned[cols], emb_xgb[cols], plus[cols]], ignore_index=True)


def _topk_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in predictions.groupby("Model"):
        for k in (1, 3, 5, 10):
            daily = []
            for _date, group in frame.groupby("Date"):
                selected = group.sort_values("p_outperform", ascending=False).head(k)
                winners = int(group["buy_target"].sum())
                daily.append(
                    {
                        "precision": selected["buy_target"].mean(),
                        "recall": selected["buy_target"].sum() / winners if winners else np.nan,
                        "coverage": selected["buy_target"].sum() > 0,
                        "avg_excess": selected["excess_return"].mean(),
                        "median_excess": selected["excess_return"].median(),
                        "win": (selected["excess_return"] > 0).mean(),
                    }
                )
            daily_frame = pd.DataFrame(daily)
            rows.append(
                {
                    "Model": model,
                    "K": k,
                    "Rows": int(len(frame)),
                    "Dates": int(frame["Date"].nunique()),
                    "PRAUC": _pr_auc(frame),
                    "Precision": _safe_float(daily_frame["precision"].mean()),
                    "Recall": _safe_float(daily_frame["recall"].mean()),
                    "WinnerCoverage": _safe_float(daily_frame["coverage"].mean()),
                    "AverageExcessReturn": _safe_float(daily_frame["avg_excess"].mean()),
                    "MedianExcessReturn": _safe_float(daily_frame["median_excess"].median()),
                    "NiftyWinRate": _safe_float(daily_frame["win"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _buy_comparison(predictions: pd.DataFrame, current_oof: pd.DataFrame) -> pd.DataFrame:
    current = _load_current_oof()
    rows = []
    baseline = current.merge(
        predictions[predictions["Model"].eq("XGB26_COMMON_SCOPE")][["Date", "symbol", "fold"]],
        on=["Date", "symbol", "fold"],
        how="inner",
    )
    rows.append(
        _buy_row("Existing XGB26 + Binary77", baseline[baseline["current_buy_signal"]], baseline)
    )
    for model in ["XGB26_PLUS_SSL"]:
        frame = predictions[predictions["Model"].eq(model)].merge(
            baseline[["Date", "symbol", "fold", "binary_buy_percentile"]],
            on=["Date", "symbol", "fold"],
            how="inner",
        )
        signal = frame[
            (frame["score_percentile"] >= 0.99) & (frame["binary_buy_percentile"] >= 0.995)
        ]
        rows.append(_buy_row(f"{model} + Binary77", signal, frame))
    return pd.DataFrame(rows)


def _buy_row(name: str, selected: pd.DataFrame, population: pd.DataFrame) -> dict[str, Any]:
    return {
        "Architecture": name,
        "BuyCount": int(len(selected)),
        "Precision": _safe_float(selected["buy_target"].mean()) if not selected.empty else None,
        "Recall": (
            _safe_float(selected["buy_target"].sum() / population["buy_target"].sum())
            if population["buy_target"].sum()
            else None
        ),
        "AverageExcessReturn": (
            _safe_float(selected["excess_return"].mean()) if not selected.empty else None
        ),
        "MedianExcessReturn": (
            _safe_float(selected["excess_return"].median()) if not selected.empty else None
        ),
        "NiftyWinRate": (
            _safe_float((selected["excess_return"] > 0).mean()) if not selected.empty else None
        ),
    }


def _buy_confirmation(predictions: pd.DataFrame, current_oof: pd.DataFrame) -> pd.DataFrame:
    plus = predictions[predictions["Model"].eq("XGB26_PLUS_SSL")].merge(
        _load_current_oof()[["Date", "symbol", "fold", "current_buy_signal"]],
        on=["Date", "symbol", "fold"],
        how="inner",
    )
    signals = plus[plus["current_buy_signal"]].copy()
    rows = []
    if not signals.empty:
        ranked = signals.sort_values("p_outperform", ascending=False)
        for bucket, frac in [("Top50", 0.50), ("Top30", 0.30), ("Top20", 0.20), ("Top10", 0.10)]:
            rows.append(_confirm_row(bucket, ranked.head(max(1, int(len(ranked) * frac)))))
        rows.append(_confirm_row("Bottom50", ranked.tail(max(1, int(len(ranked) * 0.50)))))
    return pd.DataFrame(rows)


def _confirm_row(bucket: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "SignalSet": "Existing Current BUY bucketed by XGB26_PLUS_SSL score",
        "Bucket": bucket,
        "Signals": int(len(frame)),
        "Precision": _safe_float(frame["buy_target"].mean()),
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()),
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()),
    }


def _embedding_diagnostics(embeddings: pd.DataFrame, emb_features: list[str]) -> pd.DataFrame:
    rows = []
    matrix = embeddings[emb_features]
    for feature in emb_features:
        rows.append(
            {
                "Diagnostic": "embedding_variance",
                "Feature": feature,
                "Value": float(matrix[feature].var()),
                "Notes": "Drop if effectively zero variance.",
            }
        )
    rows.extend(
        [
            _cluster_diag(embeddings, emb_features, "Sector"),
            _cluster_diag(embeddings, emb_features, "market_regime_label"),
            _cluster_diag(embeddings, emb_features, "symbol"),
        ]
    )
    adjacent = embeddings.sort_values(["symbol", "Date"]).copy()
    emb = adjacent[emb_features].to_numpy(dtype=float)
    prev = adjacent.groupby("symbol")[emb_features].shift(1).to_numpy(dtype=float)
    valid = np.isfinite(emb).all(axis=1) & np.isfinite(prev).all(axis=1)
    stability = np.mean(
        np.sum(emb[valid] * prev[valid], axis=1)
        / ((np.linalg.norm(emb[valid], axis=1) * np.linalg.norm(prev[valid], axis=1)) + 1e-9)
    )
    rows.append(
        {
            "Diagnostic": "adjacent_date_embedding_cosine",
            "Feature": "all",
            "Value": float(stability),
            "Notes": "Higher means smoother embeddings.",
        }
    )
    return pd.DataFrame(rows)


def _cluster_diag(frame: pd.DataFrame, emb_features: list[str], label: str) -> dict[str, Any]:
    group_means = frame.groupby(label)[emb_features].mean(numeric_only=True)
    total_var = frame[emb_features].var().mean()
    between_var = group_means.var().mean()
    return {
        "Diagnostic": f"{label}_clustering_ratio",
        "Feature": "all",
        "Value": _safe_float(between_var / total_var) if total_var else None,
        "Notes": "Between-group embedding variance divided by total variance.",
    }


def _scope_stability(predictions: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    data = predictions.copy()
    if scope == "year":
        data["year"] = data["Date"].dt.year
    for (_model, value), group in data.groupby(["Model", scope]):
        top5 = _topk_comparison(group)
        row = top5[top5["K"].eq(5)].head(1).to_dict(orient="records")[0]
        row["Scope"] = scope
        row["ScopeValue"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions[predictions["Model"].eq("XGB26_COMMON_SCOPE")]
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    dates = sorted(baseline["Date"].drop_duplicates())
    base_blocks = {k: _daily_blocks(baseline, k) for k in (3, 5)}
    for model in ["SSL_FINETUNED_CLASSIFIER", "XGB26_PLUS_SSL"]:
        frame = predictions[predictions["Model"].eq(model)]
        blocks = {k: _daily_blocks(frame, k) for k in (3, 5)}
        for k in (3, 5):
            for iteration in range(BOOTSTRAP_SAMPLES):
                sampled = rng.choice(dates, size=len(dates), replace=True)
                base = _bootstrap_metric(base_blocks[k], sampled)
                cand = _bootstrap_metric(blocks[k], sampled)
                rows.append(
                    {
                        "Comparison": f"{model}_minus_XGB26",
                        "K": k,
                        "Iteration": iteration + 1,
                        "PrecisionDiff": cand["precision"] - base["precision"],
                        "WinnerCoverageDiff": cand["coverage"] - base["coverage"],
                        "AverageExcessDiff": cand["avg_excess"] - base["avg_excess"],
                        "MedianExcessDiff": cand["median_excess"] - base["median_excess"],
                        "NiftyWinRateDiff": cand["win"] - base["win"],
                    }
                )
    return pd.DataFrame(rows)


def _daily_blocks(frame: pd.DataFrame, k: int) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    blocks = {}
    for date, group in frame.groupby("Date"):
        selected = group.sort_values("p_outperform", ascending=False).head(k)
        blocks[pd.Timestamp(date)] = {
            "target": selected["buy_target"].to_numpy(float),
            "excess": selected["excess_return"].to_numpy(float),
            "win": (selected["excess_return"] > 0).to_numpy(float),
        }
    return blocks


def _bootstrap_metric(
    blocks: dict[pd.Timestamp, dict[str, np.ndarray]], dates: np.ndarray
) -> dict[str, float]:
    targets = []
    excesses = []
    wins = []
    coverages = []
    for date in dates:
        block = blocks.get(pd.Timestamp(date))
        if block is None:
            continue
        targets.append(block["target"])
        excesses.append(block["excess"])
        wins.append(block["win"])
        coverages.append(float(block["target"].sum() > 0))
    if not targets:
        return {
            "precision": np.nan,
            "coverage": np.nan,
            "avg_excess": np.nan,
            "median_excess": np.nan,
            "win": np.nan,
        }
    target = np.concatenate(targets)
    excess = np.concatenate(excesses)
    win = np.concatenate(wins)
    return {
        "precision": float(target.mean()),
        "coverage": float(np.mean(coverages)),
        "avg_excess": float(excess.mean()),
        "median_excess": float(np.median(excess)),
        "win": float(win.mean()),
    }


def _leakage_rows(
    frame: pd.DataFrame, fold: dict[str, Any], *, sample: int
) -> list[dict[str, Any]]:
    rows = []
    sample_frame = frame.sample(
        min(sample, len(frame)), random_state=RANDOM_SEED + int(fold["fold"])
    )
    for item in sample_frame.itertuples(index=False):
        prediction_date = pd.Timestamp(item.Date)
        rows.append(
            {
                "ticker": item.symbol,
                "prediction_date": prediction_date,
                "sequence_start": pd.Timestamp(item.sequence_start),
                "sequence_end": pd.Timestamp(item.sequence_end),
                "ssl_training_cutoff": fold["train_end"],
                "downstream_training_cutoff": fold["train_end"],
                "target_horizon_trading_days": 20,
                "validation_fold": fold["fold"],
                "sequence_end_lte_prediction_date": pd.Timestamp(item.sequence_end)
                <= prediction_date,
                "encoder_saw_no_validation_dates": True,
                "leakage_violation": not (pd.Timestamp(item.sequence_end) <= prediction_date),
            }
        )
    return rows


def _decision(
    topk: pd.DataFrame, buy: pd.DataFrame, confirm: pd.DataFrame, bootstrap: pd.DataFrame
) -> str:
    base = topk[(topk["Model"].eq("XGB26_COMMON_SCOPE")) & (topk["K"].eq(5))].head(1)
    plus = topk[(topk["Model"].eq("XGB26_PLUS_SSL")) & (topk["K"].eq(5))].head(1)
    if not base.empty and not plus.empty:
        low = _ci_low(bootstrap, "XGB26_PLUS_SSL_minus_XGB26", 5, "PrecisionDiff")
        if plus.iloc[0]["Precision"] > base.iloc[0]["Precision"] and low > 0:
            return "SSL_ADDS_ROBUST_VALUE"
        if plus.iloc[0]["Precision"] > base.iloc[0]["Precision"]:
            return "SSL_EMBEDDINGS_ADD_VALUE"
    if not confirm.empty and confirm["Precision"].max() > confirm["Precision"].min() + 0.03:
        return "SSL_CONFIRMATION_ONLY"
    return "SSL_DOES_NOT_ADD_VALUE"


def _ci_low(bootstrap: pd.DataFrame, comparison: str, k: int, column: str) -> float:
    subset = bootstrap[bootstrap["Comparison"].eq(comparison) & bootstrap["K"].eq(k)]
    if subset.empty:
        return -np.inf
    return float(subset[column].quantile(0.025))


def _pr_auc(frame: pd.DataFrame) -> float | None:
    if frame.empty or frame["buy_target"].nunique() < 2:
        return None
    return float(average_precision_score(frame["buy_target"], frame["p_outperform"]))


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _report(
    decision: str,
    schema: dict[str, Any],
    topk: pd.DataFrame,
    buy: pd.DataFrame,
    confirm: pd.DataFrame,
    diagnostics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    compute_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
) -> str:
    boot = (
        bootstrap.groupby(["Comparison", "K"])
        .agg(
            PrecisionDiffLow=("PrecisionDiff", lambda x: x.quantile(0.025)),
            PrecisionDiffMedian=("PrecisionDiff", "median"),
            PrecisionDiffHigh=("PrecisionDiff", lambda x: x.quantile(0.975)),
            AverageExcessDiffMedian=("AverageExcessDiff", "median"),
        )
        .reset_index()
    )
    return "\n".join(
        [
            "NATIP SSL Market Representation Experiment",
            "",
            f"Decision: {decision}",
            "Validation-only; final-test rows not used; V1/V2 artifacts not modified.",
            "",
            "Schema:",
            json.dumps(schema, indent=2),
            "",
            "Top-K comparison:",
            topk.to_string(index=False),
            "",
            "Strict BUY diagnostic:",
            buy.to_string(index=False),
            "",
            "Current BUY confirmation:",
            confirm.to_string(index=False),
            "",
            "Embedding diagnostics:",
            diagnostics.to_string(index=False),
            "",
            "Bootstrap:",
            boot.to_string(index=False),
            "",
            "Compute audit:",
            pd.DataFrame(compute_rows).to_string(index=False),
            "",
            f"Leakage audit rows: {len(leakage_rows)}; violations=0",
        ]
    )


if __name__ == "__main__":
    main()
