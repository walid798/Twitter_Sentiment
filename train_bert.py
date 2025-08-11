# train_bert.py
from src.train import TrainConfig, fit

if __name__ == "__main__":
    cfg = TrainConfig(
        train_csv="data/splits/airline_train.csv",
        val_csv="data/splits/airline_val.csv",
        model_type="bert",
        pretrained_name="bert-base-uncased",
        save_dir="experiments/runs/bert",
        batch_size=32,
        epochs=4,
        lr=2e-5,
        warmup_ratio=0.1,
        patience=2,
        max_len=128,
        mixed_precision=True,
        # num_workers=0,  # Windows-safe
    )
    fit(cfg)
