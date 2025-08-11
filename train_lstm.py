import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import TrainConfig, fit

# train_lstm.py
from src.train import TrainConfig, fit

def main():
    cfg = TrainConfig(
        train_csv="data/splits/airline_train.csv",
        val_csv="data/splits/airline_val.csv",
        model_type="lstm",
        vocab_path="experiments/runs/lstm/vocab.json",
        save_dir="experiments/runs/lstm",
        batch_size=64,
        epochs=8,
        lr=1e-3,
        patience=3,
        # num_workers=0,  # <-- important on Windows
    )
    fit(cfg)

if __name__ == "__main__":
    # On Windows, multiprocessing requires spawn + main guard
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

