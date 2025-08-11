# scripts/eval_model.py
import argparse
import os
import torch

from src.evaluate import evaluate_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    ap.add_argument("--split_csv", type=str, required=True, help="CSV to evaluate on (val or test)")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to save metrics/plots")
    ap.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "bert"])
    ap.add_argument("--vocab_path", type=str, default="experiments/runs/lstm/vocab.json")
    ap.add_argument("--pretrained_name", type=str, default="bert-base-uncased")
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split_csv=args.split_csv,
        out_dir=args.out_dir,
        model_type=args.model_type,
        max_len=args.max_len,
        text_col=args.text_col,
        label_col=args.label_col,
        vocab_path=args.vocab_path,
        pretrained_name=args.pretrained_name,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    # Windows-safe guard
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
