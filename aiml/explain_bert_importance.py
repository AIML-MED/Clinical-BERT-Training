import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import read_config  # noqa: E402


DEMOGRAPHIC_PREFIXES = ("AGE:", "GENDER:", "RACE:", "ETHNICITY:")


def load_vocab(vocab_path: Path) -> dict[str, int]:
    return {
        line.rstrip("\n"): idx
        for idx, line in enumerate(vocab_path.read_text(encoding="utf-8").splitlines())
    }


def token_type(token: str) -> str:
    if token in {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"}:
        return "SPECIAL"
    if token.startswith(DEMOGRAPHIC_PREFIXES):
        return "DEMOGRAPHIC"
    if ":" in token:
        return token.split(":", 1)[0]
    return "OTHER"


def encode_row(row: pd.Series, vocab: dict[str, int], max_length: int):
    tokens = [str(token) for token in row["sorted_event_tokens"]]
    positions = [int(position) for position in row["day_position_tokens"]]
    if not tokens or tokens[0] != "[CLS]":
        tokens = ["[CLS]"] + tokens
        positions = [0] + positions

    tokens = tokens[:max_length]
    positions = positions[:max_length]
    unk_id = vocab.get("[UNK]", 0)
    input_ids = [vocab.get(token, unk_id) for token in tokens]
    attention_mask = [1] * len(input_ids)
    return tokens, input_ids, positions, attention_mask


@torch.no_grad()
def predict_probability(model, input_ids, position_ids, attention_mask, target_class: int, device):
    outputs = model(
        input_ids=torch.tensor([input_ids], dtype=torch.long, device=device),
        position_ids=torch.tensor([position_ids], dtype=torch.long, device=device),
        attention_mask=torch.tensor([attention_mask], dtype=torch.long, device=device),
    )
    probs = torch.softmax(outputs.logits, dim=-1)
    return float(probs[0, target_class].detach().cpu())


def integrated_gradients(
    model,
    input_ids,
    position_ids,
    attention_mask,
    target_class: int,
    device,
    steps: int,
):
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    position_ids_tensor = torch.tensor([position_ids], dtype=torch.long, device=device)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=device)

    embedding_layer = model.get_input_embeddings()
    input_embeds = embedding_layer(input_ids_tensor).detach()
    baseline = torch.zeros_like(input_embeds)
    total_gradients = torch.zeros_like(input_embeds)

    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        embeds = baseline + alpha * (input_embeds - baseline)
        embeds.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs = model(
            inputs_embeds=embeds,
            position_ids=position_ids_tensor,
            attention_mask=attention_mask_tensor,
        )
        target_logit = outputs.logits[0, target_class]
        target_logit.backward()
        total_gradients += embeds.grad.detach()

    avg_gradients = total_gradients / steps
    attributions = ((input_embeds - baseline) * avg_gradients).sum(dim=-1)[0]
    return attributions.detach().cpu().numpy()


def occlude_event_positions(
    model,
    tokens,
    input_ids,
    position_ids,
    attention_mask,
    original_probability: float,
    target_class: int,
    mask_token_id: int,
    device,
    include_demographics: bool,
):
    groups = defaultdict(list)
    for idx, (token, position) in enumerate(zip(tokens, position_ids)):
        if token == "[CLS]":
            continue
        if not include_demographics and position == 0:
            continue
        groups[position].append(idx)

    rows = []
    for position, indices in groups.items():
        occluded_ids = list(input_ids)
        for idx in indices:
            occluded_ids[idx] = mask_token_id
        occluded_probability = predict_probability(
            model,
            occluded_ids,
            position_ids,
            attention_mask,
            target_class,
            device,
        )
        event_tokens = [tokens[idx] for idx in indices]
        rows.append(
            {
                "position_id": position,
                "tokens": " | ".join(event_tokens),
                "token_count": len(event_tokens),
                "original_probability": original_probability,
                "occluded_probability": occluded_probability,
                "delta_probability": original_probability - occluded_probability,
            }
        )
    return rows


def resolve_path(path: str | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Integrated Gradients and event-level occlusion for a fine-tuned BERT classifier."
    )
    parser.add_argument("-c", "--config-path", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explain_oscar_reconstruct_v2"))
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--max-patients", type=int, default=50)
    parser.add_argument("--label-filter", type=int, default=None)
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--include-demographics", action="store_true")
    args = parser.parse_args()

    config = read_config(str(args.config_path))
    config_dir = args.config_path.resolve().parent
    aiml_dir = REPO_ROOT / "aiml"

    data_file = args.data_file
    if data_file is None:
        data_file = resolve_path(config["test_data_filepath"], aiml_dir)
    elif not data_file.is_absolute():
        data_file = (Path.cwd() / data_file).resolve()

    model_dir = args.model_dir
    if model_dir is None:
        model_dir = aiml_dir / "outputs" / "finetune" / config["finetuned_model_name"]
    elif not model_dir.is_absolute():
        model_dir = (Path.cwd() / model_dir).resolve()

    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_length = args.max_length or int(config.get("max_length", 512))
    vocab_path = model_dir / "vocab.txt"
    vocab = load_vocab(vocab_path)
    mask_token_id = vocab["[MASK]"]

    data = pd.read_parquet(data_file)
    if args.label_filter is not None:
        data = data[data[config["target_column"]] == args.label_filter]
    if args.max_patients > 0:
        data = data.head(args.max_patients)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = transformers.BertForSequenceClassification.from_pretrained(
        model_dir,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    token_rows = []
    event_rows = []
    patient_rows = []

    for row_idx, row in data.reset_index(drop=True).iterrows():
        patient_id = row["person_id"]
        label = int(row[config["target_column"]])
        tokens, input_ids, position_ids, attention_mask = encode_row(row, vocab, max_length)
        original_probability = predict_probability(
            model, input_ids, position_ids, attention_mask, args.target_class, device
        )
        prediction = int(original_probability >= float(config.get("threshold", 0.5)))

        attrs = integrated_gradients(
            model,
            input_ids,
            position_ids,
            attention_mask,
            args.target_class,
            device,
            args.ig_steps,
        )

        for token_index, (token, position_id, attr) in enumerate(zip(tokens, position_ids, attrs)):
            if token == "[CLS]":
                continue
            if not args.include_demographics and position_id == 0:
                continue
            token_rows.append(
                {
                    "patient_id": patient_id,
                    "label": label,
                    "prediction": prediction,
                    "probability": original_probability,
                    "token_index": token_index,
                    "position_id": position_id,
                    "token": token,
                    "token_type": token_type(token),
                    "ig_attribution": float(attr),
                    "ig_abs_attribution": float(abs(attr)),
                }
            )

        for event in occlude_event_positions(
            model,
            tokens,
            input_ids,
            position_ids,
            attention_mask,
            original_probability,
            args.target_class,
            mask_token_id,
            device,
            args.include_demographics,
        ):
            event["patient_id"] = patient_id
            event["label"] = label
            event["prediction"] = prediction
            event_rows.append(event)

        patient_rows.append(
            {
                "patient_id": patient_id,
                "label": label,
                "prediction": prediction,
                "probability": original_probability,
                "sequence_length": len(tokens),
            }
        )

    token_df = pd.DataFrame(token_rows)
    event_df = pd.DataFrame(event_rows)
    patient_df = pd.DataFrame(patient_rows)

    token_df.to_csv(output_dir / "token_integrated_gradients.csv", index=False)
    event_df.to_csv(output_dir / "event_occlusion.csv", index=False)
    patient_df.to_csv(output_dir / "patients.csv", index=False)

    if not token_df.empty:
        token_summary = (
            token_df.groupby(["token", "token_type"], as_index=False)
            .agg(
                frequency=("token", "size"),
                mean_ig=("ig_attribution", "mean"),
                mean_abs_ig=("ig_abs_attribution", "mean"),
            )
            .sort_values("mean_abs_ig", ascending=False)
        )
        token_summary.to_csv(output_dir / "token_ig_summary.csv", index=False)

        type_summary = (
            token_df.groupby("token_type", as_index=False)
            .agg(
                frequency=("token_type", "size"),
                mean_ig=("ig_attribution", "mean"),
                mean_abs_ig=("ig_abs_attribution", "mean"),
            )
            .sort_values("mean_abs_ig", ascending=False)
        )
        type_summary.to_csv(output_dir / "token_type_ig_summary.csv", index=False)

    if not event_df.empty:
        event_df.sort_values("delta_probability", ascending=False).to_csv(
            output_dir / "event_occlusion_ranked.csv", index=False
        )

    metadata = {
        "config_path": str(args.config_path),
        "data_file": str(data_file),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "target_class": args.target_class,
        "max_patients": args.max_patients,
        "label_filter": args.label_filter,
        "ig_steps": args.ig_steps,
        "include_demographics": args.include_demographics,
        "device": str(device),
        "patients_explained": len(patient_df),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
