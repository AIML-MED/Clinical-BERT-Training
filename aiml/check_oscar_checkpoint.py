import argparse
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers import AutoModelForMaskedLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify OSCAR checkpoint weight loading.")
    parser.add_argument("--model-dir", type=Path, default=Path("../oscar_omop"))
    parser.add_argument("--expanded-vocab-size", type=int)
    args = parser.parse_args()

    checkpoint_path = args.model_dir / "model.safetensors"
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
        distance_keys = sorted(key for key in checkpoint_keys if "distance_embedding" in key)
        checkpoint_tensors = {key: checkpoint.get_tensor(key).clone() for key in distance_keys}

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_dir,
        attn_implementation="eager",
    )
    model_state = model.state_dict()

    failed_distance_keys = []
    for key, checkpoint_tensor in checkpoint_tensors.items():
        loaded_tensor = model_state.get(key)
        if loaded_tensor is None or not torch.equal(checkpoint_tensor, loaded_tensor):
            failed_distance_keys.append(key)

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    decoder_is_tied = input_weight.data_ptr() == output_weight.data_ptr()
    bias_is_tied = (
        model.cls.predictions.bias.data_ptr()
        == model.cls.predictions.decoder.bias.data_ptr()
    )

    print(f"Transformers version: {transformers.__version__}")
    print(f"Attention class: {type(model.bert.encoder.layer[0].attention.self).__name__}")
    print(f"Checkpoint distance embeddings: {len(distance_keys)}")
    print(f"Distance embeddings loaded exactly: {len(failed_distance_keys) == 0}")
    print(f"Decoder weight present in checkpoint: {'cls.predictions.decoder.weight' in checkpoint_keys}")
    print(f"Decoder tied to input embeddings: {decoder_is_tied}")
    print(f"Decoder bias present in checkpoint: {'cls.predictions.decoder.bias' in checkpoint_keys}")
    print(f"Prediction bias present in checkpoint: {'cls.predictions.bias' in checkpoint_keys}")
    print(f"Decoder bias tied to prediction bias: {bias_is_tied}")

    if args.expanded_vocab_size is not None:
        model.resize_token_embeddings(args.expanded_vocab_size)
        resized_tied = (
            model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr()
        )
        print(f"Resized embedding size: {model.get_input_embeddings().num_embeddings}")
        print(f"Decoder remains tied after resize: {resized_tied}")
        if not resized_tied:
            raise RuntimeError("Decoder became untied after resizing token embeddings")

    if failed_distance_keys:
        raise RuntimeError(
            "Distance embedding weights were not loaded exactly: "
            + ", ".join(failed_distance_keys)
        )
    if not decoder_is_tied or not bias_is_tied:
        raise RuntimeError("MLM decoder weights are not correctly tied")


if __name__ == "__main__":
    main()
