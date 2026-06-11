import argparse
from pathlib import Path


def read_vocab(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def expand_vocab(base_path: Path, additions_path: Path, output_path: Path) -> None:
    base_vocab = read_vocab(base_path)
    additions = read_vocab(additions_path)

    if len(base_vocab) != len(set(base_vocab)):
        raise ValueError(f"Base vocabulary contains duplicates: {base_path}")

    expanded_vocab = list(base_vocab)
    known_tokens = set(base_vocab)
    for token in additions:
        if token not in known_tokens:
            expanded_vocab.append(token)
            known_tokens.add(token)

    output_path.write_text("\n".join(expanded_vocab) + "\n", encoding="utf-8")

    print(f"Base vocabulary: {len(base_vocab)}")
    print(f"Candidate additions: {len(additions)}")
    print(f"Appended tokens: {len(expanded_vocab) - len(base_vocab)}")
    print(f"Expanded vocabulary: {len(expanded_vocab)}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append unseen tokens to a vocabulary without changing existing token IDs."
    )
    parser.add_argument("base", type=Path, help="Existing model vocabulary.")
    parser.add_argument("additions", type=Path, help="Vocabulary containing candidate new tokens.")
    parser.add_argument("output", type=Path, help="Expanded vocabulary output path.")
    args = parser.parse_args()
    expand_vocab(args.base, args.additions, args.output)


if __name__ == "__main__":
    main()
