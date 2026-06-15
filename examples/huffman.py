"""Functions that depend on each other: decode is synthesized *against the
actual source* of encode (via ``uses=``), so its format always matches."""

from vibe import Vibe


def main() -> None:
    v = Vibe.openrouter(
        model="openrouter/meta-llama/llama-3.3-70b-instruct",
        verbose=True,
    )

    v.spec(
        "huffman_encode",
        inputs="str: a string to compress",
        returns=(
            "(dict[str, int], str): per-character occurrence counts and the "
            "compressed bit-string, as a tuple"
        ),
    )

    v.spec(
        "huffman_decode",
        uses=["huffman_encode"],  # shown encode's exact source at synthesis time
        inputs=(
            "(dict[str, int], str): the counts dict and compressed string "
            "exactly as produced by huffman_encode"
        ),
        returns="str: the original, decompressed string",
    )

    compressed = v.huffman_encode("hello world")
    print(compressed)

    decompressed = v.huffman_decode(compressed)
    print(decompressed)
    assert decompressed == "hello world"


if __name__ == "__main__":
    main()
