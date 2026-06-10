from vibe import Vibe

EXPENSES = [
    "uber to the airport, 42 bucks",
    "team lunch at the thai place — $86.50",
    "monthly figma subscription, 15",
    "oat milk latte 4.25",
    "flight to NYC, 320 dollars",
    "amazon: usb-c cables and a notebook, 23.99",
    "aws bill this month came out to 71.40",
    "movie tickets for two ~28",
]


def main() -> None:
    v = Vibe.openrouter(
        model="openrouter/meta-llama/llama-3.3-70b-instruct",
        packages=["pandas"],
        verbose=True,
    )

    v.spec(
        "category_report",
        context="Total spend per category, highest first, with each category's share of the grand total.",
        inputs="a list of {category, amount} dicts",
        returns="a pandas DataFrame with columns: category, total, pct_of_spend",
    )

    rows = [
        {
            "merchant": v.guess_merchant(note),
            "category": v.categorize_expense(note),
            "amount": float(v.extract_amount_usd(note)),
        }
        for note in EXPENSES
    ]

    print(f"\n{'MERCHANT':<22}{'CATEGORY':<16}AMOUNT")
    print("-" * 48)
    for r in rows:
        print(f"{r['merchant']:<22}{r['category']:<16}${r['amount']:>8.2f}")

    report = v.category_report([{k: r[k] for k in ("category", "amount")} for r in rows])
    print()
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
