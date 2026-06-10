"""Prototype: a contact-list importer that cleans messy exported data.

A familiar chore — you exported contacts from somewhere and the rows are a mess:
inconsistent name casing, five different phone formats, mixed-case emails, the
odd broken address. This prototype normalizes and validates every row, then
prints a tidy report plus a breakdown by email provider.

The interesting part: none of the cleaning or validation logic is written by
hand. Each ``v.<something>(...)`` call is a function that doesn't exist yet —
the model writes it from the name and argument types on first call, it runs, and
the source is cached to ./.vibe_cache/ for instant reuse afterwards.

Run inside `nix develop` (so libstdc++ is on LD_LIBRARY_PATH) with an
OPENROUTER_API_KEY in your environment:

    uv run python examples/demo.py
"""

from collections import Counter

from vibe import Vibe

RAW_CONTACTS = [
    {"name": "  ada LOVELACE ", "phone": "(415) 555-0132", "email": "Ada@Gmail.com"},
    {"name": "alan turing", "phone": "415.555.0199", "email": "alan.turing@bletchley.uk"},
    {"name": "GRACE hopper", "phone": "+1 415 555 0150", "email": "grace@navy.mil"},
    {"name": "katherine Johnson", "phone": "4155550177", "email": "not-an-email"},
    {"name": "linus  torvalds", "phone": "415-555-0164", "email": "Linus@gmail.com"},
]


def main() -> None:
    v = Vibe.openrouter(
        model="openrouter/meta-llama/Llama-3.1-8B-Instruct",
        verbose=True,
    )

    providers: Counter[str] = Counter()
    rows: list[tuple[str, str, str, bool]] = []

    for raw in RAW_CONTACTS:
        name = v.normalize_full_name(raw["name"])
        phone = v.format_phone_number_e164(raw["phone"], "US")
        email = raw["email"].strip().lower()
        valid = v.is_valid_email_address(email)

        if valid:
            shown_email = v.mask_email_address(email)
            provider = v.email_provider_from_address(email)
            if provider:  # synthesized code may return None on odd input
                providers[provider] += 1
        else:
            shown_email = "<invalid>"

        rows.append((name, phone, shown_email, valid))

    # --- cleaned report -------------------------------------------------
    print(f"\n{'NAME':<20}{'PHONE':<18}{'EMAIL':<22}STATUS")
    print("-" * 66)
    for name, phone, email, valid in rows:
        # str(): synthesized functions may return None, which can't be formatted.
        status = "ok" if valid else "REVIEW"
        print(f"{str(name):<20}{str(phone):<18}{str(email):<22}{status}")

    flagged = sum(1 for *_, valid in rows if not valid)
    print(f"\n{len(rows)} contacts imported, {flagged} flagged for review")

    print("\ncontacts by email provider:")
    for provider, count in providers.most_common():
        print(f"  {provider:<12} {count}")


if __name__ == "__main__":
    main()
