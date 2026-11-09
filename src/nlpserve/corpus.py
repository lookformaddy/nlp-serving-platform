"""Seeded synthetic corpus generator for support-ticket-style text.

Six intents, each with several hand-written templates and slot fillers
(product names, order numbers, dates, amounts). All data is synthetic —
no real tickets, no customer data, no employer content. A controlled
label-noise rate flips a small fraction of labels so the eval is honest
about noisy real-world annotation, and a fixed seed makes the corpus
fully reproducible.

Intents: billing, account_access, shipping, refund, technical_issue,
feedback.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

__all__ = ["INTENTS", "generate_corpus", "IntentExample"]

INTENTS = [
    "billing",
    "account_access",
    "shipping",
    "refund",
    "technical_issue",
    "feedback",
]

_TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for my {product} subscription this month",
        "Why is my invoice amount higher than last month for {product}",
        "My card was charged {amount} but I cancelled {product} already",
        "Please explain the extra fee on my {product} bill",
        "I need a copy of my invoice for order {order}",
        "The payment for {product} failed but money was taken from my account",
        "Can you correct the billing address on my {product} account",
        "I was charged after my free trial ended without any warning",
        "My subscription renewed at a higher price than advertised",
        "There is an unknown charge of {amount} on my statement",
    ],
    "account_access": [
        "I cannot log in to my {product} account, my password is not working",
        "Please reset my password, the reset link never arrives",
        "My account is locked after too many login attempts",
        "I lost access to the email linked to my {product} account",
        "Two factor authentication codes are not arriving on my phone",
        "How do I change the email address on my account",
        "My account was suspended and I do not know why",
        "I cannot sign in with my social login anymore",
        "Please help me recover my account, I forgot my username",
        "The verification code I received keeps saying it is expired",
    ],
    "shipping": [
        "My order {order} has not arrived yet, it was due {date}",
        "The tracking page for order {order} has not updated in days",
        "Can I change the delivery address for order {order}",
        "My package arrived damaged, what should I do",
        "How long will delivery take to my address for {product}",
        "The courier says delivered but I never received order {order}",
        "I need to reschedule the delivery of order {order}",
        "My order {order} is stuck at the sorting facility",
        "Can I pick up order {order} from a local store instead",
        "The shipping cost at checkout was much higher than expected",
    ],
    "refund": [
        "I would like a refund for order {order}, the item was faulty",
        "Please refund the {amount} charged for {product}, I returned it",
        "How long does a refund take after returning order {order}",
        "I cancelled my order {order} but have not received my refund",
        "The refund I received is less than the amount I paid",
        "Can I get a refund instead of store credit for {product}",
        "My refund was sent to the wrong card, please fix it",
        "I returned the item a week ago and the refund is still pending",
        "Please process a refund, the product never arrived",
        "I was promised a refund for {product} and it never came through",
    ],
    "technical_issue": [
        "The {product} app crashes every time I open it",
        "I keep getting an error message when I try to check out",
        "The website will not load on my phone, it shows a blank page",
        "My {product} device keeps disconnecting from wifi",
        "The app freezes when I try to upload a photo",
        "Search results are not showing the items I am looking for",
        "I cannot download the latest update for {product}",
        "The video keeps buffering and never plays properly",
        "My settings are not saving after I change them",
        "The app logs me out every few minutes on its own",
    ],
    "feedback": [
        "I love the new design of the {product} app, great work",
        "It would be great if {product} had a dark mode option",
        "Please add a way to export my data from {product}",
        "The new update made the app much faster, thank you",
        "I wish {product} supported multiple languages",
        "Great customer service, the team was very helpful",
        "Please consider adding a wishlist feature to {product}",
        "The checkout flow is much smoother now, well done",
        "I really enjoy using {product} every day",
        "A small suggestion: let users customize notifications in {product}",
    ],
}

_PRODUCTS = ["Aurora headphones", "Nimbus speaker", "Atlas laptop stand",
             "Orbit watch", "Pulse fitness band", "Halo keyboard",
             "Vertex monitor", "Echo earbuds", "Summit backpack",
             "Drift mouse"]
_AMOUNTS = ["$12.99", "$24.50", "$49.99", "$79.00", "$99.95", "$149.00"]
_DATES = ["Monday", "Tuesday", "last Friday", "yesterday", "two days ago",
          "last week"]


@dataclass(frozen=True)
class IntentExample:
    text: str
    label: str


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        product=rng.choice(_PRODUCTS),
        amount=rng.choice(_AMOUNTS),
        order=f"ORD-{rng.randint(10000, 99999)}",
        date=rng.choice(_DATES),
    )


def generate_corpus(n_per_intent: int = 200, seed: int = 42,
                    label_noise: float = 0.05,
                    test_fraction: float = 0.2
                    ) -> tuple[list[IntentExample], list[IntentExample]]:
    """Generate a seeded train/test split of synthetic ticket text.

    Returns ``(train, test)``. With ``label_noise > 0`` a fraction of
    training labels are randomly flipped to a different intent, mimicking
    annotation noise; test labels are always clean.
    """
    rng = random.Random(seed)
    train: list[IntentExample] = []
    test: list[IntentExample] = []
    for intent in INTENTS:
        templates = _TEMPLATES[intent]
        n_test = int(n_per_intent * test_fraction)
        for i in range(n_per_intent):
            text = _fill(rng.choice(templates), rng)
            label = intent
            if i >= n_test and rng.random() < label_noise:
                label = rng.choice([x for x in INTENTS if x != intent])
            example = IntentExample(text=text, label=label)
            (test if i < n_test else train).append(example)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
