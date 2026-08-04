| System | Set | N | Exact match | WER | Filler removal F1 | Repair resolution | Repairs scored |
|---|---|---|---|---|---|---|---|
| Copy baseline | held-out synthetic | 2000 | 0.032 | 0.318 | 0.000 | 0.000 | 472 |
| Mend (flan-t5-small) | held-out synthetic | 2000 | 0.644 | 0.057 | 0.999 | 0.945 | 472 |
| Copy baseline | hard set | 100 | 0.080 | 0.689 | 0.000 | 0.000 | 88 |
| Mend (flan-t5-small) | hard set | 100 | 0.530 | 0.236 | 0.667 | 0.534 | 88 |

Exact match ignores whitespace differences and treats the sentence-final mark as optional. WER is computed over whitespace tokens with punctuation attached, so a missing comma counts as one error. Filler removal F1 counts occurrences of filler words deleted, so a filler word used literally can be a false positive. Repair resolution asks whether the reparandum is gone and the repair is present; on the synthetic split it uses the injector spans, and on the hard set, which has no spans, it uses the word-multiset difference between input and gold.
