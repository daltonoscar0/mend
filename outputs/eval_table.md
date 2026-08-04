| System | Set | N | Exact match | WER | WER (words only) | Filler removal F1 | Repair resolution | Repairs scored |
|---|---|---|---|---|---|---|---|---|
| Copy baseline | held-out synthetic | 10000 | 0.030 | 0.310 | 0.132 | 0.000 | 0.000 | 2267 |
| Mend (flan-t5-small) | held-out synthetic | 10000 | 0.649 | 0.058 | 0.012 | 0.995 | 0.848 | 2267 |
| Copy baseline | hard set | 100 | 0.080 | 0.689 | 0.601 | 0.000 | 0.000 | 88 |
| Mend (flan-t5-small) | hard set | 100 | 0.550 | 0.206 | 0.171 | 0.667 | 0.557 | 88 |
| Copy baseline | real speech (SwDA) | 2000 | 0.342 | 0.227 | 0.124 | 0.000 | 0.000 | 242 |
| Mend (flan-t5-small) | real speech (SwDA) | 2000 | 0.479 | 0.132 | 0.064 | 0.862 | 0.227 | 242 |

Exact match ignores whitespace differences and treats the sentence-final mark as optional. WER is computed over whitespace tokens with punctuation attached, so a missing comma counts as one error. Filler removal F1 counts occurrences of filler words deleted, so a filler word used literally can be a false positive. Repair resolution asks whether the reparandum is gone and the repair is present; on the synthetic split it uses the injector spans, and on the hard set, which has no spans, it uses the word-multiset difference between input and gold.
