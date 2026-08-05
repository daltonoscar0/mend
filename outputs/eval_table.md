| System | Set | N | Exact match | WER | WER (words only) | Filler removal F1 | Repair resolution | Sentences with a repair |
|---|---|---|---|---|---|---|---|---|
| Copy baseline | held-out synthetic | 10000 | 0.030 [0.027, 0.033] | 0.310 | 0.132 | 0.000 | 0.000 [0.000, 0.000] | 2267 |
| Mend (synthetic only) | held-out synthetic | 10000 | 0.649 [0.639, 0.658] | 0.058 | 0.012 | 0.995 | 0.848 [0.832, 0.861] | 2267 |
| Mend (+ real speech) | held-out synthetic | 10000 | 0.646 [0.637, 0.655] | 0.058 | 0.012 | 0.996 | 0.863 [0.848, 0.877] | 2267 |
| Copy baseline | hard set | 100 | 0.080 [0.030, 0.140] | 0.684 | 0.597 | 0.000 | 0.000 [0.000, 0.000] | 88 |
| Mend (synthetic only) | hard set | 100 | 0.550 [0.450, 0.650] | 0.204 | 0.168 | 0.667 | 0.568 [0.460, 0.670] | 88 |
| Mend (+ real speech) | hard set | 100 | 0.610 [0.520, 0.710] | 0.177 | 0.143 | 0.571 | 0.648 [0.548, 0.747] | 88 |
| Copy baseline | real speech (SwDA) | 4000 | 0.325 [0.310, 0.339] | 0.222 | 0.117 | 0.000 | 0.000 [0.000, 0.000] | 519 |
| Mend (synthetic only) | real speech (SwDA) | 4000 | 0.455 [0.439, 0.470] | 0.125 | 0.056 | 0.859 | 0.326 [0.285, 0.366] | 519 |
| Mend (+ real speech) | real speech (SwDA) | 4000 | 0.616 [0.600, 0.631] | 0.073 | 0.026 | 0.955 | 0.503 [0.460, 0.546] | 519 |

Bracketed ranges are 95% percentile bootstrap intervals, resampling **sentences** with replacement. A sentence is the unit that was independently drawn from the corpus; the several disfluencies inside one sentence are not independent of each other, so resampling repairs would give intervals that are too narrow. Read the hard-set rows with this in mind -- 100 sentences cannot separate two systems a few points apart.

Exact match ignores whitespace differences and treats the sentence-final mark as optional. WER is computed over whitespace tokens with punctuation attached, so a missing comma counts as one error. Filler removal F1 counts occurrences of filler words deleted, so a filler word used literally can be a false positive. Repair resolution asks whether the reparandum is gone and the repair is present; on the synthetic split it uses the injector spans, and on the hard set, which has no spans, it uses the word-multiset difference between input and gold.
