# Mend

Mend turns spoken dictation into written text. It removes filled pauses, resolves
self-repairs, and restores casing and punctuation:

```
send it to john no wait sarah um by friday
-> Send it to Sarah by Friday.
```

The repo contains four things: a disfluency injector that builds training data with
Shriberg-structured annotations, a fine-tuned seq2seq cleanup model, an evaluation
table, and a surprisal-at-repair-boundaries figure.

Setup and usage instructions land here as the stages are built.

## A note on `data/hard_set.jsonl`

The hard set is 100 adversarial examples covering nested repairs, repairs that
cross a clause boundary, repairs whose reparandum and repair share words, words
like *like* and *sorry* used literally rather than as fillers, number and date
repairs, and sentences carrying several repairs at once. It was drafted in one
pass and **has not been reviewed by a second pair of eyes**. Read it and edit it
by hand before quoting any hard-set number as a result.
