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
