# LM_EncDec_Base_GNMT

Encoder-decoder translation model ported from legacy TF2 code C8 to PyTorch (cell=lstm, attention=bahdanau, enc_layers=4, dec_layers=4, bi_layers=1, residual_from=2, gnmt=True).

## Status

- Verification level: **Benchmarked (LM-03, one run)**

## Changes from the legacy code

- PyTorch port; beam search reimplemented (TF Addons is no longer maintained)
- Vocabulary built from the training split only (legacy built it from all pairs)
- "what's" expanded to "what is" (legacy wrote "that is")

## LM-03 setting

English -> French, 5-11 words, 12,800 pairs (seed 1234), 80/20 split, embedding 128, hidden 1024, batch 256, 20 epochs, Adam 1e-3. Small data: compare structures, not absolute BLEU.

## Results

- Parameters: 101.2 M
- Test perplexity 77.5, next-word accuracy 0.510
- BLEU-4 greedy (whole test split): 14.2
- BLEU-4 beam width 4 on 500 pairs: 15.4 (greedy on the same pairs 14.6)
- Examples:
  - `this winter has been mild .` -> `cette maison est en deliberation .` (reference `cet hiver a ete doux .`)
  - `they will be here at three .` -> `ils sont l'ouverture a trois heures .` (reference `ils seront ici a trois heures .`)
  - `i once lived in rome .` -> `je suis venu chez moi .` (reference `j'ai vecu un temps a rome .`)

## Run

```bash
pip install torch==2.5.1 einops numpy
# put fra-eng.zip from https://www.manythings.org/anki/ into data/
python run.py lm03 s2s_gnmt
```
