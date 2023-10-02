# Thesis
Repository containing the material for my master thesis

## Timeplan
![Timeplan](data/misc/timeplan.jpg)

## Approach
![Approach](data/misc/approach.jpg)

## Stress Test
- [x] OOD - Out of Domain: German parts of MLQA and XSQuAD
- [ ] NOT - Unanswerable Questions: (50-100) unanswerable questions on the GermanQuAD testset
  - maybe 3 levels of difficulty:
    - 1. Answer to Question is by no means in the text
    - 2. Answer is in the text but the context does not fit
    - 3. Question is not answerable due to small details
- [ ] DIS - Distracting Content: (25-50) short contexts from GermanQuAD testset extended with some distracting context
- [ ] ONE - Question Reduction: (50-100) Questions from GermanQuAD reduced to a single word
- [ ] SDT - SQuAD testset MT: MT version of the SQuAD(2.0) testset

## Preliminary Results

| Name        | EM   | F1   |
|-------------|------|------|
| GQUAD_paper | 68.6 | 88.1 |
| GQUAD_repli | 71.9 | 84.9 |
| raw         | 52.7 | 67.7 |
| MLQA-GER    | 49.8 | 68.0 |
|             |      |      |
|             |      |      |


## notes
MLQA guys do translation with answer span retivial via attention of the translator

Translatin of quoted(") results in no translation. lel, hehe

Quote takes way longer to translate, because each context needs to be translated as often as there are answers for it

QUOTE SUCCESS/FAIL 78.498 / 9.101

GermanQuad talk about using Top-1, when in reality what they mean is, is there any overlap between prediction and gold
truth.
one could just call that %recall>1, ..., for whatever that is worth.
Top-1-accuracy (any overlap of prediction and ground truth)
I get their intention, but its easily cheesable

### Epoch test

![Performance over training](data/plots/epoch_eval_f1_em.png)
`learing_rate=2e-5, weight_decay=0.01, batch_size=4`