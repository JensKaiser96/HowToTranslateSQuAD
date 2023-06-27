## Datasets:

### Squad 1.1/2.0
 * Size:       100k/150k Questions
 * Creation:   manually created, humans generate questions given a Wikipedia article 
 * Special:    has unanswerable questions
 * Task:       passage retrivial
 * SOTA Perf.: 90 EM / 93 F1

### Natural Questions
 * Size:       300k Questions
 * Creation:   from real google queries, get relevant Wikipedia article, select answer in text
 * Special:    Very Natural and close to real world application
 * Task:       passage retrivial for short and long answer
 * SOTA Perf:  64 EM

### HotpotQA
 * Size:       113k Questions
 * Creations:  humans generate questions given a Wikipedia article
 * Special:    answers are not contained within one paragraph, requires reasoning, model also predicts supportig facts
 * Task:       passage retrivial, supporting facts, i.e. sentences which contain the information
 * SOTA Perf.  73 EM
