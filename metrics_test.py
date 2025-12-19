# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# candidate = "the cat sat on the mat".split()
# references = [
#     "the cat is on the mat".split(),
#     "there is a cat on the mat".split()
# ]

# # NLTK uses 1-4 gram
# bleu = sentence_bleu(references, candidate, 
#                       smoothing_function=SmoothingFunction().method1)
# print(f"BLEU (NLTK): {bleu:.4f}")

# from rouge_score import rouge_scorer

# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# candidate = "the cat sat on the mat"
# reference = "the cat is on the mat"

# scores = scorer.score(reference, candidate)
# print(scores)
# # Result:
# '''{'rouge1': Score(precision=0.8333333333333334, recall=0.8333333333333334, 
# fmeasure=0.8333333333333334), 'rouge2': Score(precision=0.6, recall=0.6, 
# fmeasure=0.6), 'rougeL': Score(precision=0.8333333333333334, recall=0.8333333333333334, 
# fmeasure=0.8333333333333334)}'''

from bart_score import BARTScorer

bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')

refs = ["the cat is on the mat"]
cands = ["the cat sat on the mat"]

# 多种模式
scores = bart_scorer.score(cands, refs, batch_size=4)  # cand -> ref
print(f"BARTScore: {scores}")