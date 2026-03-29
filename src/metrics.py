from transformers import EvalPrediction
import evaluate
import torch
# bart_score.py script Github
from src.bart_score import BARTScorer

rouge=evaluate.load("rouge") # by default calculates the average
bleu=evaluate.load("sacrebleu") # same bc of evaluate default lib

# We use 'facebook/bart-large-cnn' as it's the standard for BARTScore - best choice
# device = 'cuda' if torch.cuda.is_available() else 'cpu' 
# Instead of CUDA, Macs use MPS (Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = 'mps' # Mac GPU
elif torch.cuda.is_available():
    device = 'cuda' # NVIDIA GPU
else:
    device = 'cpu' # fallback for older computers
print(f'Running on: {device}')
scorer = BARTScorer(device=device, checkpoint='vblagoje/bart_lfqa')
# facebook/bart-large-cnn = standard for BartScore - summary and formal
# eugenesiow/bart-paraphrase - paraphrase - do two sents mean the same
# vblagoje/bart_lfqa - whether the "answer" actually addresses the "question" or "symptoms" described

def compute_metrics_from_text(pred_text: list[str], ref_text: list[str]) -> dict:
    bleu_scr=bleu.compute(
        predictions=pred_text,
        references=[[r] for r in ref_text],
    )
    rouge_scr=rouge.compute(
        predictions=pred_text,
        references=ref_text,
        use_stemmer=True,
    )

    # .score() takes (source_list, target_list)
    # bart_scores = scorer.score(pred_text, ref_text, batch_size=4)
    # avg_bart_score = sum(bart_scores) / len(bart_scores)

    # INSTEAD: Use bidirectional BARTScore to reduce bias for short predictions
    # forward: P(ref | pred)
    scores_forward = scorer.score(pred_text, ref_text, batch_size=4)
    # backward: P(pred | ref)
    scores_backward = scorer.score(ref_text, pred_text, batch_size=4)
    bart_scores = [ (f + b) / 2 for f, b in zip(scores_forward, scores_backward)]
    avg_bart_score = sum(bart_scores) / len(bart_scores)

    return{
        "rouge1": float(rouge_scr["rouge1"]),
        "rouge2": float(rouge_scr["rouge2"]),
        "rougeL": float(rouge_scr["rougeL"]),
        "bleu": float(bleu_scr["score"]),
        "bartscore": float(avg_bart_score),
    }

def build_compute_metrics(tokenizer,enable_metrics=True):
    def compute_metrics(eval_predict:EvalPrediction):
    
        predictions=eval_predict.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        label_ids=eval_predict.label_ids
        
        pred_text=tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        ref_text=[]
        for id in label_ids:
            ref_ids=id[id!=-100]
            ref_text.append(
                tokenizer.decode(
                    ref_ids,
                    skip_special_tokens=True,
            ))
        return compute_metrics_from_text(pred_text, ref_text) # changed the logic, so that we can use metrics AFTER the training
    return compute_metrics
