from transformers import EvalPrediction
import evaluate

rouge=evaluate.load("rouge")
bleu=evaluate.load("sacrebleu")

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
        
        bleu_scr=bleu.compute(
            predictions=pred_text,
            references=[[r] for r in ref_text],
        )
        rouge_scr=rouge.compute(
            predictions=pred_text,
            references=ref_text,
            use_stemmer=True,
        )

        return{
            "rouge1": float(rouge_scr["rouge1"]),
            "rouge2": float(rouge_scr["rouge2"]),
            "rougeL": float(rouge_scr["rougeL"]),
            "bleu": float(bleu_scr["score"]),
        }
    return compute_metrics
