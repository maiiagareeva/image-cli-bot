import os
import torch
from transformers import TrainerCallback

class QFormerCallback(TrainerCallback):
    def __init__(self,qformer,filename="qformer.pt"):
        self.qformer=qformer
        self.filename=filename
    
    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        
        ckpt_dir=os.path.join(args.output_dir,
                              f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir,exist_ok=True)
        path=os.path.join(ckpt_dir,self.filename)
        torch.save(self.qformer.state_dict(),path)
        return control
