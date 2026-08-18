import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = []
        for i in range(new_chars):
            current_context = context[:,-context_length:]        
            logits = model(current_context)
            preds = logits[-1]
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            generator.set_state(initial_state)
            sample = torch.multinomial(probs, 1, generator=generator)
            torch.cat((context,sample), dim=1)
            result += int_to_char[sample.item()]
        return "".join(result)