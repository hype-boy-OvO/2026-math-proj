from .base_bert import BertTiny
from memory import MemoryBank

class ModifiedBert(BertTiny):
    def __init__(self, model_name="google-bert/bert-base-uncased", device=None):
        super().__init__(model_name, device)
        self.memory_bank = MemoryBank()



    def encode(self, text):
        result = super().encode(text)
        if(self.activations.get("ffn") is None):
            return {"text": text, "similar_texts": []}
        
        f = self.memory_bank.fourier_series(self.activations.get("ffn"))
        sim_num = self.memory_bank.fourier_is_close(f)

        sim = []

        if sim_num:
            for idx in sim_num:
                sim.append(self.memory_bank.memory[idx]["text"])
        
        self.memory_bank.add_memory(f, text)
        return {"text": text, "similar_texts": sim}



        