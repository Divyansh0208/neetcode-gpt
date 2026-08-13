from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        all_tokens = []
        for num in numbers:
            num_str = str(num)
            all_tokens.append(self.greedy_tokenize(num_str, vocab))
        return all_tokens             
    
    def greedy_tokenize(self, text: str, vocab: Dict[str, int]):
        w = len(text)
        tokens = []
        w = len(text)
        while w > 0:
            if text[:w] in vocab:
                tokens.append(text[:w]) 
                text = text[w:]
                w = len(text)
            else:
                w -= 1
        return tokens


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return len(self.greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        return round(self.count_tokens(text, vocab) / len(text.split(" ")), 4)
