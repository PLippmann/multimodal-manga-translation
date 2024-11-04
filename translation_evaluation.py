from abc import ABC, abstractmethod
from typing import List, Optional

from sacrebleu.metrics import BLEU, CHRF, TER

# METEOR
from nltk.translate import meteor
import nltk

# COMET
from comet import download_model, load_from_checkpoint

# BERTScore
import logging
import transformers
import matplotlib.pyplot as plt
from matplotlib import rcParams
from bert_score import score as bert_score


ROUNDING_PRECISION = 3

class TranslationEvaluator(ABC):

    @abstractmethod
    def score(self, references: List[str], hypotheses: List[str], lines: Optional[List[str]]) -> float:
        pass 

class BleuEvaluator(TranslationEvaluator):

    def __init__(self):
        self.bleu = BLEU()

    def __str__(self):
        return "BLEU"

    def score(self, references: List[str], hypotheses: List[str], lines: Optional[List[str]] = None) -> float:
        return round(self.bleu.corpus_score(hypotheses=hypotheses, references=[references]).score, ROUNDING_PRECISION)
    
class ChrfEvaluator(TranslationEvaluator):

    def __init__(self, word_order = 0):
        self.chrf = CHRF(word_order=word_order)

    def __str__(self):
        return "chrF2"

    def score(self, references: List[str], hypotheses: List[str], lines: Optional[List[str]]) -> float:
        return round(self.chrf.corpus_score(hypotheses=hypotheses, references=[references]).score, ROUNDING_PRECISION)
    
class TerEvaluator(TranslationEvaluator):

    def __init__(self):
        self.ter = TER()

    def __str__(self):
        return "TER"

    def score(self, references: List[str], hypotheses: List[str], lines: Optional[List[str]]) -> float:
        return round(self.ter.corpus_score(hypotheses=hypotheses, references=[references]).score, ROUNDING_PRECISION)
    
class MeteorEvaluator(TranslationEvaluator):

    def __init__(self):
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('punkt')

    def __str__(self):
        return "METEOR"
    
    def score(self, references: List[str], hypotheses: List[str], lines: Optional[List[str]]) -> float:
        refs_tokenized = [[nltk.word_tokenize(ref)] for ref in references]
        hypotheses_tokenized = [nltk.word_tokenize(hypo) for hypo in hypotheses]

        total_score = 0.0

        for ref, hypo in zip(refs_tokenized, hypotheses_tokenized):
            total_score += round(meteor(ref, hypo), ROUNDING_PRECISION+3)

        meteor_score = total_score / len(refs_tokenized)

        return round(meteor_score, ROUNDING_PRECISION)
    

class CometEvaluator(TranslationEvaluator):

    def __init__(self):
        model_path = download_model("Unbabel/wmt22-comet-da")
        self.model = load_from_checkpoint(model_path)

    def __str__(self):
        return "COMET"

    def score(self, references: List[str], hypotheses: List[str], lines: Optional[List[str]]) -> float:

        data = []

        if lines is not None:
            for ref, hyp, line in zip(references, hypotheses, lines):
                data.append({'src': line, 'mt': hyp, 'ref': ref})
        else:
            for ref, hyp, line in zip(references, hypotheses):
                data.append({'src': '', 'mt': hyp, 'ref': ref})
        try:
            model_output = self.model.predict(data, batch_size=8, accelerator='cuda')
        except:
            model_output = self.model.predict(data, batch_size=8, gpus=0, accelerator='cpu')

        return round(model_output.system_score, ROUNDING_PRECISION)
    
class XCometEvaluator(TranslationEvaluator):

    def __init__(self):
        model_path = download_model("Unbabel/XCOMET-XL")
        self.model = load_from_checkpoint(model_path)

    def __str__(self):
        return "XCOMET-XL"

    def score(self, references: List[str], hypotheses: List[str], lines: Optional[List[str]]) -> float:

        data = []

        if lines is not None:
            for ref, hyp, line in zip(references, hypotheses, lines):
                data.append({'src': line, 'mt': hyp, 'ref': ref})
        else:
            for ref, hyp, line in zip(references, hypotheses):
                data.append({'src': '', 'mt': hyp, 'ref': ref})
        try:
            model_output = self.model.predict(data, batch_size=8, accelerator='cuda')
        except:
            model_output = self.model.predict(data, batch_size=8, gpus=0, accelerator='cpu')

        return round(model_output.system_score, ROUNDING_PRECISION)
    
class XXLCometEvaluator(TranslationEvaluator):

    def __init__(self):
        model_path = download_model("Unbabel/XCOMET-XXL")
        self.model = load_from_checkpoint(model_path)

    def __str__(self):
        return "XCOMET-XXL"

    def score(self, references: List[str], hypotheses: List[str], lines: Optional[List[str]]) -> float:

        data = []

        if lines is not None:
            for ref, hyp, line in zip(references, hypotheses, lines):
                data.append({'src': line, 'mt': hyp, 'ref': ref})
        else:
            for ref, hyp, line in zip(references, hypotheses):
                data.append({'src': '', 'mt': hyp, 'ref': ref})
        try:
            model_output = self.model.predict(data, batch_size=8, accelerator='gpu')
        except:
            print('this is happening')
            model_output = self.model.predict(data, batch_size=8, gpus=0, accelerator='cpu')

        logging.getLogger("Experiment").info(f'XCOMET-XXL SCORES: {model_output.scores}')

        return round(model_output.system_score, ROUNDING_PRECISION)
    
class BertScoreEvaluator(TranslationEvaluator):

    def __init__(self, lang: str = 'en', verbose: bool = True):
        self.lang = lang
        self.verbose = verbose

        # hide the loading messages
        transformers.tokenization_utils.logger.setLevel(logging.ERROR)
        transformers.configuration_utils.logger.setLevel(logging.ERROR)
        transformers.modeling_utils.logger.setLevel(logging.ERROR)

        rcParams["xtick.major.size"] = 0
        rcParams["xtick.minor.size"] = 0
        rcParams["ytick.major.size"] = 0
        rcParams["ytick.minor.size"] = 0

        rcParams["axes.labelsize"] = "large"
        rcParams["axes.axisbelow"] = True
        rcParams["axes.grid"] = True

    def __str__(self):
        return "BERTScore"
    
    def score(self, references: List[str], hypotheses: List[str], lines: Optional[List[str]]) -> float:
        P, R, F1 = bert_score(cands = hypotheses, refs = references, lang=self.lang, verbose=self.verbose)

        return round(F1.mean().item(), ROUNDING_PRECISION)