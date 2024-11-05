from abc import ABC, abstractmethod
from translation import *
from translation_evaluation import *
from typing import List, Tuple
import json
import os
import time
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import logging
import enlighten


class DatasetLoader(ABC):
    @abstractmethod
    def get_page_paths(self, use_validaion_set: bool) -> List[str]:
        pass

    @abstractmethod
    def get_lines_per_page(self, use_validaion_set: bool, lang: str) -> List[List[str]]:
        pass

    @abstractmethod
    def get_page_paths_per_volume(self, use_validaion_set: bool) -> List[List[str]]:
        pass

    @abstractmethod
    def get_lines_per_page_per_volume(self, use_validaion_set: bool, lang: str) -> List[List[List[str]]]:
        pass

class OpenMantraLoader(DatasetLoader):

    lang_mapping = {
        "jp": "text_ja", 
        "en": "text_en", 
        "zh": "text_zh", 
    }

    def __init__(self, dataset_directory_path: str, validation_set_pages: List[Tuple[int, int]], test_set_pages: List[Tuple[int, int]]):
        self.dataset_path = dataset_directory_path

        annotation_file_path = dataset_directory_path + "/annotation.json"
        f = open(annotation_file_path)
        self.annotation_data = json.load(f)

        self.validation_set = validation_set_pages
        self.test_set = test_set_pages
    
    def get_page_paths(self, use_validaion_set: bool):
        if use_validaion_set:
            page_list = self.validation_set
        else:
            page_list = self.test_set

        page_path_list = [(self.dataset_path + '/' + self.annotation_data[manga_index]['pages'][page_index]['image_paths']['ja']) for manga_index, page_index in page_list]

        return page_path_list
    
    # Assumes that pages from the same volume are next to each other on the list
    def get_page_paths_per_volume(self, use_validaion_set: bool) -> List[List[str]]:
        if use_validaion_set:
            page_list = self.validation_set
        else:
            page_list = self.test_set
    
        volume_page_path_list = []

        prev_manga_index = -1

        for manga_index, page_index in page_list:
            path = self.dataset_path + '/' + self.annotation_data[manga_index]['pages'][page_index]['image_paths']['ja']
            
            if manga_index != prev_manga_index:
                volume_page_path_list.append([path])
            else:
                volume_page_path_list[-1].append(path)

            prev_manga_index = manga_index

        return volume_page_path_list


    def get_lines_per_page(self, use_validaion_set: bool, lang: str):
        if lang not in self.lang_mapping:
            raise ValueError
        
        if use_validaion_set:
            page_list = self.validation_set
        else:
            page_list = self.test_set

        lines_per_page = []

        for manga_index, page_index in page_list:
            lines_per_page.append([text_box[self.lang_mapping[lang]] for text_box in self.annotation_data[manga_index]['pages'][page_index]['text']])
        
        return lines_per_page
    
    # Assumes that pages from the same volume are next to each other on the list
    def get_lines_per_page_per_volume(self, use_validaion_set: bool, lang: str) -> List[List[List[str]]]:
        if lang not in self.lang_mapping:
            raise ValueError
        
        if use_validaion_set:
            page_list = self.validation_set
        else:
            page_list = self.test_set

        lines_per_page_per_volume = []

        prev_manga_index = -1

        for manga_index, page_index in page_list:
            lines_on_page = [text_box[self.lang_mapping[lang]] for text_box in self.annotation_data[manga_index]['pages'][page_index]['text']]

            if manga_index != prev_manga_index:
                lines_per_page_per_volume.append([lines_on_page])
            else:
                lines_per_page_per_volume[-1].append(lines_on_page)

            prev_manga_index = manga_index

        return lines_per_page_per_volume

class OpenMantraLoaderModifiedImage(OpenMantraLoader):

    def __init__(self, dataset_directory_path: str, validation_set_pages: List[Tuple[int, int]], test_set_pages: List[Tuple[int, int]]):
        super().__init__(dataset_directory_path, validation_set_pages, test_set_pages)

    def get_page_paths(self, use_validaion_set: bool):
        if use_validaion_set:
            page_list = self.validation_set
        else:
            page_list = self.test_set

        page_path_list = [(self.dataset_path + '/images_numbered/' + self.annotation_data[manga_index]['pages'][page_index]['image_paths']['ja']) for manga_index, page_index in page_list]

        return page_path_list
    
    # Assumes that pages from the same volume are next to each other on the list
    def get_page_paths_per_volume(self, use_validaion_set: bool) -> List[List[str]]:
        if use_validaion_set:
            page_list = self.validation_set
        else:
            page_list = self.test_set
    
        volume_page_path_list = []

        prev_manga_index = -1

        for manga_index, page_index in page_list:
            path = self.dataset_path + '/images_numbered/' + self.annotation_data[manga_index]['pages'][page_index]['image_paths']['ja']
            
            if manga_index != prev_manga_index:
                volume_page_path_list.append([path])
            else:
                volume_page_path_list[-1].append(path)

            prev_manga_index = manga_index

        return volume_page_path_list

class LoveHinaLoader(DatasetLoader):

    lang_mapping = {
        "jp": "text_jp", 
        "pl": "text_pl", 
    }

    def __init__(self, dataset_directory_path: str, annotation_file_path: str, validation_set_pages: List[int], test_set_pages: List[int]):
        self.dataset_path = dataset_directory_path

        f = open(annotation_file_path)
        self.annotation_data = json.load(f)

        self.validation_set = validation_set_pages
        self.test_set = test_set_pages
    
    def get_page_paths(self, use_validaion_set: bool):
        if use_validaion_set:
            page_list = self.validation_set
        else:
            page_list = self.test_set

        page_path_list = [(self.dataset_path + '/' + self.annotation_data[page_index]['path']) for page_index in page_list]

        return page_path_list
    
    # Assumes that pages from the same volume are next to each other on the list
    def get_page_paths_per_volume(self, use_validaion_set: bool) -> List[List[str]]:
        return [self.get_page_paths(use_validaion_set)]


    def get_lines_per_page(self, use_validaion_set: bool, lang: str):
        if lang not in self.lang_mapping:
            raise ValueError
        
        if use_validaion_set:
            page_list = self.validation_set
        else:
            page_list = self.test_set

        lines_per_page = []

        for page_index in page_list:
            lines_per_page.append([text_box[self.lang_mapping[lang]] for text_box in self.annotation_data[page_index]['lines']])
        
        return lines_per_page
    
    # Assumes that pages from the same volume are next to each other on the list
    def get_lines_per_page_per_volume(self, use_validaion_set: bool, lang: str) -> List[List[List[str]]]:
        return [self.get_lines_per_page(use_validaion_set, lang)]

class TestRunner():

    @classmethod
    def __test_page_translation_once(cls, translator: Translator, data_loader: DatasetLoader, evaluators: List[TranslationEvaluator], progres_bar, validation_set: bool = True, concat_lines: bool = False, orig_lang = 'jp', dest_lang = 'en'):
        page_paths = data_loader.get_page_paths(validation_set)

        lines_per_page = data_loader.get_lines_per_page(validation_set, orig_lang)

        references_per_page = data_loader.get_lines_per_page(validation_set, dest_lang)

        start_time = time.time()

        hypotheses_per_page = []
        for page_path in page_paths:
            hypotheses_per_page.append(translator.translate_page(page_path))
            progres_bar.update(1)

        total_time = round(time.time() - start_time, 1)
        logging.getLogger("Experiment").info(f'Translation of all pages took {total_time} seconds. ')

        line_list = []
        refs_list = []
        hypo_list = []

        for lines, refs, hypos in zip(lines_per_page, references_per_page, hypotheses_per_page):
            if concat_lines:
                line_list.append(';'.join(lines))
                refs_list.append(';'.join(refs))
                hypo_list.append(';'.join(hypos))
            else:
                for i, line in enumerate(lines):
                    line_list.append(line)
                    refs_list.append(refs[i])
                    try:
                        hypo_list.append(hypos[i])
                    except:
                        hypo_list.append('')
        
        logging.getLogger("Experiment").info(f'Hypotheses: {hypo_list}')
        logging.getLogger("Experiment").info(f'Lines     : {line_list}')

        scores = []

        start_time = time.time()
        for evaluator in evaluators:
            scores.append(evaluator.score(refs_list, hypo_list, line_list))
    
        total_time = round(time.time() - start_time, 1)
        logging.getLogger("Experiment").info(f'Calculating scores took {total_time} seconds. ')


        return scores
    
    @classmethod
    def test_page_translation(cls, translator: Translator, data_loader: DatasetLoader, evaluators: List[TranslationEvaluator], runs: int, validation_set: bool = True, concat_lines: bool = False, orig_lang = 'jp', dest_lang = 'en'):
        total_scores = [0.0] * len(evaluators)

        current_data_set = data_loader.validation_set if validation_set else data_loader.test_set

        logging.getLogger("Experiment").info(f'TESTING PAGE TRANSLATION WITH PAGES{current_data_set}')
        start_time = time.time()

        manager = enlighten.get_manager()
        total_iterations = runs * len(current_data_set)
        with manager.counter(total=total_iterations, desc="Translating") as pbar:
            pbar.update(0)
            for i in range(runs):
                logging.getLogger("Experiment").info(f'-------------------RUN {i+1}/{runs}-------------------')

                scores = cls.__test_page_translation_once(translator, data_loader, evaluators, pbar, validation_set, concat_lines, orig_lang, dest_lang)

                logging.getLogger("Experiment").info(f'------------------------SCORES IN RUN {i+1}/{runs}-----------------------')
                logging.getLogger("Experiment").info(scores)
                logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

                for i, score in enumerate(scores):
                    total_scores[i] += score

        final_scores = []

        for i, score in enumerate(total_scores):
            final_scores.append((str(evaluators[i]), round(score/runs, 3)))

        total_time = round(time.time() - start_time, 1)
        logging.getLogger("Experiment").info(f'All {runs} runs took {total_time} seconds. ')

        return final_scores
    
    @classmethod
    def __test_volume_translation_once(cls, translator: Translator, data_loader: DatasetLoader, evaluators: List[TranslationEvaluator], progres_bar, validation_set: bool = True, concat_lines: bool = False, orig_lang = 'jp', dest_lang = 'en'):
        page_paths_per_volume = data_loader.get_page_paths_per_volume(validation_set)

        lines_per_page = data_loader.get_lines_per_page(validation_set, orig_lang)

        references_per_page = data_loader.get_lines_per_page(validation_set, dest_lang)

        start_time = time.time()

        hypotheses_per_page = []
        for volume_page_paths in page_paths_per_volume:

            hypotheses_per_page += translator.translate_volume(volume_page_paths)
            progres_bar.update(1)

        total_time = round(time.time() - start_time, 1)
        logging.getLogger("Experiment").info(f'Translation of all pages took {total_time} seconds. ')

        line_list = []
        refs_list = []
        hypo_list = []

        for lines, refs, hypos in zip(lines_per_page, references_per_page, hypotheses_per_page):
            if concat_lines:
                line_list.append(';'.join(lines))
                refs_list.append(';'.join(refs))
                hypo_list.append(';'.join(hypos))
            else:
                for i, line in enumerate(lines):
                    line_list.append(line)
                    refs_list.append(refs[i])
                    try:
                        hypo_list.append(hypos[i])
                    except:
                        hypo_list.append('')
        
        logging.getLogger("Experiment").info(f'Hypotheses: {hypo_list}')
        logging.getLogger("Experiment").info(f'Lines     : {line_list}')

        scores = []

        start_time = time.time()
        for evaluator in evaluators:
            scores.append(evaluator.score(refs_list, hypo_list, line_list))
    
        total_time = round(time.time() - start_time, 1)
        logging.getLogger("Experiment").info(f'Calculating scores took {total_time} seconds. ')


        return scores

    @classmethod
    def test_volume_translation(cls, translator: Translator, data_loader: DatasetLoader, evaluators: List[TranslationEvaluator], runs: int, validation_set: bool = True, concat_lines: bool = False, orig_lang = 'jp', dest_lang = 'en'):
        total_scores = [0.0] * len(evaluators)

        current_data_set = data_loader.validation_set if validation_set else data_loader.test_set

        logging.getLogger("Experiment").info(f'TESTING VOLUME TRANSLATION WITH PAGES{current_data_set}')
        start_time = time.time()

        manager = enlighten.get_manager()
        total_iterations = runs * len(data_loader.get_page_paths_per_volume(validation_set))
        with manager.counter(total=total_iterations, desc="Translating") as pbar:
            pbar.update(0)
            for i in range(runs):
                logging.getLogger("Experiment").info(f'-------------------RUN {i+1}/{runs}-------------------')

                scores = cls.__test_volume_translation_once(translator, data_loader, evaluators, pbar, validation_set, concat_lines, orig_lang, dest_lang)

                logging.getLogger("Experiment").info(f'------------------------SCORES IN RUN {i+1}/{runs}-----------------------')
                logging.getLogger("Experiment").info(scores)
                logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

                for i, score in enumerate(scores):
                    total_scores[i] += score

        final_scores = []

        for i, score in enumerate(total_scores):
            final_scores.append((str(evaluators[i]), round(score/runs, 3)))

        total_time = round(time.time() - start_time, 1)
        logging.getLogger("Experiment").info(f'All {runs} runs took {total_time} seconds. ')

        return final_scores