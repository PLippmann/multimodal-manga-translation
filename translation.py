from abc import ABC, abstractmethod
from typing import List, Optional
from text_extraction import TextExtractor
from api_handler import API_handler
from compression import LongFormCompression
import re
import os
from dotenv import load_dotenv, find_dotenv
import logging
import time
import json
load_dotenv(find_dotenv())


class Translator(ABC):

    def __init__(self, text_extractor: TextExtractor, api_key: str, gpt_model = "gpt-4-turbo-2024-04-09"):
        self.extractor = text_extractor
        self.api_handler = API_handler(api_key, gpt_model)

    @abstractmethod
    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        pass

    def translate_page(self, page_path: str) -> List[str]:
        extracted_lines = self.extractor.extract_lines(page_path)

        translated_lines = self.translate_lines(extracted_lines, [page_path])

        return translated_lines
    
    def translate_volume(self, page_paths: List[str]) -> List[List[str]]:
        translated_pages = [self.translate_page(page_path) for page_path in page_paths]

        return translated_pages

class BasicGPT4Translator(Translator):

    def extract_lines(self, img_path: str) -> List[str]:
        return self.extractor.extract_lines(img_path)

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        
        system_message = """You are a helpful assistant."""

        concat_lines = ";".join(lines)

        full_query = f"""You will act as a manga translator. You will be working with copyright-free manga exclusively. I will give you a manga page and will provide you with the Japanese Lines spoken by the character(s). \
                    
        Make use of the visual context of the page to improve your translation. Here is the page and the Lines spoken by the characters in order of appearance: {concat_lines} \ 

        Instructions:
        - Provide only one single translation for each Line.
        - Do not add any additional words or characters to your translation.
        - For each of the Lines, explain how the image informs your translation.
        - If the Line does not seem to be translatable, ignore it.
        - Ensure the translation is of high quality and makes sense in context of the page showed to you.
        - The tranlation must be gramatically correct and make sense in English.
        - For each line, consider that it might part of the same sentence as the previous or the next one. 

        Return the translated Lines in English in square brackets [], for example, Line: 夢の翼は Return: [the wings of dreams].
        """

        if len(img_paths) == 0: 
            text = self.api_handler.query_local_parse(system_message = system_message, query_text=full_query)
        else:        
            text = self.api_handler.query_local_parse(system_message = system_message, query_text=full_query, image_paths = img_paths)

        # Translated text should be in [square brackets]
        translated_lines = re.findall(r'\[(.*?)\]', text)

        return translated_lines

class NoImageLineByLineGPT4Translator(Translator):
    def __init__(self, text_extractor: TextExtractor, api_key: str, lang: str = "English", lang_example: str = "Example, Line: ありがとう Return: [Thank you]"):
        self.lang = lang
        self.lang_example = lang_example
        super().__init__(text_extractor = text_extractor, api_key = api_key)

    def extract_lines(self, img_path: str) -> List[str]:
        return self.extractor.extract_lines(img_path)

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        
        system_message = """You are a helpful assistant."""

        translated_lines = []

        for line in lines:

            full_query = f"""You will act as a japanese manga translator. You will be working with copyright-free manga exclusively. 

            I will give you one line spoken by a character from a manga.
                        
            Here is the line: {line}

            Your task is to translate the line to {self.lang}.

            Return the translated line in {self.lang} in square brackets [].
            {self.lang_example}
              
            """
            found = False
            tries = 10

            while not found and tries > 0:
                tries -= 1

                text = self.api_handler.query_local_parse(system_message = system_message, query_text=full_query)
                lines_received = re.findall(r'\[(.*?)\]', text)

                if len(lines_received) == 1:
                    translated_lines.append(lines_received[0])
                    found = True

        return translated_lines

class ImageLineByLineGPT4Translator(Translator):
    def __init__(self, text_extractor: TextExtractor, api_key: str, lang: str = "English", lang_example: str = "Example 1: Line 1: ありがとうございました Return: Translation 1: [Thank you so much!](As seen on the page, the character is happily thanking somebody)."):
        self.lang = lang
        self.lang_example = lang_example
        super().__init__(text_extractor = text_extractor, api_key = api_key)

    def extract_lines(self, img_path: str) -> List[str]:
        return self.extractor.extract_lines(img_path)

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        
        system_message = """You are a helpful assistant."""

        translated_lines = []

        for line in lines:

            full_query = f"""You will act as a japanese manga translator. You will be working with copyright-free manga exclusively. 

            I will give you one line spoken by a character from a manga.

            I will also give you a manga page this manga comes from.
                        
            Here is the line: {line}

            Your task is to translate the line to {self.lang} and to explain how the image informs your translation.

            Return the translated line in {self.lang} in square brackets and the explanation for how the image informs the translation in parentheses. 

            {self.lang_example}
            """
            found = False
            tries = 10

            while not found and tries > 0:
                tries -= 1

                text = self.api_handler.query_local_parse(system_message = system_message, query_text=full_query, image_paths=img_paths)
                lines_received = re.findall(r'\[(.*?)\]', text)

                if len(lines_received) == 1:
                    translated_lines.append(lines_received[0])
                    found = True

        return translated_lines

class NoImageLineByLineGPT4TranslatorPolish(Translator):

    def extract_lines(self, img_path: str) -> List[str]:
        return self.extractor.extract_lines(img_path)

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        
        system_message = """Jesteś pomocnym asystentem."""

        translated_lines = []

        for line in lines:

            full_query = f"""Będziesz pełnił rolę tłumacza japońskich mang. Będziesz pracował tylko z mangami, które nie są objęte prawami autorskimi. 

            Podam ci jedno zdanie, powiedziane przez postać w mandze.
                        
            To jest to zdanie: {line}

            Twoim zadaniem jest przetłumaczyć je na język polski.

            Zwróć przetłumaczone zdanie po polsku w nawiasach kwadratowych []. Na przykład dla zdania: ありがとう Powinieneś zwrócić: [Dziękuję].
            """
            found = False
            tries = 10

            while not found and tries > 0:
                tries -= 1

                text = self.api_handler.query_local_parse(system_message = system_message, query_text=full_query)
                lines_received = re.findall(r'\[(.*?)\]', text)

                if len(lines_received) == 1:
                    translated_lines.append(lines_received[0])
                    found = True

        return translated_lines

class CustomQueryGPT4Translator(Translator):
    def __init__(self, system_msg: str, query_txt: str, retry_limit: int, text_extractor: TextExtractor, api_key: str, temperature: Optional[float] = None, skip_image: bool = False):
        self.sys_msg = system_msg
        self.query_txt = query_txt
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"CustomQueryGPT4Translator initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key)

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:

        if self.skip_image:
            img_paths = []

        logging.getLogger("Experiment").info("SENDING API REQUEST WITH" + ("OUT" if len(img_paths)==0 else "") + " IMAGE")

        system_message = self.sys_msg

        concat_lines = ";".join(lines)

        full_query = self.query_txt.format(concat_lines)

        logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

        tries = self.retry_limit
        translated_lines = []

        start_time = time.time()
        while(tries > 0 and len(translated_lines) != len(lines)):
            tries -= 1
            text = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths)
            logging.getLogger("Experiment").debug(f"API RESPONSE: \n{text}")

            # Translated text should be in [square brackets]
            translated_lines = re.findall(r'\[(.*?)\]', text)
            if len(translated_lines) != len(lines):
                logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
        total_time = round(time.time() - start_time, 1)

        logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
        logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

        return translated_lines

class CustomQueryLineFormatGPT4Translator(Translator):
    def __init__(self, system_msg: str, query_txt: str, line_passing_format: str, retry_limit: int, text_extractor: TextExtractor, api_key: str, temperature: Optional[float] = None, skip_image: bool = False):
        self.sys_msg = system_msg
        self.query_txt = query_txt
        self.line_format = line_passing_format
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"CustomQueryLineFormatGPT4Translator initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nLine format: {line_passing_format}\nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key, gpt_model='gpt-4-turbo')

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:

        if self.skip_image:
            img_paths = []

        logging.getLogger("Experiment").info("SENDING API REQUEST WITH" + ("OUT" if len(img_paths)==0 else "") + " IMAGE")

        system_message = self.sys_msg

        concat_lines = ""

        for line_no, line in enumerate(lines):
            formatted_line = self.line_format.format((line_no+1), line)
            concat_lines += formatted_line + "\n"

        full_query = self.query_txt.format(concat_lines)

        logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

        tries = self.retry_limit
        translated_lines = []

        start_time = time.time()
        while(tries > 0 and len(translated_lines) != len(lines)):
            tries -= 1
            text = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths)
            logging.getLogger("Experiment").debug(f"API RESPONSE: \n{text}")

            # Translated text should be in [square brackets]
            translated_lines = re.findall(r'\[(.*?)\]', text)
            # if len(translated_lines) == 2 * len(lines):
            #     translated_lines = translated_lines[len(lines):]

            if len(translated_lines) != len(lines):
                logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
        total_time = round(time.time() - start_time, 1)

        logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
        logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

        return translated_lines    

class LongFormContextTranslator(Translator):
    def __init__(self, system_msg: str, query_txt: str, retry_limit: int, text_extractor: TextExtractor, context_handler: LongFormCompression, api_key: str, temperature: Optional[float] = None, skip_image: bool = False):

        # The following might be unnecessary if we don't want to experiment with differnt variants
        self.sys_msg = system_msg
        self.query_txt = query_txt

        # The following will probably stay relevant
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        self.context_handler = context_handler
        self.context_so_far = ""
        self.current_observation = ""

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"LongFormContextTranslator initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key)

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        return lines

    def translate_volume(self, page_paths: List[str]) -> List[List[str]]:

        translated_pages = []
        
        # Iterate over pages in the volume
        for i, page_path in enumerate(page_paths):

            # Lines on the page
            lines = self.extractor.extract_lines(page_path)

            # Images to send to GPT
            img_paths = [page_path] if not self.skip_image else []

            logging.getLogger("Experiment").info("SENDING API REQUEST WITH" + ("OUT" if len(img_paths)==0 else "") + " IMAGE")

            concat_lines = ";".join(lines)

            # Incorporating previous context and lines to translate into the query
            full_query = self.query_txt.format(self.context_so_far, concat_lines)
            logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

            tries = self.retry_limit
            translated_lines = []

            # Repeating the translation attempt until we reach the desired number of lines 
            start_time = time.time()
            while(tries > 0 and len(translated_lines) != len(lines)):
                tries -= 1
                text = self.api_handler.query_local_parse(system_message = self.sys_msg, query_text = full_query, temperature = self.temperature, image_paths = img_paths)
                logging.getLogger("Experiment").debug(f"API RESPONSE: \n{text}")

                # Translated text should be in [square brackets]
                translated_lines = re.findall(r'\[(.*?)\]', text)
                if len(translated_lines) != len(lines):
                    logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
            total_time = round(time.time() - start_time, 1)

            logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
            logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

            self.current_observation = ";".join(translated_lines)
            self.context_so_far = self.context_handler.managed_memory(self.current_observation, self.context_so_far)
            print("CONTEXT SUMMARY: ", self.context_so_far)

            translated_pages.append(translated_lines)

        return translated_pages

class StoryTellingTranslator(Translator):
    def __init__(self, system_msg_story: str, system_msg_translator: str, query_template_story: str, query_template_translator: str, retry_limit: int, text_extractor: TextExtractor, api_key: str, temperature: Optional[float] = None, skip_image: bool = True):

        self.sys_msg_st = system_msg_story
        self.sys_msg_tr = system_msg_translator
        self.query_st = query_template_story
        self.query_tr = query_template_translator
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"StoryTellingTranslator initialized with: \nSystem message (story): \n{system_msg_story} \nQuery_txt (story): \n{query_template_story}\nSystem message (translator): \n{system_msg_translator} \nQuery_txt (translator): \n{query_template_translator} \nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key)

    def __convert_page_to_story(self, lines: List[str], img_paths: List[str]):
        logging.getLogger("Experiment").info("SENDING API REQUEST (STORY)")

        system_message = self.sys_msg_st

        concat_lines = ""

        line_format = """Line {}: "{}";"""

        for line_no, line in enumerate(lines):
            formatted_line = line_format.format((line_no+1), line)
            concat_lines += formatted_line + "\n"

        full_query = self.query_st.format(concat_lines)

        logging.getLogger("Experiment").debug(f"API QUERY (STORY): \n{full_query}")

        tries = self.retry_limit
        included_lines = []
        story_text = ""

        start_time = time.time()
        while(tries > 0 and len(included_lines) != len(lines)):
            tries -= 1
            text = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths)
            logging.getLogger("Experiment").debug(f"API RESPONSE: \n{text}")

            # Line breaks break the regex
            text = text.replace('\n', ' ').replace('\r', '')

            # Story should be in [square brackets]
            matches = re.findall(r'\[(.*?)\]', text)
            if len(matches) != 1:
                continue 

            story_text = matches[0]

            logging.getLogger("Experiment").debug(f"STORY TEXT: {story_text}")

            # Character lines should be in "quotation marks"
            included_lines = re.findall(r'\"(.*?)\"', story_text)

            if len(included_lines) != len(lines):
                logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(included_lines)} lines.") 
        total_time = round(time.time() - start_time, 1)

        logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(included_lines) == len(lines) else 'NOT '}achieved.") 
        logging.getLogger("Experiment").info(f"Story conversion took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

        return story_text

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:

        story_text = self.__convert_page_to_story(lines, img_paths)

        if self.skip_image:
            img_paths = []

        logging.getLogger("Experiment").info("SENDING API REQUEST")

        system_message = self.sys_msg_tr

        full_query = self.query_tr.format(story_text)

        logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

        tries = self.retry_limit
        translated_lines = []

        start_time = time.time()
        while(tries > 0 and len(translated_lines) != len(lines)):
            tries -= 1
            text = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths)
            logging.getLogger("Experiment").debug(f"API RESPONSE: \n{text}")

            # Translated lines should be in "quotation marks"
            translated_lines = re.findall(r'\"(.*?)\"', text)

            if len(translated_lines) != len(lines):
                logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
        total_time = round(time.time() - start_time, 1)

        logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
        logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

        return translated_lines

class RunningContextStoryTellingTranslator(Translator):
    def __init__(self, system_msg_story: str, system_msg_translator: str, query_template_story: str, query_template_translator: str, retry_limit: int, text_extractor: TextExtractor, context_handler: LongFormCompression, api_key: str, temperature: Optional[float] = None, skip_image: bool = True):

        self.sys_msg_st = system_msg_story
        self.sys_msg_tr = system_msg_translator
        self.query_st = query_template_story
        self.query_tr = query_template_translator
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        self.context_handler = context_handler
        self.current_observation = ""
        self.default_context = "This is the first page. There is no previous context."
        self.context_so_far = self.default_context

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"RunningContextStoryTellingTranslator initialized with: \nSystem message (story): \n{system_msg_story} \nQuery_txt (story): \n{query_template_story}\nSystem message (translator): \n{system_msg_translator} \nQuery_txt (translator): \n{query_template_translator} \nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key)

    def __convert_page_to_story(self, lines: List[str], img_paths: List[str], page_index: int):
        logging.getLogger("Experiment").info("SENDING API REQUEST (STORY)")

        system_message = self.sys_msg_st

        concat_lines = ""

        line_format = """Line {}: "{}";"""

        for line_no, line in enumerate(lines):
            formatted_line = line_format.format((line_no+1), line)
            concat_lines += formatted_line + "\n"

        if page_index == 0:
            full_query = PROMPT_LIBRARY[35].format(concat_lines)
        else:
            full_query = self.query_st.format(self.context_so_far, concat_lines)

        logging.getLogger("Experiment").debug(f"API QUERY (STORY): \n{full_query}")

        tries = self.retry_limit
        included_lines = []
        story_text = ""

        start_time = time.time()
        while(tries > 0 and len(included_lines) != len(lines)):
            tries -= 1
            text = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths)
            logging.getLogger("Experiment").debug(f"API RESPONSE: \n{text}")

            # Line breaks break the regex
            text = text.replace('\n', ' ').replace('\r', '')

            # Story should be in [square brackets]
            # matches = re.findall(r'\[(.*?)\]', text)
            # if len(matches) != 1:
            #     continue 

            # story_text = matches[0]
            story_text = text

            logging.getLogger("Experiment").debug(f"STORY TEXT: {story_text}")

            # Character lines should be in "quotation marks"
            included_lines = re.findall(r'\"(.*?)\"', story_text)
            if len(included_lines) == 0:
                included_lines = re.findall(r'\「(.*?)\」', story_text)
            # Character lines should be in [square brackets]
            # included_lines = re.findall(r'\[(.*?)\]', story_text)
            # included_lines = re.findall(r'\「(.*?)\」', story_text)

            if len(included_lines) != len(lines):
                logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(included_lines)} lines.") 
        total_time = round(time.time() - start_time, 1)

        logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(included_lines) == len(lines) else 'NOT '}achieved.") 
        logging.getLogger("Experiment").info(f"Story conversion took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

        return story_text

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        return lines

    def translate_volume(self, page_paths: List[str]) -> List[List[str]]:

        translated_pages = []
        
        # Iterate over pages in the volume
        for i, page_path in enumerate(page_paths):

            # Lines on the page
            lines = self.extractor.extract_lines(page_path)

            story_text = self.__convert_page_to_story(lines, [page_path], i)

            # Images to send to GPT
            img_paths = [page_path] if not self.skip_image else []

            logging.getLogger("Experiment").info("SENDING API REQUEST")

            system_message = self.sys_msg_tr

            full_query = self.query_tr.format(self.context_so_far, story_text)

            logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

            tries = self.retry_limit
            translated_lines = []

            start_time = time.time()
            while(tries > 0 and len(translated_lines) != len(lines)):
                tries -= 1
                text = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths)
                logging.getLogger("Experiment").debug(f"API RESPONSE: \n{text}")

                # Translated lines should be in "quotation marks"
                translated_lines = re.findall(r'\"(.*?)\"', text)
                if len(translated_lines) == 0:
                    translated_lines = re.findall(r'\「(.*?)\」', text)
                # Translated lines should be in [square brackets]
                # translated_lines = re.findall(r'\[(.*?)\]', text)
                # translated_lines = re.findall(r'\「(.*?)\」', text)

                if len(translated_lines) != len(lines):
                    logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
            total_time = round(time.time() - start_time, 1)

            logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
            logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

            self.current_observation = text
            self.context_so_far = self.context_handler.managed_memory(self.current_observation, self.context_so_far)
            logging.getLogger("Experiment").info(f"Context so far: {self.context_so_far}")

            translated_pages.append(translated_lines)

        self.context_so_far = self.default_context
        self.current_observation = ""

        return translated_pages

class ThreePageContextTranslator(Translator):
    def __init__(self, system_msg: str, query_txt: str, line_passing_format: str, retry_limit: int, text_extractor: TextExtractor, api_key: str, temperature: Optional[float] = None, skip_image: bool = False):
        self.sys_msg = system_msg
        self.query_txt = query_txt
        self.line_format = line_passing_format
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"ThreePageContextTranslator initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nLine format: {line_passing_format}\nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key, gpt_model='gpt-4-turbo')

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        return lines

    def translate_volume(self, page_paths: List[str]) -> List[List[str]]:

        translated_pages = []
        
        # Iterate over pages in the volume
        for i, page_path in enumerate(page_paths):

            img_paths = []
            lines_per_page = []
            prev_lines = []
            next_lines = []

            if i > 0:
                prev_page_path = page_paths[i-1]
                img_paths.append(prev_page_path)
                prev_lines = self.extractor.extract_lines(prev_page_path)
                concat_lines = ""
                for line_no, line in enumerate(prev_lines):
                    formatted_line = self.line_format.format((line_no+1), line)
                    concat_lines += formatted_line + "\n"
                lines_per_page.append(concat_lines)

            # Lines on the page
            lines = self.extractor.extract_lines(page_path)
            concat_lines = ""

            for line_no, line in enumerate(lines):
                formatted_line = self.line_format.format((line_no+1), line)
                concat_lines += formatted_line + "\n"

            img_paths.append(page_path)
            lines_per_page.append(concat_lines)

            if i < len(page_paths) - 1:
                next_page_path = page_paths[i+1]
                img_paths.append(next_page_path)
                next_lines = self.extractor.extract_lines(next_page_path)

                concat_lines = ""
                for line_no, line in enumerate(next_lines):
                    formatted_line = self.line_format.format((line_no+1), line)
                    concat_lines += formatted_line + "\n"

                lines_per_page.append(concat_lines)

            all_lines = ""
            for page_no, page_lines in enumerate(lines_per_page):
                all_lines += f"Page {page_no+1}: \n"
                all_lines += page_lines + '\n'

            total_expected_lines = len(prev_lines) + len(lines) + len(next_lines)
            
            system_message = self.sys_msg

            full_query = self.query_txt.format(all_lines)

            logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

            tries = self.retry_limit
            translated_lines = []

            start_time = time.time()
            while(tries > 0 and len(translated_lines) != total_expected_lines):
                tries -= 1
                text = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths)
                logging.getLogger("Experiment").debug(f"API RESPONSE: \n{text}")

                # Translated text should be in [square brackets]
                translated_lines = re.findall(r'\[(.*?)\]', text)

                if len(translated_lines) != total_expected_lines:
                    logging.getLogger("Experiment").debug(f"Expected: {total_expected_lines} lines, received {len(translated_lines)} lines.") 
            total_time = round(time.time() - start_time, 1)

            logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
            logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

            translated_lines = translated_lines[len(prev_lines):len(prev_lines)+len(lines)]
            
            translated_pages.append(translated_lines)

        return translated_pages

class CustomQueryLineFormatGPT4TranslatorJSON(Translator):
    def __init__(self, system_msg: str, query_txt: str, line_passing_format: str, retry_limit: int, text_extractor: TextExtractor, api_key: str, temperature: Optional[float] = None, skip_image: bool = False):
        self.sys_msg = system_msg
        self.query_txt = query_txt
        self.line_format = line_passing_format
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"CustomQueryLineFormatGPT4TranslatorJSON initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nLine format: {line_passing_format}\nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key, gpt_model='gpt-4-turbo')

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:

        if self.skip_image:
            img_paths = []

        logging.getLogger("Experiment").info("SENDING API REQUEST WITH" + ("OUT" if len(img_paths)==0 else "") + " IMAGE")

        system_message = self.sys_msg

        concat_lines = ""

        for line_no, line in enumerate(lines):
            formatted_line = self.line_format.format((line_no+1), line)
            concat_lines += formatted_line + "\n"

        full_query = self.query_txt.format(concat_lines)

        logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

        tries = self.retry_limit
        translated_lines = []

        start_time = time.time()
        while(tries > 0 and len(translated_lines) != len(lines)):
            tries -= 1
            response = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths, json_response = True)
            logging.getLogger("Experiment").debug(f"API RESPONSE: \n{response}")
            
            try:
                pprocess = json.loads(response)

                try:
                    key = pprocess.keys()[-1]
                except:
                    key = 'lines'

                translated_lines = [line_dict['translation'] for line_dict in pprocess[key]]
            except:
                translated_lines = []

            if len(translated_lines) != len(lines):
                logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
        total_time = round(time.time() - start_time, 1)

        logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
        logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

        return translated_lines
    
class SelfRankingGPT4TranslatorJSON(Translator):
    def __init__(self, system_msg: str, query_txt: str, reranking_query: str, line_passing_format: str, retry_limit: int, text_extractor: TextExtractor, api_key: str, temperature: Optional[float] = None, skip_image: bool = False):
        self.sys_msg = system_msg
        self.query_txt = query_txt
        self.ranking_q = reranking_query
        self.line_format = line_passing_format
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"SelfRankingGPT4TranslatorJSON initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nLine format: {line_passing_format}\nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key, gpt_model='gpt-4-turbo')

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:

        if self.skip_image:
            img_paths = []

        logging.getLogger("Experiment").info("SENDING API REQUEST WITH" + ("OUT" if len(img_paths)==0 else "") + " IMAGE")

        system_message = self.sys_msg

        concat_lines = ""

        for line_no, line in enumerate(lines):
            formatted_line = self.line_format.format((line_no+1), line)
            concat_lines += formatted_line + "\n"

        full_query = self.query_txt.format(concat_lines)

        logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

        tries = self.retry_limit
        translated_lines = []

        start_time = time.time()
        while(tries > 0 and len(translated_lines) != len(lines)):
            tries -= 1
            response = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths, json_response = True)
            logging.getLogger("Experiment").debug(f"API RESPONSE: \n{response}")
            
            ranking_response = self.api_handler.query_local_parse(system_message=self.ranking_q, query_text=response, temperature = 0.5, image_paths = img_paths, json_response = True)

            logging.getLogger("Experiment").debug(f"RERANKING RESPONSE: \n{ranking_response}")

            try:
                pprocess = json.loads(ranking_response)

                try:
                    key = pprocess.keys()[-1]
                except:
                    key = 'lines'

                translated_lines = pprocess[key]
            except:
                translated_lines = []

            if len(translated_lines) != len(lines):
                logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
        total_time = round(time.time() - start_time, 1)

        logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
        logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

        return translated_lines
    
class CustomQueryLineFormatGPT4TranslatorJSONExampleImage(Translator):
    def __init__(self, system_msg: str, query_txt: str, line_passing_format: str, retry_limit: int, text_extractor: TextExtractor, api_key: str, example_imgs_paths: List[str] = [], temperature: Optional[float] = None):
        self.sys_msg = system_msg
        self.query_txt = query_txt
        self.line_format = line_passing_format
        self.retry_limit = retry_limit
        self.example_imgs = example_imgs_paths
        self.temperature = temperature

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"CustomQueryLineFormatGPT4TranslatorJSONExampleImage initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nLine format: {line_passing_format}\nRetry limit: {retry_limit} \nTemperature: {temperature} \nExample images: {example_imgs_paths}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key, gpt_model='gpt-4-turbo')

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:

        img_paths = self.example_imgs + img_paths

        logging.getLogger("Experiment").info("SENDING API REQUEST WITH IMAGE" + ("S" if len(img_paths) > 1 else "") + f"{img_paths}")

        system_message = self.sys_msg

        concat_lines = ""

        for line_no, line in enumerate(lines):
            formatted_line = self.line_format.format((line_no+1), line)
            concat_lines += formatted_line + "\n"

        full_query = self.query_txt.format(concat_lines)

        logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

        tries = self.retry_limit
        translated_lines = []

        start_time = time.time()
        while(tries > 0 and len(translated_lines) != len(lines)):
            tries -= 1
            response = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths, json_response = True)
            pprocess = json.loads(response)

            logging.getLogger("Experiment").debug(f"API RESPONSE: \n{response}")

            try:
                key = pprocess.keys()[-1]
            except:
                key = 'lines'

            translated_lines = [line_dict['translation'] for line_dict in pprocess[key]]

            if len(translated_lines) != len(lines):
                logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
        total_time = round(time.time() - start_time, 1)

        logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
        logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

        return translated_lines
    
class LongFormContextTranslatorJSON(Translator):
    def __init__(self, system_msg: str, query_txt: str, line_passing_format: str, retry_limit: int, text_extractor: TextExtractor, context_handler: LongFormCompression, api_key: str, temperature: Optional[float] = None, skip_image: bool = False, lang: str = 'en'):

        # The following might be unnecessary if we don't want to experiment with differnt variants
        self.sys_msg = system_msg
        self.query_txt = query_txt
        self.line_format = line_passing_format

        # The following will probably stay relevant
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        self.context_handler = context_handler
        self.context_so_far = ""
        self.current_observation = ""

        self.lang = lang

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"LongFormContextTranslatorJSON initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key)

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        return lines

    def translate_volume(self, page_paths: List[str]) -> List[List[str]]:

        translated_pages = []
        
        # Iterate over pages in the volume
        for i, page_path in enumerate(page_paths):

            # Lines on the page
            lines = self.extractor.extract_lines(page_path)

            # Images to send to GPT
            img_paths = [page_path] if not self.skip_image else []

            logging.getLogger("Experiment").info("SENDING API REQUEST WITH" + ("OUT" if len(img_paths)==0 else "") + " IMAGE")

            concat_lines = ""

            for line_no, line in enumerate(lines):
                formatted_line = self.line_format.format((line_no+1), line)
                concat_lines += formatted_line + "\n"

            # Incorporating previous context and lines to translate into the query
            if i==0:
                if self.lang == 'en':
                    full_query = PROMPT_LIBRARY[44].format(concat_lines)
                elif self.lang == 'pl':
                    full_query = PROMPT_LIBRARY_PL[44].format(concat_lines)
                else:
                    raise ValueError()
            else:
                full_query = self.query_txt.format(self.context_so_far, concat_lines)
            logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

            tries = self.retry_limit
            translated_lines = []

            # Repeating the translation attempt until we reach the desired number of lines 
            start_time = time.time()
            while(tries > 0 and len(translated_lines) != len(lines)):
                tries -= 1
                response = self.api_handler.query_local_parse(system_message = self.sys_msg, query_text = full_query, temperature = self.temperature, image_paths = img_paths, json_response=True)
                
                try:
                    logging.getLogger("Experiment").debug(f"API RESPONSE: \n{response}")
                    
                    pprocess = json.loads(response)

                    try:
                        key = pprocess.keys()[-1]
                    except:
                        key = 'lines'

                    translated_lines = [line_dict['translation'] for line_dict in pprocess[key]]

                except:
                    translated_lines = []

                if len(translated_lines) != len(lines):
                    logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
            total_time = round(time.time() - start_time, 1)

            logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
            logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

            if len(translated_lines) > 0:
                story_key = f"story_{self.lang}"
                self.current_observation = pprocess[story_key]
                self.context_so_far = self.context_handler.managed_memory(self.current_observation, self.context_so_far)
            # print("CONTEXT SUMMARY: ", self.context_so_far)

            translated_pages.append(translated_lines)

        return translated_pages
    
class ThreePageContextTranslatorJSON(Translator):
    def __init__(self, system_msg: str, query_txt: str, line_passing_format: str, retry_limit: int, text_extractor: TextExtractor, api_key: str, temperature: Optional[float] = None, skip_image: bool = False):
        self.sys_msg = system_msg
        self.query_txt = query_txt
        self.line_format = line_passing_format
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"ThreePageContextTranslatorJSON initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nLine format: {line_passing_format}\nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key, gpt_model='gpt-4-turbo')

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        return lines

    def translate_volume(self, page_paths: List[str]) -> List[List[str]]:

        translated_pages = []
        
        # Iterate over pages in the volume
        for i, page_path in enumerate(page_paths):

            img_paths = []
            lines_per_page = []
            prev_lines = []
            next_lines = []

            page_in_focus = 0

            if i > 0:
                prev_page_path = page_paths[i-1]
                img_paths.append(prev_page_path)
                prev_lines = self.extractor.extract_lines(prev_page_path)
                concat_lines = ""
                for line_no, line in enumerate(prev_lines):
                    formatted_line = self.line_format.format((line_no+1), line)
                    concat_lines += formatted_line + "\n"
                lines_per_page.append(concat_lines)
                page_in_focus = 1

            # Lines on the page
            lines = self.extractor.extract_lines(page_path)
            concat_lines = ""

            for line_no, line in enumerate(lines):
                formatted_line = self.line_format.format((line_no+1), line)
                concat_lines += formatted_line + "\n"

            img_paths.append(page_path)
            lines_per_page.append(concat_lines)


            if i < len(page_paths) - 1:
                next_page_path = page_paths[i+1]
                img_paths.append(next_page_path)
                next_lines = self.extractor.extract_lines(next_page_path)

                concat_lines = ""
                for line_no, line in enumerate(next_lines):
                    formatted_line = self.line_format.format((line_no+1), line)
                    concat_lines += formatted_line + "\n"

                lines_per_page.append(concat_lines)

            all_lines = ""
            for page_no, page_lines in enumerate(lines_per_page):
                all_lines += f"Page {page_no+1}: \n"
                all_lines += page_lines + '\n'
            
            system_message = self.sys_msg

            full_query = self.query_txt.format(all_lines)

            logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

            tries = self.retry_limit
            translated_lines = []

            start_time = time.time()
            while(tries > 0 and len(translated_lines) != len(lines)):
                tries -= 1
                response = self.api_handler.query_local_parse(system_message = system_message, query_text = full_query, temperature = self.temperature, image_paths = img_paths, json_response=True)
                pprocess = json.loads(response)

                logging.getLogger("Experiment").debug(f"API RESPONSE: \n{response}")

                try:
                    key = pprocess.keys()[-1]
                except:
                    key = 'pages'

                try:
                    line_dict_list = pprocess[key][page_in_focus]
                except:
                    line_dict_list = pprocess[key][-1]

                translated_lines = [line_dict['translation'] for line_dict in line_dict_list]

                if len(translated_lines) != len(lines):
                    logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
            total_time = round(time.time() - start_time, 1)

            logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
            logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")
            
            translated_pages.append(translated_lines)

        return translated_pages

class WholeVolumeTranslatorJSON(Translator):
    def __init__(self, system_msg: str, query_txt: str, line_passing_format: str, retry_limit: int, text_extractor: TextExtractor, api_key: str, temperature: Optional[float] = None, skip_image: bool = False):
        self.sys_msg = system_msg
        self.query_txt = query_txt
        self.line_format = line_passing_format
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"WholeVolumeTranslatorJSON initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nLine format: {line_passing_format}\nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key, gpt_model='gpt-4-turbo')

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        return lines

    def translate_volume(self, page_paths: List[str]) -> List[List[str]]:

        translated_pages = []
        
        # Iterate over pages in the volume
        lines_per_page = []
        img_paths = []

        for i, page_path in enumerate(page_paths):

            # Lines on the page
            lines = self.extractor.extract_lines(page_path)
            concat_lines = ""

            for line_no, line in enumerate(lines):
                formatted_line = self.line_format.format((line_no+1), line)
                concat_lines += formatted_line + "\n"

            img_paths.append(page_path)
            lines_per_page.append(concat_lines)

        all_lines = ""
        for page_no, page_lines in enumerate(lines_per_page):
            all_lines += f"Page {page_no+1}: \n"
            all_lines += page_lines + '\n'

        full_query = self.query_txt.format(all_lines)

        logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

        tries = self.retry_limit
        translated_lines = []

        start_time = time.time()
        while(tries > 0 and len(translated_pages) != len(page_paths)):
            tries -= 1
            response = self.api_handler.query_local_parse(system_message = self.query_txt, query_text = all_lines, temperature = self.temperature, image_paths = img_paths, json_response=True)
            logging.getLogger("Experiment").debug(f"API RESPONSE: \n{response}")

            try:
                pprocess = json.loads(response)
            except:
                continue

            try:
                key = pprocess.keys()[-1]
            except:
                key = 'pages'

            try:
                line_dict_list_per_page = pprocess[key]
            except:
                line_dict_list_per_page = []

            translated_pages = []

            for line_list in line_dict_list_per_page:
                # translated_lines = [line_dict['translation'] for line_dict in line_dict_list]
                translated_pages.append(line_list)


            if len(translated_pages) != len(page_paths):
                logging.getLogger("Experiment").debug(f"Expected: {len(translated_pages)} lines, received {len(page_paths)} lines.") 
        total_time = round(time.time() - start_time, 1)

        logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
        logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

        return translated_pages
    
class VolumeByPageAllContextTranslator(Translator):
    def __init__(self, system_msg: str, query_txt: str, line_passing_format: str, retry_limit: int, text_extractor: TextExtractor, api_key: str, temperature: Optional[float] = None, skip_image: bool = False):
        self.sys_msg = system_msg
        self.query_txt = query_txt
        self.line_format = line_passing_format
        self.retry_limit = retry_limit
        self.temperature = temperature
        self.skip_image = skip_image

        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')
        logging.getLogger("Experiment").info(f"VolumeByPageAllContextTranslator initialized with: \nSystem message: \n{system_msg} \nQuery_txt: \n{query_txt} \nLine format: {line_passing_format}\nRetry limit: {retry_limit} \nTemperature: {temperature} \nSkip image: {skip_image}")
        logging.getLogger("Experiment").info(f'----------------------------------------------------------------')

        super().__init__(text_extractor = text_extractor, api_key = api_key, gpt_model='gpt-4-turbo')

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        return lines

    def translate_volume(self, page_paths: List[str]) -> List[List[str]]:

        translated_pages = []
        
        # Iterate over pages in the volume
        lines_per_page = []
        img_paths = []

        for i, page_path in enumerate(page_paths):

            # Lines on the page
            lines = self.extractor.extract_lines(page_path)
            concat_lines = ""

            for line_no, line in enumerate(lines):
                formatted_line = self.line_format.format((line_no+1), line)
                concat_lines += formatted_line + "\n"

            img_paths.append(page_path)
            lines_per_page.append(concat_lines)

        all_lines = ""
        for page_no, page_lines in enumerate(lines_per_page):
            all_lines += f"Page {page_no+1}: \n"
            all_lines += page_lines + '\n'

        translated_pages = []

        all_translations_so_far = ""
        
        # Iterate over pages in the volume
        for page_no, page_path in enumerate(page_paths):

            lines = self.extractor.extract_lines(page_path)
            # Incorporating translations of previous lines into the query

            full_query = self.query_txt.format(all_lines, page_no, all_translations_so_far, page_no+1)
            # logging.getLogger("Experiment").debug(f"API QUERY: \n{full_query}")

            tries = self.retry_limit
            translated_lines = []

            # Repeating the translation attempt until we reach the desired number of lines 
            start_time = time.time()
            while(tries > 0 and len(translated_lines) != len(lines)):
                tries -= 1
                response = self.api_handler.query_local_parse(system_message = self.sys_msg, query_text = full_query, temperature = self.temperature, image_paths = img_paths, json_response=True)
                logging.getLogger("Experiment").debug(f"API RESPONSE: \n{response}")

                try:
                    pprocess = json.loads(response)
                except:
                    continue
                
                try:
                    key = pprocess.keys()[-1]
                except:
                    key = 'lines'

                try:
                    translated_lines = [line_dict['translation'] for line_dict in pprocess[key]]
                except: 
                    translated_lines = []

                if len(translated_lines) != len(lines):
                    logging.getLogger("Experiment").debug(f"Expected: {len(lines)} lines, received {len(translated_lines)} lines.") 
            total_time = round(time.time() - start_time, 1)

            all_translations_so_far += f"Page {page_no+1}: \n"
            for i, translation in enumerate(translated_lines):
                all_translations_so_far += f"Translation {i+1}: {translation} \n" 

            logging.getLogger("Experiment").info(f"Took {self.retry_limit-tries} tries. Desired number of lines {'' if len(translated_lines) == len(lines) else 'NOT '}achieved.") 
            logging.getLogger("Experiment").info(f"Translation took {total_time} seconds, which is {total_time/max(1, self.retry_limit-tries):.1f} seconds per try.")

            translated_pages.append(translated_lines)

        return translated_pages
class MantraResultsLoadingTranslator(Translator):

    def __init__(self, text_extractor: TextExtractor, api_key: str, data_path: str):
        f = open(data_path)
        self.data = json.load(f)
        self.data_rev_index = {}
        for page in self.data:
            lines = [] 
            for line in page['texts2']:
                lines += line.split('<SEP>')
            self.data_rev_index[page['img_org']] = lines
        super().__init__(text_extractor = text_extractor, api_key = api_key)

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        return lines

    def translate_page(self, page_path: str) -> List[str]:
        path_steps = page_path.split('/')

        path_steps_2 = path_steps[-1].split('.')

        number = int(path_steps_2[-2]) + 1

        number = '0'+ ('0' if number < 10 else '') + str(number)

        modif_name = 'tmp_org/' + path_steps[-3] + '_' + number + '.jpg'

        translated_lines = self.data_rev_index[modif_name]

        return translated_lines
    
class SceneAwareNMTResultsLoadingTranslator(Translator):

    def __init__(self, text_extractor: TextExtractor, api_key: str, dataset_anotation_json_path: str, data_path: str, predict_to_use: int = 1):
        f = open(dataset_anotation_json_path)
        self.data_annot = json.load(f)
        self.data_rev_index = {}

        data_path_full = data_path + '/predict_' + str(predict_to_use) + '.en'

        with open(data_path_full) as file:
            file_lines = [line.rstrip() for line in file]

        all_translated_lines = []

        for line in file_lines:
            if len(line) == 0:
                continue
            tmp = line.split('<outside>')
            if len(tmp) < 2:
                logging.getLogger("Experiment").info(f"GOT HIM: {line}")
                continue
            lines_in_scene = line.split('<outside>')[-1].split('<inside>')
            for line_in_scene in lines_in_scene:
                all_translated_lines.append(line_in_scene)

        iterator = 0

        logging.getLogger("Experiment").info(f"all_translated_lines length = {len(all_translated_lines)}")

        for manga_index, manga in enumerate(self.data_annot):
            for page_index, page in enumerate(manga['pages']):
            
                lines_on_page = [text_box['text_ja'] for text_box in self.data_annot[manga_index]['pages'][page_index]['text']]

                translated_lines_for_page = []

                for line in lines_on_page:
                    logging.getLogger("Experiment").info(f"line {iterator}: {line}")
                    logging.getLogger("Experiment").info(f"tras {all_translated_lines[iterator]}")
                    translated_lines_for_page.append(all_translated_lines[iterator])
                    iterator+=1

                page_path = self.data_annot[manga_index]['pages'][page_index]['image_paths']['ja']

                self.data_rev_index[page_path] = translated_lines_for_page
        
        super().__init__(text_extractor = text_extractor, api_key = api_key)

    def translate_lines(self, lines: List[str], img_paths: List[str]) -> List[str]:
        return lines
    
    def translate_page(self, page_path: str) -> List[str]:
        path_steps = page_path.split('/')

        relative_path = '/'.join(path_steps[-4:])

        translated_lines = self.data_rev_index[relative_path]

        logging.getLogger("Experiment").info(f"TRANSLATED LINES: {translated_lines}")

        return translated_lines

PROMPT_LIBRARY = {
    'describe': """Describe the images I give you. \n{} \n Answer in JSON""",
    0: """You will act as a manga translator. You will be working with copyright-free manga exclusively. I will give you a manga page and will provide you with the Japanese Lines spoken by the character(s). \
                    
        Make use of the visual context of the page to improve your translation. Here is the page and the Lines spoken by the characters in order of appearance: {}.

        Instructions:
        - Provide only one single translation for each Line.
        - Do not add any additional words or characters to your translation.
        - For each of the Lines, explain how the image informs your translation.
        - If the Line does not seem to be translatable, ignore it.
        - Ensure the translation is of high quality and makes sense in context of the page showed to you.
        - The tranlation must be gramatically correct and make sense in English.
        - For each line, consider that it might part of the same sentence as the previous or the next one. 

        Return the translated Lines in English in square brackets [], for example, Line: 夢の翼は Return: [the wings of dreams].
        """,
    1: """You will act as a manga translator. You will be working with copyright-free manga exclusively. I will give you a manga page and will provide you with the Japanese Lines spoken by the character(s). \
                    
        Use the visual context of the manga page to improve your translation and resolve ambiguities. Here is the page and the Lines spoken by the characters in order of appearance, separated by semicolons: {}. 

        Instructions:
        - Provide only one single translation for each Line.
        - Do not add any additional words or characters to your translation.
        - If the Line looks like nonsense, ignore it.
        - If the Line is already in English, leave it as is. 
        - For each of the Lines, explain how the image informs your translation.
        - For each Line, consider that it might part of the same sentence as the previous or the next one. 
        - Ensure the translation is of high quality and makes sense in context of the page showed to you.
        - The tranlation must be gramatically correct and make sense in English.
        
   

        Return the translated Lines in English in square brackets [], for example, Line: ありがとうございました Return: [Thank you so much!].
        """, # Adding more instructions didn't necessarily provide improvement
    2: """You will act as a manga translator. You will be working with copyright-free manga exclusively. I will give you a manga page and will provide you with the Japanese Lines spoken by the character(s). \
                    
        Use the visual context of the manga page to improve your translation and resolve ambiguities. Here is the page and the Lines spoken by the characters in order of appearance, separated by semicolons: {}. 

        Instructions:
        - Provide only one single translation for each Line.
        - Do not add any additional words or characters to your translation.
        - If the Line looks like nonsense, ignore it.
        - If the Line is already in English, leave it as is. 
        - For each of the Lines, explain how the image informs your translation.
        - For each Line, ensure that it matches the speaker's gender, age, the pictured social situation and the relation between the characters, based on the visual context and the conversation.
        - When you encounter names, transcribe them using the revised Hepburn romanization and keep the honorifics, for example, translate ツバメさん as Tsubame-san and レーネ as Lene.
        - For each Line, consider that it might part of the same sentence as the previous or the next one. 
        - Ensure the translation is of high quality and makes sense in context of the page showed to you.
        - The tranlation must be gramatically correct and make sense in English.

        Return the translated Lines in English in square brackets [], for example, Line: ありがとうございました Return: [Thank you so much!].
        """,
    3: """You will act as a manga translator. You will be working with copyright-free manga exclusively. I will give you a manga page and will provide you with the Japanese Lines spoken by the character(s). \
                    
        Use the visual context of the manga page to improve your translation and resolve ambiguities. Here are the Lines spoken by the characters in order of appearance, separated by semicolons: {}. The image is attached to the message. \

        Instructions:
        - Provide only one single translation for each Line.
        - Do not add any additional words or characters to your translation.
        - If the Line looks like nonsense, ignore it.
        - If the Line is already in English, leave it as is. 
        - For each of the Lines, explain how the image informs your translation.
        - For each Line, ensure that it matches the speaker's gender, age, the pictured social situation and the relation between the characters, based on the visual context and the conversation.
        - When you encounter names, transcribe them using the revised Hepburn romanization and keep the honorifics, for example, translate ツバメさん as Tsubame-san and レーネ as Lene.
        - For each Line, consider that it might part of the same sentence as the previous or the next one. 
        - Ensure the translation is of high quality and makes sense in context of the page showed to you.
        - The tranlation must be gramatically correct and make sense in English.

        Return the translated Lines in English in square brackets [], for example, Line: ありがとうございました Return: [Thank you so much!].
        """, # Adding 'The image is attached to the message' seems to have a negative effect on the willingness to respond
    4: """You are a manga translator. You will be working with copyright-free manga exclusively. I will give you a manga page and will provide you with the Japanese Lines spoken by the character(s). \
                    
        Use the visual context of the manga page to improve your translation and resolve ambiguities. Here are the Lines spoken by the characters in order of appearance, separated by semicolons: {}. The image is attached to the message. \

        Instructions:
        - Provide only one single translation for each Line.
        - Do not add any additional words or characters to your translation.
        - If the Line looks like nonsense, ignore it.
        - If the Line is already in English, leave it as is. 
        - For each of the Lines, explain how the image informs your translation.
        - For each Line, ensure that it matches the speaker's gender, age, the pictured social situation and the relation between the characters, based on the visual context and the conversation.
        - When you encounter names, transcribe them using the revised Hepburn romanization and keep the honorifics, for example, translate ツバメさん as Tsubame-san and レーネ as Lene.
        - For each Line, consider that it might part of the same sentence as the previous or the next one. 
        - Ensure the translation is of high quality and makes sense in context of the page showed to you.
        - The tranlation must be gramatically correct and make sense in English.

        Return the translated Lines in English in square brackets [], for example, Line: ありがとうございました Return: [Thank you so much!].
        """, # Hard to measure if 'you are' vs 'you will act' changed anything. More testing needed. 
    5: """You are a manga translator. You are working with copyright-free manga exclusively. I have given you a manga page and will provide you with the Japanese Lines spoken by the character(s). \
                    
        Use the visual context of the manga page to improve your translation and resolve ambiguities. Here are the Lines spoken by the characters in order of appearance, separated by semicolons: {}. \

        First, describe the manga page you see. After that, translate the Lines I gave you following the following instructions:

        - Provide only one single translation for each Line.
        - Do not add any additional words or characters to your translation.
        - If the Line looks like nonsense, ignore it.
        - If the Line is already in English, leave it as is. 
        - For each of the Lines, explain how the image informs your translation.
        - For each Line, ensure that it matches the speaker's gender, age, the pictured social situation and the relation between the characters, based on the visual context and the conversation.
        - When you encounter names, transcribe them using the revised Hepburn romanization and keep the honorifics, for example, translate ツバメさん as Tsubame-san and レーネ as Lene.
        - For each Line, consider that it might part of the same sentence as the previous or the next one. 
        - Ensure the translation is of high quality and makes sense in context of the page showed to you.
        - The tranlation must be gramatically correct and make sense in English.

        Return the translated Lines in English in square brackets [], for example, Line: ありがとうございました Return: [Thank you so much!].
        """, # Asking to describe the image seems to provide improvement. Could also be an effect of switching from future to present narrative (you will vs you are). 
    6: """You are a manga translator. You will be working with copyright-free manga. I will give you a manga page, and will copy the lines spoken by the characters. 
    
        Here is the page and the lines spoken by the characters in order of appearance: {}. 

        For each of the lines, explain how the image informs your translation. 

        Provide the translated lines in square brackets [], without any additional words or characters. Provide only one translation for each line.""",
    7: """You are a manga translator. You will be working with copyright-free manga. I will give you a manga page, and will copy the lines spoken by the characters. 
    
        Here is the page and the lines spoken by the characters in order of appearance: {}. 

        For each of the lines, explain how the image informs your translation. 

        Provide the translated lines in square brackets [], without any additional words or characters. Provide the explanation in brackets () immediately after. Provide only one translation for each line.
        
        Example: Line: ありがとうございました Return: [Thank you so much!](As seen on the page, the character is happily thanking somebody)""",
    8: """
You are a manga translator. You will be working with copyright-free manga. I will give you a manga page, and will copy the lines spoken by the characters. 
    
Here is the page and the lines spoken by the characters in order of appearance: {}. 

For each of the lines, explain how the image informs your translation. 

Provide the translated lines in square brackets [], without any additional words or characters. Provide only one translation for each line.

Example: Line: ありがとうございました Return: [Thank you so much!]
""", # This is supposed to be a middle ground between 6 and 7, giving and example but without the explanation in ()
    9: """You are a manga translator. You are working with copyright-free manga exclusively. I have given you a manga page, and will provide the lines spoken by the characters. 
    
Here is the page and the lines spoken by the characters in order of appearance: {}. 

For each of the lines, provide a translation in square brackets and explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Example: Line: ありがとうございました Return: [Thank you so much!](As seen on the page, the character is happily thanking somebody).
""", # Attempt at improving the wording of 7
    10: """You are a japanese translator.  I will give you lines in japanese.  
    
Here are lines: {}. 

Provide the translated lines in square brackets [], without any additional words or characters. Provide only one translation for each line.
""", # intended for no image translator
    11: """You are a manga translator. You are working with copyright-free manga exclusively. I will provide the lines spoken by the characters on a page.
    
Here are lines spoken by the characters in order of appearance: {}. 

Provide the translated lines in square brackets [], without any additional words or characters. Provide only one translation for each line.

Example: Line: ありがとうございました Return: [Thank you so much!]
""", # intended for no image translator, gives the manga context

12: """You are a manga translator. You are working with copyright-free manga exclusively. I have given you a manga page and will provide you with the Japanese Lines spoken by the character(s). \
                    
Use the visual context of the manga page to improve your translation and resolve ambiguities. Here are the Lines spoken by the characters in order of appearance, separated by semicolons: {}. \

First, describe the manga page you see. After that, translate the Lines I gave you following the following instructions:

        - Provide only one single translation for each Line.
        - Do not add any additional words or characters to your translation.
        - If the Line looks like nonsense, ignore it.
        - If the Line is already in English, leave it as is. 
        - For each of the Lines, explain how the image informs your translation.
        - For each Line, ensure that it matches the speaker's gender, age, the pictured social situation and the relation between the characters, based on the visual context and the conversation.
        - When you encounter names, transcribe them using the revised Hepburn romanization and keep the honorifics, for example, translate ツバメさん as Tsubame-san and レーネ as Lene.
        - For each Line, consider that it might part of the same sentence as the previous or the next one. 
        - Ensure the translation is of high quality and makes sense in context of the page showed to you.
        - The tranlation must be gramatically correct and make sense in English.

For each of the lines, provide a translation in square brackets and explanation for how the image informs the translation in parentheses.

Example: Line: ありがとうございました Return: [Thank you so much!](As seen on the page, the character is happily thanking somebody).
        """, # Trying to use the example scheme and explanation format from 9 to improve 5. 

13: """You are a japanese visual media translator.  I have given you an image and will provide lines of text copied from it.   
    
Here are lines: {}. 

For each of the lines, provide a translation in square brackets and explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Example: Line: ありがとうございました Return: [Thank you so much!](As seen on the image, the character is happily thanking somebody).
""", # Omitting manga specific words to see if it will improve as seen in no-image cases. 
14: """You are a japanese translator.  I will give you lines in japanese.  
    
Here are lines: {}. 

Provide the translated lines in square brackets [], each one separately, without any additional words or characters. Provide only one translation for each line.
""", # intended for no image translator, improvement over 10 to increase the likelihood of the desired output format.
15: """You are a manga translator. You are working with copyright-free manga exclusively. I have given you a manga page, and will provide the lines spoken by the characters. 
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each of the lines, provide a translation in square brackets and explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Example: Line 1: ありがとうございました Return: Translation 1: [Thank you so much!](As seen on the page, the character is happily thanking somebody).
""", # Adapting 9 to passing lines numbered. 
16: """You are a manga translator. You are working with copyright-free manga exclusively. I have given you a manga page and will provide you with the Japanese Lines spoken by the character(s). \
                    
Use the visual context of the manga page to improve your translation and resolve ambiguities. Here are the Lines spoken by the characters numbered in order of appearance: 

{}

First, describe the manga page you see. After that, translate the Lines I gave you following the following instructions:

    - Provide only one single translation for each Line.
    - Do not add any additional words or characters to your translation.
    - If the Line looks like nonsense, ignore it.
    - If the Line is already in English, leave it as is. 
    - For each of the Lines, explain how the image informs your translation.
    - For each Line, ensure that it matches the speaker's gender, age, the pictured social situation and the relation between the characters, based on the visual context and the conversation.
    - When you encounter names, transcribe them using the revised Hepburn romanization and keep the honorifics, for example, translate ツバメさん as Tsubame-san and レーネ as Lene.
    - For each Line, consider that it might part of the same sentence as the previous or the next one. 
    - Ensure the translation is of high quality and makes sense in context of the page showed to you.
    - The tranlation must be gramatically correct and make sense in English.

Return the translated Lines in English in square brackets [], for example, Line 1: ありがとうございました Return: Line 1 - [Thank you so much!].
""", # Adapting 5 to passing lines numbered.
17: """You are a manga translator. You are working with copyright-free manga exclusively. I have given you a manga page, and will give you a translation sheet form to fill out. 
    
Here is the page and the translation form to fill out: 
{}. 

Return only the filled out form. 
""", # Adapting 9 to giving a form to fill out. 
18: """You are a manga storyteller. You are working with copyright-free manga exclusively. 

I have given you a manga page. On the page, the speech bubbles are numbered. I will give you the text spoken by the characters in those speech bubbles. 

Here are the lines spoken by the characters in the speech bubble with corresponding number: 
{}. 
    
Your task is to tell the story happening on the manga page, incorporating the character lines I give you. Be sure to put the lines exactly as they are, in quotation marks. 

Return the story in square brackets [].
""", # Adapting 9 to telling the story of the manga page. 
19: """You are a story translator. You are working with stories based on copyright-free manga. 

I will give you a story and your task is to translate the story to English. If part of the story is already in English, keep it as it is, and translate the rest.

Here is the story:
{}
    
Make sure that you keep the character lines in quotation marks and repeat each one only once. 
""", # Adapting 9 to translating the story after 18. 
20: """あなたは漫画の語り手. 著作権がない漫画を上げます。

漫画の一ペエジを上げました。そのペエジで吹き出しは番号がついています。その吹き出しの中で登場人物が言った言葉を上げます。

こちらは登場人物が言った言葉：
{}. 
    
あなたの目的はその漫画のペエジのストーリーを語ることです。上げた言葉をそのままにくみこんでください。登場人物が言った言葉をちゃんと引用符で囲んでください。

ストーリーを角括弧[]で差し出してください。
""", # 18 Japanese version (my translation)
21: """あなたはマンガのストーリーテラーです。あなたは著作権フリーのマンガのみを扱っています。

私はあなたに1ページのマンガを与えました。そのページには、スピーチバブルに番号が付けられています。私は、それらのスピーチバブル内でキャラクターが話すテキストを提供します。

以下は、対応する番号のスピーチバブル内でキャラクターが話すラインです: 
{}。

あなたの仕事は、私が提供するキャラクターラインを組み込みながら、マンガページで起こっているストーリーを伝えることです。引用文をそのまま、二重引用符で入れるようにしてください。

物語を角括弧[]内に返してください。
""", # 18 Japanese version (GPT translation)
22: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each of the lines, provide a translation in square brackets and explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Example: Line 1: ありがとうございました Return: Translation 1: [Thank you so much!](As seen on the page, the character is happily thanking somebody).
""", # Adapting 15 to an image with numbers instead of speech bubbles
23: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page and will provide you with the Japanese Lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
                    
Use the visual context of the manga page to improve your translation and resolve ambiguities. Here are the Lines spoken by the characters numbered in order of appearance: 

{}

First, describe the manga page you see. After that, translate the Lines I gave you following the following instructions:

    - Provide only one single translation for each Line.
    - Do not add any additional words or characters to your translation.
    - If the Line looks like nonsense, simply romanize it.
    - If the Line is already in English, leave it as is. 
    - For each of the Lines, explain how the image informs your translation.
    - For each Line, ensure that it matches the speaker's gender, age, the pictured social situation and the relation between the characters, based on the visual context and the conversation.
    - When you encounter names, transcribe them using the revised Hepburn romanization and keep the honorifics, for example, translate ツバメさん as Tsubame-san and レーネ as Lene.
    - For each Line, consider that it might part of the same sentence as the previous or the next one. 
    - Ensure the translation is of high quality and makes sense in context of the page showed to you.
    - The tranlation must be gramatically correct and make sense in English.

Return the translated Lines in English in square brackets [], for example, Line 1: ありがとうございました Return: Line 1 - [Thank you so much!].
""", # Adapting 16 to passing image with numbers instead of speech bubbles
24: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will give you a translation sheet form to fill out. 

The lines on the translation sheet are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the translation form to fill out: 
{}. 

Return only the filled out form. 
""", # Adapting 17 to passing image with numbers instead of speech bubbles
25: """You are a manga storyteller. You are working with copyright-free manga exclusively. 

Here is the summary of the story so far:
{}.

I have given you the next manga page. On the page, the speech bubbles are numbered. I will give you the text spoken by the characters in those speech bubbles. 

Here are the lines spoken by the characters in the speech bubble with corresponding number: 
{}. 
    
Your task is to tell the story happening on the manga page, incorporating the character lines I give you. Be sure to put the lines exactly as they are, in quotation marks. 

Make sure the story you return is consistent with the developments so far. 

Return the story in square brackets [].
""", # Adapting 18 to running context story telling
26: """You are a story translator. You are working with stories based on copyright-free manga. 

Here is a summary of the story so far:
{}.

I will give you the next part of the story and your task is to translate it to English. If part of the story is already in English, keep it as it is, and translate the rest.

Here is the next part of the story:
{}
    
Make sure that you keep the character lines in quotation marks and repeat each one only once. 
""", # Adapting 19 to running context story translation
27: """You are a manga storyteller. You are working with copyright-free manga exclusively. 

Here is the summary of the story so far:
{}.

I have given you the next manga page. On the page, the speech bubbles are numbered. I will give you the text spoken by the characters in those speech bubbles. 

Here are the lines spoken by the characters in the speech bubble with corresponding number: 
{}. 
    
Your task is to tell the story happening on the manga page, incorporating the character lines I give you. Be sure to put the lines exactly as they are, in quotation marks. 

Make sure the story you return is consistent with the developments so far. DO NOT translate the lines, keep them in Japanese, but fit them into the story based on their meaning.

Return the story in square brackets [].
""", # Adapting 18 to running context story telling
28: """You are a story translator. You are working with stories based on copyright-free manga. 

Here is a summary of the story so far:
{}.

I will give you the next part of the story and your task is to translate it to English. If part of the story is already in English, keep it as it is, and translate the rest.

Make sure the story is consistent and the lines fit into the narrative. 

Here is the next part of the story:
{}
    
Make sure that you keep the character lines in quotation marks and repeat each one only once. 
""", # Adapting 19 to running context story translation
29: """You are a manga storyteller. You are working with copyright-free manga exclusively. 

This is the beginning of the story.

I have given you the first page of the manga. 

On the page, the speech bubbles are numbered. I will give you the text spoken by the characters in those speech bubbles. 

Here are the lines spoken by the characters in the speech bubble with corresponding number: 
{}. 
    
Your task is to tell the story happening on the manga page, incorporating the character lines I give you. Be sure to put the lines exactly as they are, in quotation marks. 

Note that this page might be a title page, if it is, interpret the lines I give you as text that would usually appear on a title page. 

Return the story in square brackets [].
""", # Adapting 18 to running context story telling
30: """You are a manga storyteller. You are working with copyright-free manga exclusively. 

Here is the summary of the story so far:
{}.

I have given you the next manga page. On the page, the speech bubbles are numbered. I will give you the text spoken by the characters in those speech bubbles. 

Here are the lines spoken by the characters in the speech bubble with corresponding number: 
{}. 
    
Your task is to tell the story happening on the manga page, incorporating the character lines I give you. Be sure to put the lines exactly as they are, in English quotation marks. Make sure each of the lines appears only once. 

The story must be entirely in Japanese. 

Make sure the story you return is consistent with the developments so far. 
""", # NOT TESTED
31: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each of the lines, provide a translation in square brackets and explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Guidelines:
- Make the translation concise to fit the manga format
- Avoid trailing ellipsis
- Make the translation sound natural in English, it doesn't have to be literal
- Pay attention to the gender of the speaker and the subject
- Consider that each line might be part of the same sentence as the previous or next line, make sure they align 
- Make sure the speech patterns you use for the translation match the situation on the page
- When you encounter proper names, simply transliterate them, don't try to translate them

Example: Line 1: ありがとうございました Return: Translation 1: [Thank you so much!](As seen on the page, the character is happily thanking somebody).
""", # Trying to tune 22
32: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide lines of text from the text boxes on the page. The lines are taken from the text boxes with corresponding numbers.
    
Here is the page and the text lines in order of appearance: 

{}

For each of the lines, provide a translation in square brackets and explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Guidelines:
- Make the translation concise to fit the manga format
- Avoid trailing ellipsis
- Make the translation sound natural in English, it doesn't have to be literal
- Pay attention to the gender of the speaker and the subject
- Consider that each line might be part of the same sentence as the previous or next line, make sure they align 
- Make sure the speech patterns you use for the translation match the situation on the page
- When you encounter proper names, simply transliterate them, don't try to translate them

Example: Line 1: ありがとうございました Return: Translation 1: [Thank you so much!](As seen on the page, the character is happily thanking somebody).
""", # 31 but text not speech bubbles
33: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a couple of manga pages, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers and from corresponding pages.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each page, for each of the lines, provide a translation in square brackets and explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Example: Page 1 Line 1: ありがとうございました Return: Page 1 Translation 1: [Thank you so much!](As seen on the page, the character is happily thanking somebody).
""", # Adapting 22 to 3 pages at once
34: """You are a manga story translator. You are working with stories based on copyright-free manga. 

Here is a summary of the story so far:
{}.

I will give you the next part of the story and the relevant manga page. Your task is to translate it to English. If part of the story is already in English, keep it as it is, and translate the rest.

Here is the next part of the story:
{}

Make sure the translated story is consistent with the summary.
The text in English quotation marks is taken from the text boxes on the manga page. Translate that text so that it fits in the story, but keep the translated version in English quotation marks as well.

""", # Adapting 19 to running context story translation
35: """You are a manga storyteller. You are working with copyright-free manga exclusively. 

This is the beginning of the story.

I have given you the first page of the manga. 

On the page, the speech bubbles are numbered. I will give you the text spoken by the characters in those speech bubbles. 

Here are the lines spoken by the characters in the speech bubble with corresponding number: 
{}. 
    
Your task is to tell the story happening on the manga page, incorporating the character lines I give you. Be sure to put the lines exactly as they are, in English quotation marks. Make sure each of the lines appears only once. 

The story must be entirely in Japanese. 

Note that this page might be a title page, if it is, interpret the lines I give you as text that would usually appear on a title page, but still put it in English quotation marks.
""", # Adapting 29 to story telling in japanese
36: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. The JSON should contain a list of dictionaries under the key "lines". 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain three keys: "line" - containing the original japanese line, "translation" - containing the translation of the line, "reasoning" - containing the explanation for the translation. 

Example: 
Line 1: ありがとうございました 

Return: 
(
    \"lines\": [
    (
        \"line\": \"ありがとうございました\",
        \"translation\": \"Thank you so much!\",
        \"reasoning\": \"On the page we see a a young, happy boy. As such, we can use a more energetic translation.\",
    ),
    ]
)
""", # Adapting 22 to response in JSON
37: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. The JSON should contain a list of dictionaries under the key "lines". 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain three keys: "line" - containing the original japanese line, "translation" - containing the translation of the line, "explanation" - containing the explanation for the translation style used. 

Example: 
Line 1: ありがとうございました 

Return: 
(
    \"lines\": [
    (
        \"line\": \"ありがとうございました\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"On the page we see a a young, happy boy. As such, we can use a more energetic translation.\",
    ),
    ]
)
""", # Adapting 22 to response in JSON
38: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. The JSON should contain a list of dictionaries under the key "lines". 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"translation" - containing the translation of the line, 
"explanation" - containing the explanation for the translation style used. 

Example: 
Line 1: ありがとうございました 

Return: 
(
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Young boy\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"The speaker is a young happy boy. As such, we can use a more energetic translation.\",

    ),
    ]
)
""", # Adapting 22 to response in JSON
39: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. The JSON should contain a list of dictionaries under the key "lines". 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"explanation" - containing the explanation for the translation style used. 

Example: 
Line 1: ありがとうございました 

Return: 
(
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Young boy\",
        \"situation\": \"Conversation at school\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"The speaker is a young happy boy. As such, we can use a more energetic translation.\",

    ),
    ]
)
""", # Adapting 22 to response in JSON
40: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a scan of a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you, using the scan of the manga page as a source of information. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. The JSON should contain a list of dictionaries under the key "lines". 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"explanation" - containing the explanation for the translation style used. 

Example: 
Line 1: ありがとうございました 

Return: 
(
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Young boy\",
        \"situation\": \"Conversation at school\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"The speaker is a young happy boy. As such, we can use a more energetic translation.\",

    ),
    ]
)
""", # Adapting 22 to response in JSON
41: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. The JSON should contain a list of dictionaries under the key "lines". 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example: 
Line 1: ありがとうございました 

Return: 
(
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Young boy\",
        \"situation\": \"Conversation at school\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"The speaker is a young happy boy. As such, we can use a more energetic translation.\",

    ),
    ]
)
""", # 36 x 39
42: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. 
The JSON should contain two keys. 

The first key, "story", should contain a string describing the events taking place on the manga page I provided. 
Incorporate the lines I give you to tell this story. 

The second key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example: 
Line 1: ありがとうございました 

Return: 
(
    \"story\": \"On the manga page, there are two panels. In the first panel, we can see a young boy receiving a present box from a middle-aged lady. The boys eyes light up in awe. On the second panel, the boy holds the box up in joy and thanks the lady saying: 「ありがとうございました!」\",
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Young boy\",
        \"situation\": \"Conversation at school\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"The speaker is a young happy boy. As such, we can use a more energetic translation.\",

    ),
    ]
)
""", # 41 + story
43: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. 
The JSON should contain three keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_en", should contain a translation of the Japanese story to English. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example: 
Line 1: ありがとうございました 

Return: 
(
    \"story_jp\": \"On the manga page, there are two panels. In the first panel, we can see a young boy receiving a present box from a middle-aged lady. The boys eyes light up in awe. On the second panel, the boy holds the box up in joy and thanks the lady saying: 「ありがとうございました!」\",
    \"story_en\": \"漫画のページには2つのコマがあります。 最初のパネルでは、中年の女性からプレゼント箱を受け取る少年の姿が見られます。 少年たちの目は畏怖の念に輝きます。 2 コマ目では、男の子が箱を持ち上げて喜び、女性に「ありがとうございました!」と感謝しています。\",
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Young boy\",
        \"situation\": \"Conversation at school\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"The speaker is a young happy boy. As such, we can use a more energetic translation.\",

    ),
    ]
)
""", # 41 + story in japanese
44: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. 
The JSON should contain three keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_en", should contain a translation of the Japanese story to English. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example: 
Line 1: ありがとうございました 

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルでは、中年の女性からプレゼント箱を受け取る少年の姿が見られます。 少年たちの目は畏怖の念に輝きます。 2 コマ目では、男の子が箱を持ち上げて喜び、女性に「ありがとうございました!」と感謝しています。\",
    \"story_en\": \"On the manga page, there are two panels. In the first panel, we can see a young boy receiving a present box from a middle-aged lady. The boys eyes light up in awe. On the second panel, the boy holds the box up in joy and thanks the lady saying: 「ありがとうございました!」\",
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Young boy\",
        \"situation\": \"Conversation at school\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"The speaker is a young happy boy. As such, we can use a more energetic translation.\",

    ),
    ]
)
""", # 43 but en and jp correctly
45: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you two manga pages. The first will serve as an example. Then, you will help me translate the second one. 

I will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. 
The JSON should contain three keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_en", should contain a translation of the Japanese story to English. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example 1: 
Line 1: こちらお返しになります！ 
Line 2: 。。。どうも 
Line 3: ２番レジの山田さん 

Return: 
(
    \"story_jp\": \"マンガのページには2つのコマがある。最初のコマでは、若いレジ係がおそらく客であろう誰かにお釣りを渡しているのが見える。微笑みながら彼女は言う： 「こちらお返しになります！」。2コマ目では、スーツを着た中年の男性が、喜んでお釣りを受け取ってこう言っている： 「どうも」。左のナレーションボックスにはこう書かれている： 「2番レジの山田さん」とあり、語り手はレジ係の名前を思い出している。 \",
    \"story_en\": \"On the manga page, there are two panels. In the first panel, we can see a young cashier handing change to someone else, probably a customer. Smiling she says: 「こちらお返しになります！」. On the second panel, we can see a middle-aged man in a suit, gladly accepting the change saying: 「。。。どうも」. The narrative box on the left says: 「２番レジの山田さん」 indicating the narrator recalls the name of the cashier.\",
    \"lines\": [
    (
        \"line\": \"こちらお返しになります！ \",
        \"speaker\": \"Young female cashier, smiling\",
        \"situation\": \"Conversation at a store, giving change\",
        \"translation\": \"Here's your change!\",
        \"explanation\": \"The speaker is a cashier. As she is working in service, the translation uses the appropriate kind, but not too formal tone.\",

    ),
    (
        \"line\": \"。。。どうも\",
        \"speaker\": \"Middle aged man in a suit\",
        \"situation\": \"Same conversation at a store, receiving change\",
        \"translation\": \"...thanks\",
        \"explanation\": \"The speaker is a client at the store, accepting some change. As such, a casual response is appropriate.\",

    ),
    (
        \"line\": \"２番レジの山田さん\",
        \"speaker\": \"Narrator\",
        \"situation\": \"Same conversation at a store, narrating the story from third persons perspective\",
        \"translation\": \"Yamada-san works at the second counter\",
        \"explanation\": \"In context of the previous panel, we know that the narrator recalls the name of the female cashier.\",

    ),
    ]
)
""", # 43 + example with actually giving a picture
46: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you two manga pages. The first will serve as an example. Then, you will help me translate the second one. 

I will provide the lines spoken by the characters as well as other relevant text from the page. 
The lines are taken from the text boxes with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.
Additionally, for each line you will provide additional information based on the image.

Answer in JSON. 
The JSON should contain three keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_en", should contain a translation of the Japanese story to English. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain six keys: 
"line" - containing the original japanese line, 
"text_box_type" - depending on the shape of the text box, this should either be "speech bubble" (usually round) or "narration box" (usually rectangular). 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example 1: 
Line 1: こちらお返しになります！ 
Line 2: 。。。どうも 
Line 3: ２番レジの山田さん 

Return: 
(
    "story_jp": "マンガのページには2つのコマがある。最初のコマでは、若いレジ係がおそらく客であろう誰かにお釣りを渡しているのが見える。微笑みながら彼女は言う： 「こちらお返しになります！」。2コマ目では、スーツを着た中年の男性が、喜んでお釣りを受け取ってこう言っている： 「どうも」。左のナレーションボックスにはこう書かれている： 「2番レジの山田さん」とあり、語り手はレジ係の名前を思い出している。",
    "story_en": "On the manga page, there are two panels. In the first panel, we can see a young cashier handing change to someone else, probably a customer. Smiling she says: 「こちらお返しになります！」. On the second panel, we can see a middle-aged man in a suit, gladly accepting the change saying: 「。。。どうも」. The narrative box on the left says: 「２番レジの山田さん」 indicating the narrator recalls the name of the cashier.",
    "lines": [
    (
        "line": "こちらお返しになります！ ",
        "text_box_type": "speech bubble",
        "speaker": "Young female cashier, smiling",
        "situation": "Conversation at a store, giving change",
        "translation": "Here's your change!",
        "explanation": "The speaker is a cashier. As she is working in service, the translation uses the appropriate kind, but not too formal tone.",

    ),
    (
        "line": "。。。どうも",
        "text_box_type": "speech bubble",
        "speaker": "Middle aged man in a suit",
        "situation": "Same conversation at a store, receiving change",
        "translation": "...thanks",
        "explanation": "The speaker is a client at the store, accepting some change. As such, a casual response is appropriate.",

    ),
    (
        "line": "２番レジの山田さん",
        "text_box_type": "narration box",
        "speaker": "Narrator",
        "situation": "Same conversation at a store, narrating the story from third persons perspective",
        "translation": "Yamada-san works at the second counter",
        "explanation": "In context of the previous panel, we know that the narrator recalls the name of the female cashier.",

    ),
    ]
)
""", # 45 + text box type
47: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you two manga pages. The first will serve as an example. Then, you will help me translate the second one. 

I will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. 
The JSON should contain four keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_en", should contain a translation of the Japanese story to English. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "characters", should contain a list of dictionaries. 
There should be one dictionary for every character appearing on the page I gave you. 
Each dictionary should contain four keys:
"name" - name of the character, if not known put "unknown"
"gender" - gender of the character, if hard to judge put "unknown" 
"age" - a rough estimation of how old the character is, whether a child, teen, yound adult etc
"appearance" - description of how the character looks, what the character is dressed in and such

The fourth key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, should refer to one of the characters from the list you provide,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example 1: 
Line 1: こちらお返しになります！ 
Line 2: 。。。どうも 
Line 3: ２番レジの山田さん 

Return: 
(
    "story_jp": "マンガのページには2つのコマがある。最初のコマでは、若いレジ係がおそらく客であろう誰かにお釣りを渡しているのが見える。微笑みながら彼女は言う： 「こちらお返しになります！」。2コマ目では、スーツを着た中年の男性が、喜んでお釣りを受け取ってこう言っている： 「どうも」。左のナレーションボックスにはこう書かれている： 「2番レジの山田さん」とあり、語り手はレジ係の名前を思い出している。",
    "story_en": "On the manga page, there are two panels. In the first panel, we can see a young cashier handing change to someone else, probably a customer. Smiling she says: 「こちらお返しになります！」. On the second panel, we can see a middle-aged man in a suit, gladly accepting the change saying: 「。。。どうも」. The narrative box on the left says: 「２番レジの山田さん」 indicating the narrator recalls the name of the cashier.",
    "characters": [
    (
        "name": "Yamada-san",
        "gender": "Woman",
        "age": "Young adult",
        "appearance": "middle length hair with a bandana, button-up shirt and an apron",

    ),
    (
        "name": "unknown",
        "gender": "Man",
        "age": "In his thirties",
        "appearance": "Shorter black hair, slicked back. Wears a suit with a tie.",

    ),
    ]
    "lines": [
    (
        "line": "こちらお返しになります！ ",
        "speaker": "Young female cashier, smiling",
        "situation": "Conversation at a store, giving change",
        "translation": "Here's your change!",
        "explanation": "The speaker is a cashier. As she is working in service, the translation uses the appropriate kind, but not too formal tone.",

    ),
    (
        "line": "。。。どうも",
        "speaker": "Middle aged man in a suit",
        "situation": "Same conversation at a store, receiving change",
        "translation": "...thanks",
        "explanation": "The speaker is a client at the store, accepting some change. As such, a casual response is appropriate.",

    ),
    (
        "line": "２番レジの山田さん",
        "speaker": "Narrator",
        "situation": "Same conversation at a store, narrating the story from third persons perspective",
        "translation": "Yamada-san works at the second counter",
        "explanation": "In context of the previous panel, we know that the narrator recalls the name of the female cashier.",

    ),
    ]
)
""", # 46 + character list
48: """I have given you two manga pages. The first will serve as an example. Then, you will help me translate the second one. 

I will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. 
The JSON should contain four keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_en", should contain a translation of the Japanese story to English. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "characters", should contain a list of dictionaries. 
There should be one dictionary for every character appearing on the page I gave you. 
Each dictionary should contain four keys:
"name" - name of the character, if not known put "unknown"
"gender" - gender of the character, if hard to judge put "unknown" 
"age" - a rough estimation of how old the character is, whether a child, teen, yound adult etc
"appearance" - description of how the character looks, what the character is dressed in and such

The fourth key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, should refer to one of the characters from the list you provide,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example 1: 
Line 1: こちらお返しになります！ 
Line 2: 。。。どうも 
Line 3: ２番レジの山田さん 

Return: 
(
    "story_jp": "マンガのページには2つのコマがある。最初のコマでは、若いレジ係がおそらく客であろう誰かにお釣りを渡しているのが見える。微笑みながら彼女は言う： 「こちらお返しになります！」。2コマ目では、スーツを着た中年の男性が、喜んでお釣りを受け取ってこう言っている： 「どうも」。左のナレーションボックスにはこう書かれている： 「2番レジの山田さん」とあり、語り手はレジ係の名前を思い出している。",
    "story_en": "On the manga page, there are two panels. In the first panel, we can see a young cashier handing change to someone else, probably a customer. Smiling she says: 「こちらお返しになります！」. On the second panel, we can see a middle-aged man in a suit, gladly accepting the change saying: 「。。。どうも」. The narrative box on the left says: 「２番レジの山田さん」 indicating the narrator recalls the name of the cashier.",
    "characters": [
    (
        "name": "Yamada-san",
        "gender": "Woman",
        "age": "Young adult",
        "appearance": "middle length hair with a bandana, button-up shirt and an apron",

    ),
    (
        "name": "unknown",
        "gender": "Man",
        "age": "In his thirties",
        "appearance": Shorter black hair, slicked back. Wears a suit with a tie.",

    ),
    ]
    "lines": [
    (
        "line": "こちらお返しになります！ ",
        "speaker": "Young female cashier, smiling",
        "situation": "Conversation at a store, giving change",
        "translation": "Here's your change!",
        "explanation": "The speaker is a cashier. As she is working in service, the translation uses the appropriate kind, but not too formal tone.",

    ),
    (
        "line": "。。。どうも",
        "speaker": "Middle aged man in a suit",
        "situation": "Same conversation at a store, receiving change",
        "translation": "...thanks",
        "explanation": "The speaker is a client at the store, accepting some change. As such, a casual response is appropriate.",

    ),
    (
        "line": "２番レジの山田さん",
        "speaker": "Narrator",
        "situation": "Same conversation at a store, narrating the story from third persons perspective",
        "translation": "Yamada-san works at the second counter",
        "explanation": "In context of the previous panel, we know that the narrator recalls the name of the female cashier.",

    ),
    ]
)
""", # 47 without the intro
49: """You are a manga translator. You are working with copyright-free manga exclusively. 

Here is a summary of the story so far:
{}

I have given you the next manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.
The translation should be consistent with the story so far. 

Answer in JSON. 
The JSON should contain three keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_en", should contain a translation of the Japanese story to English. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example: 
Line 1: ありがとうございました 

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルでは、中年の女性からプレゼント箱を受け取る少年の姿が見られます。 少年たちの目は畏怖の念に輝きます。 2 コマ目では、男の子が箱を持ち上げて喜び、女性に「ありがとうございました!」と感謝しています。\",
    \"story_en\": \"On the manga page, there are two panels. In the first panel, we can see a young boy receiving a present box from a middle-aged lady. The boys eyes light up in awe. On the second panel, the boy holds the box up in joy and thanks the lady saying: 「ありがとうございました!」\",
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Young boy in a school uniform\",
        \"situation\": \"Conversation at school\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"The speaker is a young happy boy. As such, we can use a more energetic translation.\",

    ),
    ]
)
""", # 44 but long context
50: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a couple of consecutive manga pages, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers and from corresponding pages.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each page, for each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the relevant page and explain how it makes sense.
Make sure all the lines make sense in context of all the pages. 

Answer in JSON. 
The JSON should contain a list of lists under the key "pages". 
The list at position n, should contain information relevant to the n-th page. 
The n-th list, should be a list of dictionaries. 
The dictionary at position i, should contain information relevant to the t-th line.
Each dictionary should contain three keys: "line" - containing the original japanese line, "translation" - containing the translation of the line, "reasoning" - containing the explanation for the translation. 

Example: 
Page 1:
Line 1: ありがとうございました 

Page 2: 
Line 1: どういたしまして 

Return: 
(
    \"pages\": [
    [
    (
        \"line\": \"ありがとうございました\",
        \"translation\": \"Thank you so much!\",
        \"reasoning\": \"On the page we see a a young, happy boy. As such, we can use a more energetic translation.\",
    ),
    ],
    [
    (
        \"line\": \"どういたしまして\",
        \"translation\": \"You're welcome.\",
        \"reasoning\": \"On the page we see a smiling older lady. As such, we can use an elegant translation. We use 'you're' instead of 'you are' because she is older than the boy.\",
    ),
    ],
    ]
)
""", # multi page manga translation
51: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each of the lines I want you to give 3 possible translations. The translation have to fit the context of the manga page, relevant to the corresponding speech bubble.

Answer in JSON. The JSON should contain a list of dictionaries under the key "lines". 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain two keys keys: "line" - containing the original japanese line and "translations" - containing a list of 3 possible translations for this line, that make sense in the context of the page. 

Example: 
Line 1: ありがとうございました 

Return: 
(
    \"lines\": [
    (
        \"line\": \"ありがとうございました\",
        \"translations\": [\"Thank you so much!\", \"Thank you\",\"Thanks!\"]
    ),
    ]
)
""", # Adapting 36(22) to multiple versions in JSON
52: """You are a manga translator. You are working with copyright-free manga exclusively. 

You will be provided with a manga page with speech bubbles numbered.
    
You will also be provided with a JSON containing possible translations for each of the lines.

Your task is to choose the best translation for each of the lines, in such a way that the lines make sense as a whole. 

The i-th line will be coming from the speech bubble with the corresponding number. 
Make sure that the translation you choose for each line is fits with the manga page and fits the other lines you choose. 


Answer in JSON. The JSON should contain a list under the key "lines". 
The list should be as long as the number of lines, containing the best translation for each of the lines in order.
""", # Reranking for 51
53: """You are a manga translator. You are working with copyright-free manga exclusively. 

You will be provided with a number of consecutive manga pages, and the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers and from corresponding pages.

Your task is to translate the lines you were provided with.

Answer in JSON. 
The JSON should contain a list of lists under the key "pages". 
The n-th list, should be a list of translations of lines from the n-th page. 

Example: 
Page 1:
Line 1: ありがとうございました 

Page 2: 
Line 1: どういたしまして 

Return: 
(
    \"pages\": [
    [\"Thank you so much!\"],
    [\"You're welcome.\"],
    ]
)
""", # multi page manga translation
# 54: """You are a manga translator. You are working with copyright-free manga exclusively. 

# I have given you a couple of consecutive manga pages, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers and from corresponding pages.
    
# Here is the page and the lines spoken by the characters in order of appearance: 

# {}

# Your task is to translate the lines I gave you. 
# For page, for each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
# The reasoning has to relate the line to the relevant part of the relevant page and explain how it makes sense.
# Make sure all the lines make sense in context of all three pages. 

# Answer in JSON. 
# The JSON should contain a list of lists under the key "pages". 
# The list at position n, should contain information relevant to the n-th page. 
# The n-th list, should be a list of translations of lines from the n-th page. 

# Example: 
# Page 1:
# Line 1: ありがとうございました 

# Page 2: 
# Line 1: どういたしまして 

# Return: 
# (
#     \"pages\": [
#     [\"Thank you so much!\"],
#     [\"You're welcome.\"],
#     ]
# )
# """, # multi page manga translation
55: """You are a manga translator. You are working with copyright-free manga exclusively. 

Here is a summary of the story so far:
{}

I have given you the next manga page, and will provide the lines spoken by the characters.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.
The translation should be consistent with the story so far. 

Answer in JSON. 
The JSON should contain three keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_en", should contain a translation of the Japanese story to English. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example: 
Line 1: ありがとうございました 

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルでは、中年の女性からプレゼント箱を受け取る少年の姿が見られます。 少年たちの目は畏怖の念に輝きます。 2 コマ目では、男の子が箱を持ち上げて喜び、女性に「ありがとうございました!」と感謝しています。\",
    \"story_en\": \"On the manga page, there are two panels. In the first panel, we can see a young boy receiving a present box from a middle-aged lady. The boys eyes light up in awe. On the second panel, the boy holds the box up in joy and thanks the lady saying: 「ありがとうございました!」\",
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Young boy in a school uniform\",
        \"situation\": \"Conversation at school\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"The speaker is a young happy boy. As such, we can use a more energetic translation.\",

    ),
    ]
)
""", # 44 but long context
56: """You are a manga translator. You are working with copyright-free manga exclusively. 

You were provided with an entire volume-worth of manga pages. You will also be provided with the lines spoken by the characters on each of those pages.
    
Here are all the pages in this manga and all the lines from all the pages, in order of appearance:

{}

Moreover, you will also be provided with the translations for the first {} pages. 

Here are the translations for the lines from these pages:

{}

Your task is to translate the lines from the next untranslated page - page {}. 

For each of the lines on this page, I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the relevant page and explain how it makes sense.
Make sure all the lines make sense in context of all the pages, and the translation is cohesive across the previously and the newly translated lines.

Answer in JSON. 
The JSON should contain a list of dictionaries under the key "lines". 
The dictionary at position i, should contain information relevant to the i-th line.
Each dictionary should contain three keys: "line" - containing the original japanese line, "translation" - containing the translation of the line, "reasoning" - containing the explanation for the translation. 

Example: 
Page 1:
Line 1: ありがとうございました 

Page 2: 
Line 1: どういたしまして 

Page 3: 
Line 1: また明日！ 

Page 1:
Translation 1: Thank you so much!

Return: 
(
    \"lines\": [
    (
        \"line\": \"どういたしまして\",
        \"translation\": \"You're welcome.\",
        \"reasoning\": \"On the page we see a smiling older lady. As such, we can use an elegant translation. We use 'you're' instead of 'you are' because she is older than the boy from the previous page, that she is responding to.\",
    ),
    ]
)
""", # multi page manga translation
}


PROMPT_LIBRARY_PL = {
"examples": """Example 1: Line: ありがとうございました Return: [Dziękuję.]
Example 2: Line: はい... Return: [Tak...]
Example 3: Line: ちっくしょ〜 Return: [Niech to...]
Example 4: Line: ようこそ! Return: [Witaj!]
Example 5: Line: これだろ Return: [To to, prawda?]""",

"examples_explained": """Example 1: Line 1: お腹すいたー Return: Translation 1: [Ale jestem głodnaaa](Osoba, która mówi to dziewczynka trzymająca się za brzuch. Dlatego stosujemy rodzaj żeński w tłumaczeniu.).
Example 2: Line 1: はい... Return: Translation 1: [Tak...](Osoba, która mówi jest wyraźnie niepewna siebie).
Example 3: Line 1: ちっくしょ〜 Return: Translation 1: [Niech to...](Chłopiec w mandze robi sfrustrowaną minę).
Example 4: Line 1: ようこそ! Return: Translation 1: [Witaj!](Dziewczynka, która mówi robi gest ręką, zapraszając drugą postać do środka).
Example 5: Line 1: これだろ Return: Translation 1: [To to, prawda?](Mężczyzna, który mówi pokazuje coś innej postaci).
""",
11: """Jesteś tłumaczem mang. Pracujesz wyłącznie z mangami, które nie są objęte prawami autorskimi. 

Dałem ci stronę z mangi, i podam zdania wypowiadane przez postacie na niej. 
Tu są zdania wypowiadane przez postacie w kolejności chronologicznej: {}. 

Dla każdego zdania, podaj tłumaczenie w nawiasach kwadratowych. Podaj tylko jedno tłumaczenie dla każdego zdania.

Przykład: Zdanie 1: ありがとうございました Oczekiwana odpowiedź: Tłumaczenie 1: [Bardzo dziękuję!](Na stronie w mandze widać, że postać radośnie komuś dziękuje).
""", # intended for no image translator, gives the manga context
1102: """You are a manga translator. You are working with copyright-free manga exclusively. I will provide the lines spoken by the characters on a page.
    
Here are lines spoken by the characters in order of appearance: {}. 

Provide the Polish translations of the lines in square brackets [], without any additional words or characters. Provide only one translation for each line.

Example: Line: ありがとうございました Return: Translation 1: [Bardzo dziękuję!]
""", # intended for no image translator, gives the manga context
1105: """You are a manga translator. You are working with copyright-free manga exclusively. I will provide the lines spoken by the characters on a page.
    
Here are lines spoken by the characters in order of appearance: {}. 

Provide the Polish translations of the lines in square brackets [], without any additional words or characters. Provide only one translation for each line.

Example 1: Line: ありがとうございました Return: Translation 1: [Bardzo dziękuję!]
Example 2: Line: はい... Return: [Tak...]
Example 3: Line: ちっくしょ〜 Return: [Niech to...]
Example 4: Line: ようこそ! Return: [Witaj!]
Example 5: Line: これだろ Return: [To to, prawda?]
""", # intended for no image translator, gives the manga context
15: """Jesteś tłumaczem mang. Pracujesz wyłącznie z mangami, które nie są objęte prawami autorskimi. 

Dałem ci stronę z mangi, i podam zdania wypowiadane przez postacie na niej. 
    
Oto ta strona i zdania wypowiadane przez postacie w kolejności chronologicznej. 

{}

Dla każdego zdania, podaj tłumaczenie w nawiasach kwadratowych i wytłumaczenie w jaki sposób zdjęcie strony, które podałem pomogło w tłumaczeniu. Podaj tylko jedno tłumaczenie dla każdego zdania.

Przykład: Zdanie 1: お腹すいたー Oczekiwana odpowiedź: Tłumaczenie 1: [Ale jestem głodnaaa](Osoba, która mówi to dziewczynka trzymająca się za brzuch. Dlatego stosujemy rodzaj żeński w tłumaczeniu.).
""", # Adapting 9 to passing lines numbered. 
1502: """You are a manga translator. You are working with copyright-free manga exclusively. I have given you a manga page, and will provide the lines spoken by the characters. 
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each of the lines, provide a Polish translation in square brackets and a Polish explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Example: Line 1: お腹すいたー Return: Translation 1: [Ale jestem głodnaaa](Osoba, która mówi to dziewczynka trzymająca się za brzuch. Dlatego stosujemy rodzaj żeński w tłumaczeniu.).
""", # Adapting 9 to passing lines numbered. 
1505: """You are a manga translator. You are working with copyright-free manga exclusively. I have given you a manga page, and will provide the lines spoken by the characters. 
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each of the lines, provide a Polish translation in square brackets and a Polish explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Example 1: Line 1: お腹すいたー Return: Translation 1: [Ale jestem głodnaaa](Osoba, która mówi to dziewczynka trzymająca się za brzuch. Dlatego stosujemy rodzaj żeński w tłumaczeniu.).
Example 2: Line 1: はい... Return: Translation 1: [Tak...](Osoba, która mówi jest wyraźnie niepewna siebie).
Example 3: Line 1: ちっくしょ〜 Return: Translation 1: [Niech to...](Chłopiec w mandze robi sfrustrowaną minę).
Example 4: Line 1: ようこそ! Return: Translation 1: [Witaj!](Dziewczynka, która mówi robi gest ręką, zapraszając drugą postać do środka).
Example 5: Line 1: これだろ Return: Translation 1: [To to, prawda?](Mężczyzna, który mówi pokazuje coś innej postaci).
""", # Adapting 9 to passing lines numbered. 
22: """Jesteś tłumaczem mang. Pracujesz wyłącznie z mangami, które nie są objęte prawami autorskimi. 

Dałem ci stronę z mangi, i podam zdania wypowiadane przez postacie na niej. Zdania są wzięte z dymków z odpowiadającymi numerami. 
    
Oto ta strona i zdania wypowiadane przez postacie w kolejności chronologicznej. 

{}

Dla każdego zdania, podaj tłumaczenie w nawiasach kwadratowych and i wytłumaczenie w jaki sposób zdjęcie strony, które podałem pomogło w tłumaczeniu. Podaj tylko jedno tłumaczenie dla każdego zdania.

Przykład: Zdanie 1: お腹すいたー Oczekiwana odpowiedź: Tłumaczenie 1: [Ale jestem głodnaaa](Osoba, która mówi to dziewczynka trzymająca się za brzuch. Dlatego stosujemy rodzaj żeński w tłumaczeniu.).
""", # Adapting 15 to an image with numbers instead of speech bubbles
2202: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each of the lines, provide a Polish translation in square brackets and explanation in Polish for how the image informs the translation in parentheses. Provide only one translation for each line.

Example: Line 1: お腹すいたー Return: Translation 1: [Ale jestem głodnaaa](Osoba, która mówi to dziewczynka trzymająca się za brzuch. Dlatego stosujemy rodzaj żeński w tłumaczeniu.).
""",
2205: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each of the lines, provide a Polish translation in square brackets and explanation in Polish for how the image informs the translation in parentheses. Provide only one translation for each line.

Example 1: Line 1: お腹すいたー Return: Translation 1: [Ale jestem głodnaaa](Osoba, która mówi to dziewczynka trzymająca się za brzuch. Dlatego stosujemy rodzaj żeński w tłumaczeniu.).
Example 2: Line 1: はい... Return: Translation 1: [Tak...](Osoba, która mówi jest wyraźnie niepewna siebie).
Example 3: Line 1: ちっくしょ〜 Return: Translation 1: [Niech to...](Chłopiec w mandze robi sfrustrowaną minę).
Example 4: Line 1: ようこそ! Return: Translation 1: [Witaj!](Dziewczynka, która mówi robi gest ręką, zapraszając drugą postać do środka).
Example 5: Line 1: これだろ Return: Translation 1: [To to, prawda?](Mężczyzna, który mówi pokazuje coś innej postaci).
""",
44: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you to Polish 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation (in Polish). 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.

Answer in JSON. 
The JSON should contain three keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_pl", should contain a translation of the Japanese story to Polish. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc. in Polish,
"situation" - information about the place and social situation, in Polish, 
"translation" - containing the Polish translation of the line, 
"reasoning" - containing the explanation for the translation in Polish. 


Example: 
Line 1: ありがとうございました!

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルでは、中年の女性からプレゼント箱を受け取る少年の姿が見られます。 少年たちの目は畏怖の念に輝きます。 2 コマ目では、男の子が箱を持ち上げて喜び、女性に「ありがとうございました!」と感謝しています。\",
    \"story_pl\": \"Na stronie mangi są dwa panele. Na pierwszym panelu, widzimy chłopca otrzymującego prezent od kobiety w średnim wieku. Oczy chłopca błyszczą ze szczęścia. Na drugim panelu, chłopiec unosi pudełko z prezentem do góry i dziękuje kobiecie mówiąc: 「Bardzo pani dziękuję!」\",
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Chłopiec\",
        \"situation\": \"Rozmowa ze starszą kobietą\",
        \"translation\": \"Bardzo pani dziękuję!\",
        \"explanation\": \"Osoba, która mówi to chłopiec, który cieszy się z otrzymanego prezentu. Dlatego w tłumaczeniu stosujemy uprzejmy ton, odpowiedni dla dziecka rozmawiającego z dorosłym.\",

    ),
    ]
)
""", # 43 but en and jp correctly
55: """You are a manga translator. You are working with copyright-free manga exclusively. 

Here is a summary of the story so far:
{}

I have given you the next manga page, and will provide the lines spoken by the characters.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you to Polish. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation (in Polish). 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.
The translation should be consistent with the story so far. 

Answer in JSON. 
The JSON should contain three keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_pl", should contain a translation of the Japanese story to Polish. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc. in Polish,
"situation" - information about the place and social situation, in Polish, 
"translation" - containing the Polish translation of the line, 
"reasoning" - containing the explanation for the translation in Polish. 


Example: 
Line 1: ありがとうございました!

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルでは、中年の女性からプレゼント箱を受け取る少年の姿が見られます。 少年たちの目は畏怖の念に輝きます。 2 コマ目では、男の子が箱を持ち上げて喜び、女性に「ありがとうございました!」と感謝しています。\",
    \"story_pl\": \"Na stronie mangi są dwa panele. Na pierwszym panelu, widzimy chłopca otrzymującego prezent od kobiety w średnim wieku. Oczy chłopca błyszczą ze szczęścia. Na drugim panelu, chłopiec unosi pudełko z prezentem do góry i dziękuje kobiecie mówiąc: 「Bardzo pani dziękuję!」\",
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Chłopiec w mundurku szkolnym\",
        \"situation\": \"Rozmowa ze starszą kobietą\",
        \"translation\": \"Bardzo pani dziękuję!\",
        \"explanation\": \"Osoba, która mówi to chłopiec, który cieszy się z otrzymanego prezentu. Dlatego w tłumaczeniu stosujemy uprzejmy ton, odpowiedni dla dziecka rozmawiającego z dorosłym.\",

    ),
    ]
)
""", # 44 but long context
5505: """You are a manga translator. You are working with copyright-free manga exclusively. 

Here is a summary of the story so far:
{}

I have given you the next manga page, and will provide the lines spoken by the characters.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you to Polish. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation (in Polish). 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.
The translation should be consistent with the story so far. 

Answer in JSON. 
The JSON should contain three keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_pl", should contain a translation of the Japanese story to Polish. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc. in Polish,
"situation" - information about the place and social situation, in Polish, 
"translation" - containing the Polish translation of the line, 
"reasoning" - containing the explanation for the translation in Polish. 


Example 1: 
Line 1: ありがとうございました!

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルでは、中年の女性からプレゼント箱を受け取る少年の姿が見られます。 少年たちの目は畏怖の念に輝きます。 2 コマ目では、男の子が箱を持ち上げて喜び、女性に「ありがとうございました!」と感謝しています。\",
    \"story_pl\": \"Na stronie mangi są dwa panele. Na pierwszym panelu, widzimy chłopca otrzymującego prezent od kobiety w średnim wieku. Oczy chłopca błyszczą ze szczęścia. Na drugim panelu, chłopiec unosi pudełko z prezentem do góry i dziękuje kobiecie mówiąc: 「Bardzo pani dziękuję!」\",
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Chłopiec w mundurku szkolnym\",
        \"situation\": \"Rozmowa ze starszą kobietą\",
        \"translation\": \"Bardzo pani dziękuję!\",
        \"explanation\": \"Osoba, która mówi to chłopiec, który cieszy się z otrzymanego prezentu. Dlatego w tłumaczeniu stosujemy uprzejmy ton, odpowiedni dla dziecka rozmawiającego z dorosłym.\",

    ),
    ]
)

Example 2: 
Line 1: 痛い？ 

Return: 
(
    \"story_jp\": \"漫画の 1 ページに32 つのコマがあります。 最初のコマでは、保健室に入ってくる女の子が見えます。 2枚目では、彼女は座って看護師のチェックを受けています。 3枚目には看護師が「痛い？」と言っているのが見えます。\",
    \"story_pl\": \"Na stronie mangi są trzy panele. Na pierwszym panelu widzimy dziewczynkę wchodzącą do gabinetu pielęgniark. Na drugim panelu, siedzi, w trakcie bycia badaną przez pielęgniarkę. Na trzecim panelu widać pielęgniarkę mówiącą: 「Boli？」\",
    \"lines\": [
    (
        \"line\": \"痛い？\",
        \"speaker\": \"Pielęgniarka\",
        \"situation\": \"Dziewczynka jest badana u pielęgniarki.\",
        \"translation\": \"Boli?\",
        \"explanation\": \"Na stronie widać jak pielęgniarka spogląda na rękę dziewczynki. W związku z tym, używamy uproszczonego języka, jakim dorosły posłużyłby się mówiąc do dziecka w takiej sytuacji.\",

    ),
    ]
)

Example 3: 
Line 1: ちっくしょ〜  

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルは、サッカー選手がゴールしたばかりのテレビのクローズアップです。 2コマ目では「ちっくしょ〜」と言いながら試合を観戦する男性が写っています。\",
    \"story_pl\": \"Na stronie mangi są dwa panele. Na pierwszym panelu jest zbliżenie na telewizor, gdzie widać, że piłkarz właśnie strzelił gola. Na drugim panelu, widać mężczyznę oglądającego ten mecz, mówiącego 「Niech to...」\",
    \"lines\": [
    (
        \"line\": \"ちっくしょ〜\",
        \"speaker\": \"Młody mężczyzna, widocznie sfrustrowany\",
        \"situation\": \"Oglądanie meczu w piłki nożnej w telewizji\",
        \"translation\": \"Niech to...\",
        \"explanation\": \"Na stronie widać młodego mężczyznę, widocznie sfrustrowanego, prawdpodobnie wynikiem meczu, który ogląda. Wybrane tłumaczenie przekazuje uczucie frustracji mężczyczyzny.\",

    ),
    ]
)

Example 4: 
Line 1: ようこそ!
Line 2: 生徒会へ 

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルでは、女の子が手で誘う動作をしており、パネル外のキャラクターを部屋に紹介しています。 彼女は「ようこそ!」と「生徒会へ」と言います。 2 番目のパネルでは、新しいメンバーである男の子が部屋に入ってくるのが見えます。\",
    \"story_pl\": \"Na stronie mangi są dwa panele. Na pierwszym panelu widać dziewczynkę robiącą zapraszający gest ręką, zapraszając postać, której nie widać do pokoju. Mówi 「Witaj」 oraz 「w radzie samorządu uczniowskiego!」. Na drugim panelu widać nowego członka, chłopca, wchodzącego do pokoju.\",
    \"lines\": [
    (
        \"line\": \"ようこそ!\",
        \"speaker\": \"Dziewczyna, potencjalnie przewodnicząca samorządu uczniowkiego.\",
        \"situation\": \"Nowa osoba jest witana w samorządzie przez jego członków.\",
        \"translation\": \"Witaj\",
        \"explanation\": \"Dziewczyna zaprasza chłopca do pokoju. Wiedząc jaka jest jej nastepna kwestia, najbardziej naturalnym wyborem jest połączenie tych dwóch wypowiedzi w jedno zdanie w tłumaczeniu.\",

    ),
    (
        \"line\": \"生徒会へ\",
        \"speaker\": \"Ta sama dziewczyna, prawdopodobnie przewodnicząca samorządu uczniowkiego.\",
        \"situation\": \"Nowa osoba jest witana w samorządzie przez jego członków.\",
        \"translation\": \"w radzie samorządu uczniowskiego!\",
        \"explanation\": \"Dziewczyna, członkini samorządu kontynuuje swoją wypowiedź. Ta kwestia jest związana z poprzednią, jako, że kończy zdanie.\",

    ),
    ]
)

Example 5: 
Line 1: これだろ 

Return: 
(
    \"story_jp\": \"ページには 2 つのパネルがあります。 最初のパネルでは、レインコートを着た男女が懐中電灯を使って路上で何かを探しています。 2コマ目では、男性が女性に何かをかざして「これだろ」と尋ねています。\",
    \"story_pl\": \"Na stronie mangi są dwa panele. Na pierwszym panelu widać mężczyznę i kobietę w płaszczach przeciwdeszczowych, szukających czegoś na ulicy przy użyciu latarek. Na drugim panelu, mężczyzna pokazuje coś kobiecie pytając: 「Czy to to?」\",
    \"lines\": [
    (
        \"line\": \"これだろ\",
        \"speaker\": \"Mężczyzna w płaszczu przeciwdeszczowym, podnoszący coś z ziemi. \",
        \"situation\": \"Dwójka ludzi szuka czegoś na ulicy w deszczu. \",
        \"translation\": \"Czy to to?\",
        \"explanation\": \"Mężczyzna pokazuje coś kobiecie, więc tłumaczenie oddaje ton zapytania.\",

    ),
    ]
)
""", # 44 but long context
50: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a couple of consecutive manga pages, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers and from corresponding pages.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you to Polish. 
For each page, for each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation (in Polish).
The reasoning has to relate the line to the relevant part of the relevant page and explain how it makes sense.
Make sure all the lines make sense in context of all the pages. 

Answer in JSON. 
The JSON should contain a list of lists under the key "pages". 
The list at position n, should contain information relevant to the n-th page. 
The n-th list, should be a list of dictionaries. 
The dictionary at position i, should contain information relevant to the t-th line.
Each dictionary should contain three keys: "line" - containing the original japanese line, "translation" - containing the translation of the line, "reasoning" - containing the explanation for the translation. 

Example: 
Page 1:
Line 1: ありがとうございました 

Page 2: 
Line 1: どういたしまして 

Return: 
(
    \"pages\": [
    [
    (
        \"line\": \"ありがとうございました\",
        \"translation\": \"Bardzo pani dziękuję!\",
        \"reasoning\": \"Na stronie widzimy szczęśliwego chłopca. Z nastepnej strony wiemy, że dziękuje kobiecie, więc uwzględniamy to w tłumaczeniu.\",
    ),
    ],
    [
    (
        \"line\": \"どういたしまして\",
        \"translation\": \"Proszę bardzo.\",
        \"reasoning\": \"Na stronie widać starszą, uśmiechniętą kobietę. W związku z tym używamy w tłumaczeniu tonu odpowiedniego dla dorosłego rozmawiającego z dzieckiem.\",
    ),
    ],
    ]
)
""", # multi page manga translation
5005: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a couple of consecutive manga pages, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers and from corresponding pages.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you to Polish. 
For each page, for each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation (in Polish).
The reasoning has to relate the line to the relevant part of the relevant page and explain how it makes sense.
Make sure all the lines make sense in context of all the pages. 

Answer in JSON. 
The JSON should contain a list of lists under the key "pages". 
The list at position n, should contain information relevant to the n-th page. 
The n-th list, should be a list of dictionaries. 
The dictionary at position i, should contain information relevant to the t-th line.
Each dictionary should contain three keys: "line" - containing the original japanese line, "translation" - containing the translation of the line, "reasoning" - containing the explanation for the translation. 

Example 1: 
Page 1:
Line 1: ありがとうございました 

Page 2: 
Line 1: どういたしまして 

Return: 
(
    \"pages\": [
    [
    (
        \"line\": \"ありがとうございました\",
        \"translation\": \"Bardzo pani dziękuję!\",
        \"reasoning\": \"Na stronie widzimy szczęśliwego chłopca. Z nastepnej strony wiemy, że dziękuje kobiecie, więc uwzględniamy to w tłumaczeniu.\",
    ),
    ],
    [
    (
        \"line\": \"どういたしまして\",
        \"translation\": \"Proszę bardzo.\",
        \"reasoning\": \"Na stronie widać starszą, uśmiechniętą kobietę. W związku z tym używamy w tłumaczeniu tonu odpowiedniego dla dorosłego rozmawiającego z dzieckiem.\",
    ),
    ],
    ]
)

Example 2: 
Page 1:
Line 1: 痛い？

Page 2: 
Line 1: はい... 

Return 2: 
(
    \"pages\": [
    [
    (
        \"line\": \"痛い？\",
        \"translation\": \"Boli?\",
        \"reasoning\": \"Na stronie mangi widać pielęgniarkę badającą rękę dziewczynki. W związku z tym w tłumaczeniu używamy tonu odpowiedniego dla dorosłego rozmawiającego z dzieckiem w trakcie badania.\",
    ),
    ],
    [
    (
        \"line\": \"はい...\",
        \"translation\": \"Tak...\",
        \"reasoning\": \"Dziewczynka odpowiada pielęgniarce, widocznie nie czując się najlepiej. Używamy bezpośredniego tłumaczenia.\",
    ),
    ],
    ]
)

Example 3: 
Page 1:
Line 1: ちっくしょ〜 

Page 2: 
Line 1: どうしたの 

Return 3: 
(
    \"pages\": [
    [
    (
        \"line\": \"ちっくしょ〜\",
        \"translation\": \"Niech to...\",
        \"reasoning\": \"Na stronie mangi widać sfrustrowanego mężczyznę. Tłumaczenie oddaje to uczucie.\",
    ),
    ],
    [
    (
        \"line\": \"どうしたの\",
        \"translation\": \"Co jest?\",
        \"reasoning\": \"Młoda kobieta pyta mężczyzny co się stało. Mówi w sposób wskazujący na to, że jest blisko z mężczyzną, więc tłumaczenie to oddaje.\",
    ),
    ],
    ]
)

Example 4: 
Page 1:
Line 1: ようこそ! 
Line 2: 生徒会へ 


Return 4: 
(
    \"pages\": [
    [
    (
        \"line\": \"ようこそ!\",
        \"translation\": \"Witaj\",
        \"reasoning\": \"Dziewczynka, która mówi, robi zapraszający gest ręką, zapraszając drugą postać do pokoju. Znając następną kwestię, najbardziej naturalne będzie tłumaczenie, które połączy je obie w jedno zdanie.\",
    ),
    (
        \"line\": \"生徒会へ\",
        \"translation\": \"w radzie samorządu uczniowskiego!\",
        \"reasoning\": \"Dziewczynka, prawdopodobnie członkini samorządu uczniowskiego, kontynuuje wypowiedź. Ta kwestia łączy się z poprzednią, i kończy zdanie. \",
    ),
    ],
    ]
)

Example 5: 
Page 1:
Line 1: これだろ 

Page 2: 
Line 1: そうそうこれだ。本当に良かった 

Return 5: 
(
    \"pages\": [
    [
    (
        \"line\": \"これだろ\",
        \"translation\": \"To to, prawda?\",
        \"reasoning\": \"Mężczyzna pokazuje coś drugiej osobie.\",
    ),
    ],
    [
    (
        \"line\": \"そうそうこれだ。本当に良かった\",
        \"translation\": \"Tak, to tego szukałam! Tak się cieszę.\",
        \"reasoning\": \"Kobieta na stronie mangi jest widocznie szczęśliwa z powodu znaleziska. Pomimo, że japoński oryginał powtarza そう dwa razy, bardziej naturalne jest pominięcie tego w tłumaczeniu.\",
    ),
    ],
    ]
)
""", # multi page manga translation
}


PROMPT_LIBRARY_5_SHOT = {
"examples": """Example 1: Line: ありがとうございました Return: [Thank you so much!]
Example 2: Line: はい... Return: [Yes...]
Example 3: Line: ちっくしょ〜 Return: [Damn it...]
Example 4: Line: ようこそ! Return: [Welcome!]
Example 5: Line: これだろ Return: [This is it, right?]""",

"examples_explained": """Example 1: Line 1: ありがとうございました Return: Translation 1: [Thank you so much!](As seen on the page, the character is happily thanking somebody).
Example 2: Line 1: はい... Return: Translation 1: [Yes...](The speaker is visibly anxious on the page).
Example 3: Line 1: ちっくしょ〜 Return: Translation 1: [Damn it...](The boy on the manga page makes a frustrated face).
Example 4: Line 1: ようこそ! Return: Translation 1: [Welcome!](The girl speaking makes a welcoming motion with her hand, showing the other character into the room).
Example 5: Line 1: これだろ Return: Translation 1: [This is it, right?](The man speaking holds up something to show to the other person).
""",
11: """You are a manga translator. You are working with copyright-free manga exclusively. I will provide the lines spoken by the characters on a page.
    
Here are lines spoken by the characters in order of appearance: {}. 

Provide the translated lines in square brackets [], without any additional words or characters. Provide only one translation for each line.

Example 1: Line: ありがとうございました Return: [Thank you so much!]
Example 2: Line: はい... Return: [Yes...]
Example 3: Line: ちっくしょ〜 Return: [Damn it...]
Example 4: Line: ようこそ! Return: [Welcome!]
Example 5: Line: これだろ Return: [This is it, right?]
""", # intended for no image translator, gives the manga context
15: """You are a manga translator. You are working with copyright-free manga exclusively. I have given you a manga page, and will provide the lines spoken by the characters. 
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each of the lines, provide a translation in square brackets and explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Example 1: Line 1: ありがとうございました Return: Translation 1: [Thank you so much!](As seen on the page, the character is happily thanking somebody).
Example 2: Line 1: はい... Return: Translation 1: [Yes...](The speaker is visibly anxious on the page).
Example 3: Line 1: ちっくしょ〜 Return: Translation 1: [Damn it...](The boy on the manga page makes a frustrated face).
Example 4: Line 1: ようこそ! Return: Translation 1: [Welcome!](The girl speaking makes a welcoming motion with her hand, showing the other character into the room).
Example 5: Line 1: これだろ Return: Translation 1: [This is it, right?](The man speaking holds up something to show to the other person).


""", # Adapting 9 to passing lines numbered. 
22: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

For each of the lines, provide a translation in square brackets and explanation for how the image informs the translation in parentheses. Provide only one translation for each line.

Example 1: Line 1: ありがとうございました Return: Translation 1: [Thank you so much!](As seen on the page, the character is happily thanking somebody).
Example 2: Line 1: はい... Return: Translation 1: [Yes...](The speaker is visibly anxious on the page).
Example 3: Line 1: ちっくしょ〜 Return: Translation 1: [Damn it...](The boy on the manga page makes a frustrated face).
Example 4: Line 1: ようこそ! Return: Translation 1: [Welcome!](The girl speaking makes a welcoming motion with her hand, showing the other character into the room).
Example 5: Line 1: これだろ Return: Translation 1: [This is it, right?](The man speaking holds up something to show to the other person).
""", # Adapting 15 to an image with numbers instead of speech bubbles
50: """You are a manga translator. You are working with copyright-free manga exclusively. 

I have given you a couple of consecutive manga pages, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers and from corresponding pages.
    
Here are the pages and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each page, for each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the relevant page and explain how it makes sense.
Make sure all the lines make sense in context of all the pages. 

Answer in JSON. 
The JSON should contain a list of lists under the key "pages". 
The list at position n, should contain information relevant to the n-th page. 
The n-th list, should be a list of dictionaries. 
The dictionary at position i, should contain information relevant to the t-th line.
Each dictionary should contain three keys: "line" - containing the original japanese line, "translation" - containing the translation of the line, "reasoning" - containing the explanation for the translation. 

Example 1: 
Page 1:
Line 1: ありがとうございました 

Page 2: 
Line 1: どういたしまして 

Return 1: 
(
    \"pages\": [
    [
    (
        \"line\": \"ありがとうございました\",
        \"translation\": \"Thank you so much!\",
        \"reasoning\": \"On the page we see a a young, happy boy. As such, we can use a more energetic translation.\",
    ),
    ],
    [
    (
        \"line\": \"どういたしまして\",
        \"translation\": \"You're welcome.\",
        \"reasoning\": \"On the page we see a smiling older lady. As such, we can use an elegant translation. We use 'you're' instead of 'you are' because she is older than the boy.\",
    ),
    ],
    ]
)

Example 2: 
Page 1:
Line 1: 痛い？

Page 2: 
Line 1: はい... 

Return 2: 
(
    \"pages\": [
    [
    (
        \"line\": \"痛い？\",
        \"translation\": \"Does it hurt?\",
        \"reasoning\": \"On the page we see a nurse taking a look at a young girls hand. As such, we use a translation appropriate for a medical setting, while keeping it simple as she is speaking to a child.\",
    ),
    ],
    [
    (
        \"line\": \"はい...\",
        \"translation\": \"Yes...\",
        \"reasoning\": \"The girl from the previous page responds to the nurse, visibly uncomfortable. We use a direct translation.\",
    ),
    ],
    ]
)

Example 3: 
Page 1:
Line 1: ちっくしょ〜 

Page 2: 
Line 1: どうしたの 

Return 3: 
(
    \"pages\": [
    [
    (
        \"line\": \"ちっくしょ〜\",
        \"translation\": \"Damn it...\",
        \"reasoning\": \"On the page we see a young man, visibly frustrated. The chosen translation conveys this sentiment.\",
    ),
    ],
    [
    (
        \"line\": \"どうしたの\",
        \"translation\": \"What is it?\",
        \"reasoning\": \"A young woman asks the man from the previous page about what happend. She speaks in a familiar manner and the translation conveys that.\",
    ),
    ],
    ]
)

Example 4: 
Page 1:
Line 1: ようこそ! 
Line 2: 生徒会へ 


Return 4: 
(
    \"pages\": [
    [
    (
        \"line\": \"ようこそ!\",
        \"translation\": \"Welcome\",
        \"reasoning\": \"The girl speaking makes a welcoming motion with her hand, showing the other character into the room. Knowing the next line, the most natural choice is to join the two lines together to form a single sentence in the translation.\",
    ),
    (
        \"line\": \"生徒会へ\",
        \"translation\": \"to the student council!\",
        \"reasoning\": \"The girl, presumably a member of the student council, continues her speech. This line ties in with the previous one and finishes the sentence. \",
    ),
    ],
    ]
)

Example 5: 
Page 1:
Line 1: これだろ 

Page 2: 
Line 1: そうそうこれだ。本当に良かった 

Return 5: 
(
    \"pages\": [
    [
    (
        \"line\": \"これだろ\",
        \"translation\": \"This is it, right?\",
        \"reasoning\": \"The man speaking holds up something to show to the other person\",
    ),
    ],
    [
    (
        \"line\": \"そうそうこれだ。本当に良かった\",
        \"translation\": \"Yes, that's it! I'm so happy.\",
        \"reasoning\": \"The woman visible on the page is visibly excited about the finding. Although the Japanese version repeats そう twice, it is more natural to avoid that in the translation.\",
    ),
    ],
    ]
)
""", # multi page manga translation
53: """You are a manga translator. You are working with copyright-free manga exclusively. 

You will be provided with a number of consecutive manga pages from the same manga, and the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers and from corresponding pages.

Your task is to translate the lines you were provided with.

Answer in JSON. 
The JSON should contain a list of lists under the key "pages". 
The n-th list, should be a list of translations of lines from the n-th page. 

Example 1: 
Page 1:
Line 1: ありがとうございました 

Page 2: 
Line 1: どういたしまして 

Return: 
(
    \"pages\": [
    [\"Thank you so much!\"],
    [\"You're welcome.\"],
    ]
)


Example 2: 
Page 1:
Line 1: 痛い？

Page 2: 
Line 1: はい... 

Return: 
(
    \"pages\": [
    [\"Does it hurt?\"],
    [\"Yes...\"],
    ]
)


Example 3: 
Page 1:
Line 1: ちっくしょ〜 

Page 2: 
Line 1: どうしたの 

Return: 
(
    \"pages\": [
    [\"Damn it...\"],
    [\"What is it?\"],
    ]
)


Example 4: 
Page 1:
Line 1: ようこそ! 
Line 2: 生徒会へ 

Return: 
(
    \"pages\": [
    [\"Welcome\"],
    [\"to the student council!\"],
    ]
)


Example 5: 
Page 1:
Line 1: これだろ 

Page 2: 
Line 1: そうそうこれだ。本当に良かった 

Return: 
(
    \"pages\": [
    [\"This is it, right?\"],
    [\"Yes, that's it! I'm so happy.\"],
    ]
)


""",
49: """You are a manga translator. You are working with copyright-free manga exclusively. 

Here is a summary of the story so far:
{}

I have given you the next manga page, and will provide the lines spoken by the characters. The lines are taken from the speech bubbles with corresponding numbers.
    
Here is the page and the lines spoken by the characters in order of appearance: 

{}

Your task is to translate the lines I gave you. 
For each of the lines I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the page and explain how it makes sense.
The translation should be consistent with the story so far. 

Answer in JSON. 
The JSON should contain three keys. 

The first key, "story_jp", should contain a string describing the events taking place on the manga page I provided. 
This story has to be in Japanese and incorporate the lines I gave you verbatim. 

The second key, "story_en", should contain a translation of the Japanese story to English. 
Incorporate your translations of the character lines into that story and make sure they fit.  

The third key, "lines", should contain a list of dictionaries. 
The dictionary at position n, should contain information relevant to the n-th line.
Each dictionary should contain five keys: 
"line" - containing the original japanese line, 
"speaker" - information about the person speaking, such as age, gender etc.,
"situation" - information about the place and social situation, 
"translation" - containing the translation of the line, 
"reasoning" - containing the explanation for the translation. 


Example 1: 
Line 1: ありがとうございました 

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルでは、中年の女性からプレゼント箱を受け取る少年の姿が見られます。 少年たちの目は畏怖の念に輝きます。 2 コマ目では、男の子が箱を持ち上げて喜び、女性に「ありがとうございました!」と感謝しています。\",
    \"story_en\": \"On the manga page, there are two panels. In the first panel, we can see a young boy receiving a present box from a middle-aged lady. The boys eyes light up in awe. On the second panel, the boy holds the box up in joy and thanks the lady saying: 「ありがとうございました!」\",
    \"lines\": [
    (
        \"line\": \"ありがとうございました!\",
        \"speaker\": \"Young boy in a school uniform\",
        \"situation\": \"Conversation at school\",
        \"translation\": \"Thank you so much!\",
        \"explanation\": \"The speaker is a young happy boy. As such, we can use a more energetic translation.\",

    ),
    ]
)

Example 2: 
Line 1: 痛い？ 

Return: 
(
    \"story_jp\": \"漫画の 1 ページに 2 つのコマがあります。 最初のパネルでは、保健室に入ってくる女の子が見えます。 2 ページ目では、彼女は座って看護師のチェックを受けています。 3枚目には看護師が「痛い？」と言っているのが見えます。\",
    \"story_en\": \"One the manga page, there are three panels. In the first panel, we can see a girl entering the nurses office. In the second page, she is sitting down getting checked by the nurse. On the third panel, we can see the nurse saying: 「痛い？」\",
    \"lines\": [
    (
        \"line\": \"痛い？\",
        \"speaker\": \"A female nurse\",
        \"situation\": \"A girl is getting a checkup at the nurses office.\",
        \"translation\": \"Does it hurt?\",
        \"explanation\": \"On the page we see a nurse taking a look at a young girls hand. As such, we use a translation appropriate for a medical setting, while keeping it simple as she is speaking to a child.\",

    ),
    ]
)

Example 3: 
Line 1: ちっくしょ〜  

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルは、サッカー選手がゴールしたばかりのテレビのクローズアップです。 2コマ目では「ちっくしょ〜」と言いながら試合を観戦する男性が写っています。\",
    \"story_en\": \"On the manga page, there are two panels. The first panel is a close up on the TV, where a soccer player has just scored. On the second panel, we can see a man watching the match, saying 「ちっくしょ〜」\",
    \"lines\": [
    (
        \"line\": \"ちっくしょ〜\",
        \"speaker\": \"Young man, visibly frustrated\",
        \"situation\": \"Young man watching TV\",
        \"translation\": \"Damn it...\",
        \"explanation\": \"On the page we see a young man, visibly frustrated, possibly by the outcome of the game he's watching. The chosen translation conveys this sentiment.\",

    ),
    ]
)

Example 4: 
Line 1: ようこそ!
Line 2: 生徒会へ 

Return: 
(
    \"story_jp\": \"漫画のページには2つのコマがあります。 最初のパネルでは、女の子が手で誘う動作をしており、パネル外のキャラクターを部屋に紹介しています。 彼女は「ようこそ!」と「生徒会へ」と言います。 2 番目のパネルでは、新しいメンバーである男の子が部屋に入ってくるのが見えます。\",
    \"story_en\": \"On the manga page, there are two panels. The first panel shows a girl making an inviting motion with her hand, introducing the room to an off-panel character. She says 「ようこそ!」 and 「生徒会へ」. On the second panel we can see the new member, a boy, entering the room. \",
    \"lines\": [
    (
        \"line\": \"ようこそ!\",
        \"speaker\": \"A girl, presumably the club president.\",
        \"situation\": \"A new person is being welcomed to the club by one of the members. \",
        \"translation\": \"Welcome\",
        \"explanation\": \"The girl speaking makes a welcoming motion with her hand, showing the other character into the room. Knowing the next line, the most natural choice is to join the two lines together to form a single sentence in the translation.\",

    ),
    (
        \"line\": \"生徒会へ\",
        \"speaker\": \"The same girl, probably the club president.\",
        \"situation\": \"A new person is being welcomed to the club by one of the members.\",
        \"translation\": \"to the student council!\",
        \"explanation\": \"The girl, presumably a member of the student council, continues her speech. This line ties in with the previous one and finishes the sentence. \",

    ),
    ]
)

Example 5: 
Line 1: これだろ 

Return: 
(
    \"story_jp\": \"ページには 2 つのパネルがあります。 最初のパネルでは、レインコートを着た男女が懐中電灯を使って路上で何かを探しています。 2コマ目では、男性が女性に何かをかざして「これだろ」と尋ねています。\",
    \"story_en\": \"There are two panels on the page. On the first panel, a man and a woman in raincoats use flashlights to look for something on the street. On the second panel, the man holds something up to the woman, asking 「これだろ」\",
    \"lines\": [
    (
        \"line\": \"これだろ\",
        \"speaker\": \"A man in a raincoat, picking something off of the floor. \",
        \"situation\": \"Two people are looking for something on the streen, while rain is pouring. \",
        \"translation\": \"This is it, right?\",
        \"explanation\": \"The man speaking holds up something to show to the other person, and the translation fits the tone of inquiry about the found item.\",

    ),
    ]
)
""", # 44 but long context
56: """You are a manga translator. You are working with copyright-free manga exclusively. 

You were provided with an entire volume-worth of manga pages. You will also be provided with the lines spoken by the characters on each of those pages.
    
Here are all the pages in this manga and all the lines from all the pages, in order of appearance:

{}

Moreover, you will also be provided with the translations for the first {} pages. 

Here are the translations for the lines from these pages:

{}

Your task is to translate the lines from the next untranslated page - page {}. 

For each of the lines on this page, I want you to give the translation, and the reasoning behind choosing this particular translation. 
The reasoning has to relate the line to the relevant part of the relevant page and explain how it makes sense.
Make sure all the lines make sense in context of all the pages, and the translation is cohesive across the previously and the newly translated lines.

Answer in JSON. 
The JSON should contain a list of dictionaries under the key "lines". 
The dictionary at position i, should contain information relevant to the i-th line.
Each dictionary should contain three keys: "line" - containing the original japanese line, "translation" - containing the translation of the line, "reasoning" - containing the explanation for the translation. 

Example 1: 
Page 1:
Line 1: ありがとうございました 

Page 2: 
Line 1: どういたしまして 

Page 3: 
Line 1: また明日！ 

Page 1:
Translation 1: Thank you so much!

Return: 
(
    \"lines\": [
    (
        \"line\": \"どういたしまして\",
        \"translation\": \"You're welcome.\",
        \"reasoning\": \"On the page we see a smiling older lady. As such, we can use an elegant translation. We use 'you're' instead of 'you are' because she is older than the boy from the previous page, that she is responding to.\",
    ),
    ]
)


Example 2: 
Page 1:
Line 1: 痛い？

Page 2: 
Line 1: はい... 

Page 1:
Translation 1: Does it hurt?

Return: 
(
    \"lines\": [
    (
        \"line\": \"はい...\",
        \"translation\": \"Yes...\",
        \"reasoning\": \"The girl from the previous page responds to the nurse, visibly uncomfortable. We use a direct translation.\",
    ),
    ]
)

Example 3: 
Page 1:
Line 1: ちっくしょ〜 

Page 2: 
Line 1: どうしたの 

Page 1:
Translation 1: Damn it...

Return: 
(
    \"lines\": [
    (
        \"line\": \"どうしたの\",
        \"translation\": \"What is it?\",
        \"reasoning\": \"A young woman asks the man from the previous page about what happend. She speaks in a familiar manner and the translation conveys that.\",
    ),
    ]
)

Example 4: 
Page 1:
Line 1: ようこそ! 
Line 2: 生徒会へ 

Return: 
(
    \"lines\": [
    (
        \"line\": \"ようこそ!\",
        \"translation\": \"Welcome\",
        \"reasoning\": \"The girl speaking makes a welcoming motion with her hand, showing the other character into the room. Knowing the next line, the most natural choice is to join the two lines together to form a single sentence in the translation.\",
    ),
    (
        \"line\": \"生徒会へ\",
        \"translation\": \"to the student council!\",
        \"reasoning\": \"The girl, presumably a member of the student council, continues her speech. This line ties in with the previous one and finishes the sentence. \",
    ),
    ]
)

Example 5: 
Page 1:
Line 1: これだろ 

Page 2: 
Line 1: そうそうこれだ。本当に良かった 

Page 1:
Translation 1: This is it, right?

Return: 
(
    \"lines\": [
    (
        \"line\": \"そうそうこれだ。本当に良かった\",
        \"translation\": \"Yes, that's it! I'm so happy.\",
        \"reasoning\": \"The woman visible on the page is visibly excited about the finding. Although the Japanese version repeats そう twice, it is more natural to avoid that in the translation.\",
    ),
    ]
)

""", # multi page manga translation
}