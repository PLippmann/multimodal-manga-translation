import openai
from openai import OpenAI
import json
import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)


class LongFormCompression:
    def __init__(self, api_key: str, gpt_model: str = "gpt-4-turbo", max_tokens: int = 1024, lang: str = "English"):
        self.model = gpt_model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)  
        self.lang = lang

    @retry(
    retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError, openai.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10))
    
    def llm_query(self, system_message: str = "You are a helpful assistant.", prompt: str = "Return YEET.", print_output: bool = False):
        """
        Generate the response from the GPT model.
        :param system_message: the system message for the model
        :param prompt: the prompt for the model
        :param print_output: whether to print the output
        :return: the response from the model
        """
        # API reference: https://platform.openai.com/docs/api-reference/chat/create
        response = self.client.chat.completions.create(model=self.model,                           # GPT model
        max_tokens=self.max_tokens,                 # Maximum tokens in the prompt AND response
        response_format={ "type": "json_object" },  # Return the response as a JSON object
        messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
        ])

        if response.choices[0].finish_reason != "stop":
            print(f"The response was not properly generated. The reason for the early stop was: {response.choices[0].finish_reason}.")
            return None
            
        if print_output:
            print(response)

        # Return the response
        return response.choices[0].message.content

    def managed_memory(self, observation: str = None, prev_context: str = None, lmax: int = 200):
        """
        Implementation of long context managed memory using an LLM. 
        Here, the context is iteratively compressed to a fixed size, following Chain of Density summarization.
        This compressed context is later used to make a more informed translation.

        :param observation: the most recent observation from the translation
        :param prev_context: the previous context
        :param lmax: the maximum length of the context
        :return: the compressed context in plain text
        """
        if prev_context is None:  # Avoid using prev_context on first call when it's not available and replace with objective.
            prev_context = "There is no previous context. This is the first translation."

        system_message = f"""Help me keep track of all relevant details while I perform a translation by creating a new summary using an existing one and new information. 
            The translation is a Japanese to {self.lang} translation and I want you to help me remember everything important about the story."""
        
        prompt_old = f"""Existing Summary from the previous Translation: {prev_context}

            The most recent Observation from the Japanese text was: {observation}

            You will generate new increasingly concise, entity-dense summaries based on the above Existing Summary and most recent Observation.

            Perform the following 2 steps 3 times.

            Step 1. If possible, identify 1-3 Informative Entities (";" delimited) from the most recent Observation which are missing from the Existing Summary.
            Step 2. Write a new, denser summary of identical length which covers every entity, action, and detail from the previous Existing Summary plus the Informative Entities from the Observation.

            An Informative Entity is:
            - Relevant: to the translation's unfolding narrative.
            - Specific: descriptive yet concise (5 words or fewer).
            - Novel: not in the previous summary.
            - Faithful: an accurate, detailed reflection of the translation.
            - Anywhere: can be derived from any part of the translation.

            Guidelines:
            - The first of the three summaries must be long (but less than ~{lmax} words) yet highly non-specific, containing little information beyond the entities marked as missing. Use verbose language and fillers (e.g., "In this part of the translation, the main character encounters ...") to reach ~{lmax} words.
            - Make every word count: rewrite the previous summary to improve flow and make space for additional Informative Entities.
            - Make space with fusion, compression, and removal of uninformative phrases like "the scenario presents".
            - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without referencing the fact that a translation is being performed, and contain all information of the narrative thus far.
            - Informative Entities can appear anywhere in the new summary.
            - Only drop the least relevant Informative Entities from the previous summary if the summary length exceeds ~{lmax} words. Otherwise carry all previous Informative Entities to the new summary.

            Answer in JSON. The JSON should be a list (length 3) of dictionaries whose keys are "Informative_Entities" and "Denser_Summary". Each of these pairs should have a key from 1 to 3.
            """
        
        mid_prompt = f"""Here is a summary of the story so far: {prev_context}

            Here is the most recent part of the story: {observation}

            You will generate a new summary that contains information from the story so far and the most recent part.

            Guidelines:
            - The summary must be shorter than ~{lmax} words. 
            - Make every word count
            - Make space with fusion, compression, and removal of uninformative phrases like "the scenario presents".
            - The summary should be highly dense and concise yet self-contained, e.g., easily understood without referencing the fact that a translation is being performed, and contain all information of the narrative thus far.
            - Only drop the least relevant information from the previous summary if the summary length exceeds ~{lmax} words. Otherwise carry all previous information to the new summary.

            Answer in JSON. The JSON should be a dictionaries with key "Denser_Summary" containing the new summary.
            """
        
        prompt = f"""Existing Summary from the previous Translation: {prev_context}

            The most recent Observation from the {self.lang} translation was: {observation}

            You will generate new increasingly concise, entity-dense summaries based on the above Existing Summary and most recent Observation.

            Keep the summaries in {self.lang}.

            You will create 3 summaries. You will create each of them by following the following two setps:

            - Step 1. If possible, identify 1-3 Informative Entities (";" delimited) from the most recent Observation which are missing from the Existing Summary.
            - Step 2. Write a new, denser summary of identical length which covers every entity, action, and detail from the previous Existing Summary plus the Informative Entities from the Observation.

            An Informative Entity is:
            - Relevant: to the translation's unfolding narrative.
            - Specific: descriptive yet concise (10 words or fewer).
            - Novel: not in the previous summary.
            - Faithful: an accurate, detailed reflection of the translation.

            Guidelines:
            - The first of the three summaries must be long (but less than ~{lmax} words) yet highly non-specific, containing little information beyond the entities marked as missing. Use verbose language and fillers (e.g., "In this part of the translation, the main character encounters ...") to reach ~{lmax} words.
            - Make every word count: rewrite the previous summary to improve flow and make space for additional Informative Entities.
            - Make space with fusion, compression, and removal of uninformative phrases like "the scenario presents".
            - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without referencing the fact that a translation is being performed, and contain all information of the narrative thus far.
            - Informative Entities can appear anywhere in the new summary.
            - Only drop the least relevant Informative Entities from the previous summary if the summary length exceeds ~{lmax} words. Otherwise carry all previous Informative Entities to the new summary.

            Answer in JSON. The JSON should be a list (length 3) of dictionaries under the key "summaries". Each dictionary should contain keys "Informative_Entities" (storing the Informative Entities included in the corresponding summary) and "Denser_Summary" (containing the summary).
            """

        try:
            response = self.llm_query(system_message, prompt)
            logging.getLogger("Experiment").debug(f"COMPRESSION API RESPONSE: \n{response}")
            pprocess = json.loads(response) # Process the JSON response to extract the summary and return in plain text.
            try:
                key = pprocess.keys()[0]
            except:
                key = 'summaries'
            return pprocess[key][2]['Denser_Summary']
        except Exception as e:
            print(f"An error occurred: {e}")
            # Should there be an error, return the previous context and the most recent observation instead.
            print("WARNING: The LLM did not return a valid response. The response will be set to the prev context plus most recent observation.")
            return prev_context + " " + observation
        