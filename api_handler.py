from typing import Tuple, List, Optional
import base64
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_exponential,
    retry_if_exception_type
)
import openai
import logging

class API_handler: 
    
    def __init__(self, api_key, gpt_model = "gpt-4-turbo", detail_level = "low"):
        self.client = openai.OpenAI(api_key = api_key, max_retries=0)
        self.gpt_model = gpt_model
        self.detail_level = detail_level

    @classmethod
    def encode_image(cls, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    @classmethod
    def parse_response(cls, response) -> Tuple[str, int, str]:
        # logging.getLogger("Experiment").info(f"API RESPONSE WITH {len(response.choices)} CHOICES") 
        # logging.getLogger("Experiment").debug(f"API RESPONSE FULL {response}") 

        response_text = ""
        for choice in response.choices:
            response_text += choice.message.content
        return response_text

    # Had to change the erros, probably library incompatibility
    @retry(
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError, openai.Timeout)), 
        wait=wait_exponential(multiplier=2, max=300),
        stop=stop_after_attempt(10)
    )
    def send_request(self, system_message: str = "You are a helpful assistant.", query_text : str = "What's in this image?", temperature: Optional[float] = None, urls : List[str] = [], json_response: bool = False):

        messages = [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": f"{system_message}"
                }
            ],
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{query_text}"
                },
            ],
            }
        ]

        for url in urls:
            messages[1]["content"].append(
                {
                "type": "image_url",
                "image_url": {
                    "url": f"{url}",
                    "detail": f"{self.detail_level}"
                }
                })
        if json_response:
            response = self.client.chat.completions.create(messages=messages, model = self.gpt_model, temperature=temperature, response_format = { "type": "json_object" })
        else:
            response = self.client.chat.completions.create(messages=messages, model = self.gpt_model, temperature=temperature)

        
        return response

    def query_with_local_image(self, system_message: str, query_text: str, temperature: Optional[float] = None, image_paths: List[str] = [], json_response: bool = False): 
        urls = []
        for image_path in image_paths:
            base64_image = self.encode_image(image_path)
            url = f"data:image/jpeg;base64,{base64_image}"
            urls.append(url)
        return self.send_request(system_message = system_message, query_text = query_text, temperature = temperature, urls = urls, json_response=json_response)

    def query_local_parse(self, system_message: str, query_text: str, temperature: Optional[float] = None, image_paths: List[str] = [], json_response: bool = False):
        response = self.query_with_local_image(system_message, query_text, temperature, image_paths, json_response=json_response)
        return self.parse_response(response)