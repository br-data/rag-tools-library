import logging
import os
from enum import Enum
import tiktoken
from typing import List, Dict
from openai import OpenAI
import requests


class LLMName(Enum):
    GPT35TURBO = "gpt-3.5-turbo"
    GPT35TURBO0613 = "gpt-3.5-turbo-0613"
    GPT35TURBO1106 = "gpt-3.5-turbo-1106"
    GPT40314 = "gpt-4-0314"
    GPT40613 = "gpt-4-0613"
    GPT4 = "gpt-4"
    IGEL = "igel"
    BISON001 = "text-bison@001"

    @property
    def family(self) -> str:
        if self in [LLMName.GPT35TURBO0613, LLMName.GPT35TURBO, LLMName.GPT4, LLMName.GPT40314, LLMName.GPT40613, LLMName.GPT35TURBO1106]:
            return "GPT"
        elif self in [LLMName.IGEL]:
            return "IGEL"
        elif self in [LLMName.BISON001]:
            return "BARD"
        else:
            logging.warning(f"Unknown model family for LLM {self}. Opt for default context window size of 2048.")
            return ""

    @property
    def max_input_tokens(self) -> int:
        if self in [LLMName.GPT35TURBO0613, LLMName.GPT35TURBO]:
            return 4096
        elif self in [LLMName.GPT4, LLMName.GPT40314, LLMName.GPT40613, LLMName.IGEL, LLMName.BISON001]:
            # context window size size IGEL: https://github.com/bigscience-workshop/petals/issues/146
            return 8192
        elif self == LLMName.GPT35TURBO1106:
            return 16385
        else:
            logging.warning(f"Unknown context window size for LLM {self}. Opt for default context window size of 2048.")
            return 2048


class Generator:
    """
    :class: Generator

    This class represents a generator for text generation using language models.

    :param model: The language model to be used for text generation.
    :type model: LLMName
    :param token: The API token to access the language model. If not provided, the token will be fetched based on the model value.
    :type token: str, optional
    :param temperature: The temperature parameter for text generation. A higher value (e.g., 1.0) makes the output more random, while a lower value (e.g., 0.2) makes it more focused and deterministic. If not provided, the default model's temperature will be used.
    :type temperature: float, optional
    :param max_new_tokens: The maximum number of new tokens to generate. If not provided, the default model's maximum new tokens value will be used.
    :type max_new_tokens: int, optional
    :param top_p: The top-p probability threshold for text generation. Only tokens with cumulative probability less than or equal to the threshold will be considered. If not provided, the default model's top-p value will be used.
    :type top_p: float, optional
    :param top_k: The top-k number of tokens to consider for text generation. Only the k most probable tokens will be considered. If not provided, the default model's top-k value will be used.
    :type top_k: int, optional
    :param length_penalty: The length penalty factor. It determines how much influence the length of the generated text has on the probability distribution. A higher value (e.g., 0.8) encourages generating shorter text, while a lower value (e.g., 1.2) encourages longer text. If not provided, the default model's length penalty value will be used.
    :type length_penalty: float, optional
    :param number_of_responses: The number of responses to generate. If set, the generator will return a list of responses instead of a single response. If not provided, the generator will return a single response.
    :type number_of_responses: int, optional

    :ivar model: The language model to be used for text generation.
    :vartype model: LLMName
    :ivar auth_token: The API token to access the language model.
    :vartype token: str
    :ivar temperature: The temperature parameter for text generation.
    :vartype temperature: float
    :ivar max_new_tokens: The maximum number of new tokens to generate.
    :vartype max_new_tokens: int
    :ivar top_p: The top-p probability threshold for text generation.
    :vartype top_p: float
    :ivar top_k: The top-k number of tokens to consider for text generation.
    :vartype top_k: int
    :ivar length_penalty: The length penalty factor.
    :vartype length_penalty: float
    :ivar number_of_responses: The number of responses to generate.
    :vartype number_of_responses: int

    """

    def __init__(self,
                 model: LLMName,
                 auth_token: str = None,
                 temperature: float = None,
                 max_new_tokens: int = None,
                 top_p: float = None,
                 top_k: int = None,
                 length_penalty: float = None,
                 number_of_responses: int = None,
                 max_token_length: int = None
                 ):

        self.model: LLMName = model
        self.auth_token: str = self.get_token(auth_token)
        self.temperature: float = temperature
        self.max_new_tokens: int = max_new_tokens
        self.top_p: float = top_p
        self.top_k: int = top_k
        self.length_penalty: float = length_penalty
        self.number_of_responses = number_of_responses
        self.max_token_length: int = max_token_length
        self.history = ChatHistory(self.model)

    def get_token(self, token: str) -> str:
        """
        Returns the given auth_token or retrieves the appropriate auth_token based on the model value.

        :param token: The auth_token to be used for authentication.
        :type token: str
        :return: The retrieved auth_token or the given auth_token.
        :rtype: str
        :raises ValueError: If no auth_token is provided for the model value.
        """
        if token is not None:
            return token
        else:
            if self.model.value.startswith("gpt"):
                token = os.environ.get("OPENAI_TOKEN")
            elif self.model.value == "igel":
                token = os.environ.get("IGEL_TOKEN")
            if self.model.value.startswith("text-bison"):
                token = os.environ.get("GOOGLE_TOKEN")

        if token is None:
            raise ValueError(f"No auth_token provided for model {self.model.value}.")
        else:
            return token

    def _estimate_tokens_openai(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model.value)
        tokens = encoding.encode(text)
        return len(tokens)

    def _estimate_tokens_rough(self, text: str) -> int:
        return int(len(text.split(" ")) * .7)

    def estimate_tokens(self, text: str):
        if self.model.value.startswith("gpt"):
            return self._estimate_tokens_openai(text)
        else:
            return self._estimate_tokens_rough(text)

    def fit_to_context_window(self, prompt: str, context: List[str]) -> List[str]:
        """
        Reduces a list of semantic search results to fit the context window of the given LLM.

        Token lengths are estimated and may differ from the real token vector's length.

        Guesses for OpenAI models are more accurate than for other models.

        :param prompt: Your prompt for the LLM
        :param context: The context retrieved by the semantic search.
        :return: The reduced context
        :raises ValueError: If the prompt is too long to fit any context
        """

        context_window_fit = False

        while not context_window_fit:
            temp = prompt + " " + " ".join(context)

            if self.estimate_tokens(temp) <= self.model.max_input_tokens:
                context_window_fit = True
            else:
                try:
                    context.pop(-1)
                except IndexError:
                    raise ValueError(f"Prompt is too long to add context from semantic search. Please try to reduce "
                                     f"the length of your prompt to fit it in the context window size of {self.max_token_length}."
                                     f"Your current prompt has an estimated length of {self.estimate_tokens(prompt)}.")

        return context

    def prompt(self, prompt: str) -> str:
        """
        Prompts the user with a prompt string and returns the user's input.

        :param prompt: The prompt string to display to the user.
        :type prompt: str
        :return: The user's input as a string.
        :rtype: str
        """
        raise NotImplementedError("Please implement the prompt method.")

    def chat(self, prompt: str) -> str:
        raise NotImplementedError("Please implement the chat method.")


class OpenAi(Generator):
    """
    Create text from OpenAi.

    :param model: The name of the language model to use.
    :type model: LLMName
    :param token: The API auth_token for accessing the OpenAi API.
    :type token: str
    """

    def __init__(self, model: LLMName, auth_token: str):
        super().__init__(
            model=model,
            auth_token=auth_token,
            temperature=1.0,
            max_new_tokens=256,
            top_p=0.9,
            number_of_responses=1)

        self.client = OpenAI(api_key=self.auth_token)

    def prompt(self, prompt: str) -> str:
        """Generate text with GPT model family."""

        gen_response = self.client.chat.completions.create(
            model=self.model.value,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            n=self.number_of_responses,
        )

        return gen_response.choices[0].message.content

    def chat(self, prompt: str) -> str:
        self.history.add(Role.USER, prompt)

        gen_response = self.client.chat.completions.create(
            model=self.model.value,
            messages=self.history.get_content(),
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            n=self.number_of_responses,
        )
        
        content = gen_response.choices[0].message.content
        self.history.add(Role.SYSTEM, content)
        return content



class IGEL(Generator):
    """
    A class representing the IGEL generator.

    The IGEL generator is a type of generator that uses the LLMName.IGEL model to generate text.
    It accepts a token for authentication and various parameters that control the generation process.

    Args:
        token (str, optional): A token for authentication. Defaults to None.

    Attributes:
        model (LLMName): The model used by the IGEL generator.
        auth_token (str): The token used for authentication.
        temperature (float): The temperature parameter for generation.
        max_new_tokens (int): The maximum number of new tokens to generate.
        top_p (float): The top-p parameter for generation.
        length_penalty (float): The length penalty parameter for generation.

    Methods:
        prompt(prompt: str) -> str: Generates text based on the provided prompt.

    """

    def __init__(self, token=None):
        super().__init__(
            model=LLMName.IGEL,
            temperature=1.0,
            max_new_tokens=256,
            top_p=0.9,
            length_penalty=1.0)

    def prompt(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }

        data = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "length_penalty": self.length_penalty,
        }

        response = requests.post(
            "https://modelhub-gpu.brdata-dev.de/v1/prompt", headers=headers,
            json=data
        )

        response.raise_for_status()
        return response.json()["response"]

    def chat(self, prompt: str) -> str:
        raise NotImplementedError("The IGEL Language Model does not support chat messages. Please use a model from the "
                                  "GPT or BARD Family instead.")


class GeneratorFactory:
    """

    The `GeneratorFactory` class is responsible for creating instances of different generator classes based on the provided model name.

    """

    def __init__(self, model_name: LLMName, token: str = None):
        self.model_name = model_name
        self.token = token

    def select_generator(self):
        if self.model_name in [LLMName.GPT4, LLMName.GPT40314, LLMName.GPT40613,
                               LLMName.GPT35TURBO, LLMName.GPT35TURBO0613]:
            return OpenAi(self.model_name, self.token)
        elif self.model_name == LLMName.IGEL:
            return IGEL(self.token)
        else:
            raise ValueError(f"{self.model_name.value} is not supported yet.")


class LLM:
    """
    Class representing a Language Model.

    Args:
        model_name (LLMName): The name of the language model.
        token (str, optional): The API auth_token for the language model. Defaults to None.

    Attributes:
        model_name (LLMName): The name of the language model.
        token (str): The API auth_token for the language model.
        model (Generator): The language model generator.

    Methods:
        prompt: Generate text based on the given prompt.

    """

    def __init__(self, model_name: LLMName, token: str = None):
        self.model_name: LLMName = model_name
        self.token = token
        self.model = GeneratorFactory(self.model_name).select_generator()

    def prompt(self, prompt: str) -> str:
        """
        Prompt the model with a given prompt and return the generated output.

        :param prompt: The prompt string to provide to the model.
        :type prompt: str
        :return: The generated output from the model in response to the given prompt.
        :rtype: str
        """
        return self.model.prompt(prompt)
    
    def chat(self, prompt: str) -> str:
        return self.model.chat(prompt)

    def new_chat(self):
        self.model.history.reset()
    


class Role(Enum):
    USER = "user"
    SYSTEM = "system"


class ChatHistory:
    def __init__(self, model_name: LLMName):
        self._history: List[Dict] = []
        self.model_name = model_name

    def add(self, role: Role, message: str):
        self._history.append({"role": role.value, "content": message})
    
    def reset(self):
        self._history = []

    def get_content(self):
        if self.model_name.family == "GPT":
            return self._history
        elif self.model_name.family == "IGEL":
            raise NotImplementedError("IGEL is not implemented yet")
        elif self.model_name.family == "BARD":
            raise NotImplementedError("BARD is not implemented yet")
        elif self.model_name.family == "":
            raise ValueError("Unknown Model.")
