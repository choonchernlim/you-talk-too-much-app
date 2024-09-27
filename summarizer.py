import logging
import os

import vertexai
from dotenv import load_dotenv
from google.oauth2 import service_account
from markdown import markdown
from vertexai.generative_models import GenerativeModel, Part

from utils import read_file, write_file

logger = logging.getLogger(__name__)

load_dotenv()

GCP_VERTEX_PROJECT = os.getenv('GCP_VERTEX_PROJECT')
GCP_VERTEX_LOCATION = os.getenv('GCP_VERTEX_LOCATION')
GCP_VERTEX_SA_KEY = os.getenv('GCP_VERTEX_SA_KEY')
GCP_VERTEX_MODEL = os.getenv('GCP_VERTEX_MODEL')


class Summarizer:
    def __init__(self) -> None:
        vertexai.init(
            project=GCP_VERTEX_PROJECT,
            location=GCP_VERTEX_LOCATION,
            credentials=service_account.Credentials.from_service_account_file(GCP_VERTEX_SA_KEY),
        )

        self.model = GenerativeModel(GCP_VERTEX_MODEL)

    def summarize(self, conversation_file_path) -> str:
        logger.info(f'Summarizing conversation in {conversation_file_path}...')

        prompt = '''
        You are a helpful agent in summarizing lengthy conversations. 
        The summarization must be grounded to the provided document.
        Use the markdown format below when responding.
        Follow the order of the sections and provide the details as requested.\
        If you are unsure about the content, please do not make up any information.
        
        For each section, provide the following details:
        1. "Executive Summary" section: 3 to 5 bullet points summarizing the key details and decisions.
        2. "Detailed Summary" section: Details and key decisions in bullet points, grouped by topics.
        3. "Action Items" section: 3 to 5 bullet points under "Executive Summary" section.

        [MARKDOWN FORMAT]:        
        
        # Executive Summary
        
        * [SUMMARY]
        * [SUMMARY]

        # Detailed Summary

        ## [TOPIC]
    
        * **[SUBTOPIC]**
          * [DETAIL]
          * [DETAIL]

        # Action Items
        
        * [ACTION ITEM]
        * [ACTION ITEM]
        '''

        doc = Part.from_text(read_file(conversation_file_path))

        response = self.model.generate_content(
            [prompt, doc],
            generation_config={
                'max_output_tokens': 8192,
                'temperature': 1,
                'top_p': 0.95,
            },
        )

        html_content = markdown(response.text)

        write_file(f'{conversation_file_path}.md', response.text)
        write_file(f'{conversation_file_path}.html', html_content)

        return html_content
