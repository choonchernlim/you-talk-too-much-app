import vertexai
from google.oauth2 import service_account
from markdown import markdown
from vertexai.generative_models import GenerativeModel, Part

from you_talk_too_much.log_config import setup_logger
from you_talk_too_much.utils import read_file, write_file

logger = setup_logger(__name__)


class LLM:
    def __init__(self, project, location, sa_key, model) -> None:
        logger.info('Initializing Vertex LLM...')

        vertexai.init(
            project=project,
            location=location,
            credentials=service_account.Credentials.from_service_account_file(sa_key),
        )

        self.model = GenerativeModel(model)

    def summarize(self, conversation_file_path) -> str:
        logger.info(f'Summarizing conversation [File: {conversation_file_path}] ...')

        prompt = '''
        You are a helpful agent, capable in summarizing lengthy conversations clearly and succinctly. 

        <INSTRUCTIONS>        
        1. The summarization must be grounded to the provided document.
        2. Use the exact markdown format below when responding.
        3. Follow the order of the sections and provide the details as requested.
        4. If you are unsure about the content, please do not make up any information.
        </INSTRUCTIONS>        
        
        <SECTION STRUCTURE>        
        1. "Executive Summary" section: 3 to 5 bullet points summarizing the key details and decisions.
        2. "Detailed Summary" section: Details and key decisions in bullet points, grouped by topics.
        3. "Action Items" section: 3 to 5 bullet points on actions to be taken. Provide timeline if available.
        </SECTION STRUCTURE>        

        <MARKDOWN FORMAT>        
        # Executive Summary
        
        * [TEXT]
        * [TEXT]

        # Detailed Summary

        ## [TOPIC]
    
        * **[SHORT SUMMARY]:** [TEXT]
        * **[SHORT SUMMARY]:** [TEXT]

        # Action Items
        
        * [TEXT]
        * [TEXT]
        </MARKDOWN FORMAT>        

        <MARKDOWN RULES>
        1. [SUBTOPIC] must be in bold.
        2. [TEXT] must NOT in bold.        
        </MARKDOWN RULES>
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

        # remove file extension from conversation_file_path, ex: filename.txt -> filename
        filename = conversation_file_path.rsplit('.', 1)[0]

        write_file(f'{filename}.md', response.text)
        write_file(f'{filename}.html', html_content)

        return html_content
