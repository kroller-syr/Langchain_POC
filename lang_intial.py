import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Need to add .env file with OPENAI_API_KEY
#added
#load the API key from the .env file
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)   

#Intialize the OpenAI LLM
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

#Prompt 1: Extract the information
prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text: \n\n{text_input}"
) 


#Prompt 2: Transform to JSON
prompt_transform = ChatPromptTemplate.from_template(
    """Transform the following specifications into a JSON object with
    'cpu', 'memory', 'storage' as keys: \n\n{specifications}"""
)

#Build the Chain using LCEL
#The StrOutputParser converts the LLMs meesage to a simple string
extraction_chain = prompt_extract | llm | StrOutputParser()

#The full chain passes the output of the extraction chain into the 'specificaitons'
#variable for the transformation prompt
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()

)

#Run the chain with some input text
input_text = """The new laptop model features a 3.5 GHz octa-core
processor, 16GB of RAM, and a 1TB NVMe SSD."""

#Execute the chain with the input text dictionary
final_result = full_chain.invoke({"text_input": input_text})

print("\n--- Final JSON Output ---")
print(final_result)




