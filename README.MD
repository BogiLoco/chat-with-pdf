## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This is simple application which provide possibility to load document, divide content into chunks and embed it. 
This process let to communicate through model and talk with bot about the pdfs content.r.
	
## Technologies
Project is created with:
* python 3.9

	
## Setup
To run this project, install it locally:

create .env in the main directory file with API Key to HuggingFace

```
create virtual environment: 
python -m venv env

run virtual environment:
source env/bin/activate

install dependencies:
pip install -r requirements.txt

run application:
streamlit run app.py
```