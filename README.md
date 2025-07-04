# KDT Bon Project Team #2 backend application
## Introduction
This is a part of a bigger project; We're developing an experimental service that helps users get ready for the navigator examination. In the service, we'll provide answers and explanations for the past exam questions, and a useful chatbot for further education. In particular, this Python application serves as the backend server among the frontend, the local AI model, and external LLM APIs.
- Python
    - FastAPI
- MySQL
    - Local Sever

## Installation
```
python -m venv .venv
.venv/scripts/activate
(.venv) pip install -r requirements.txt
```

## Run the code
1. Import the latest MySQL dump in the MySQL Workbench
    ```
    1. [Server]
    2. [Data Import]
    3. Import from Self-Contained File
    4. Insert some data into your desired schema so that you can test
    ```

2. Create .env file and enter
    > DATABASE_URL=mysql+pymysql://\<yourId>:\<yourPw>@localhost:3306/\<schemaName>

3. Run the server with:
    ```
    fastapi dev app/apitest.py
    ```

## Progress
- [x] Create the backend project
- [x] Set the basic DB structure in MySQL
- [x] Set the equivalent SQLModel table model classes for FastAPI
- [ ] Implement required CRUD functions for API returns
- [ ] Implement required API endpoints
- [ ] Debug / Fix / Additional features
- [ ] Build

## Credits
[@Ohyeon Kwon](https://github.com/ohyeon1002)

## See Also
Frontend Project by [@rsh2231](https://github.com/rsh2231/MarinAI)\
vLLM Project by [@DevLibrary04](https://github.com/DevLibrary04/marine_officer_test)