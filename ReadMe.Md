This project is a conversational chatbot, it allows you to chat with the provided pdf file using the cpu since not everyone can afford a good gpu machine. 
To run this project you need to follow these steps: 
1-Download the llama2 model from this link :
    7b model : https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin
    13b model : https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/blob/main/llama-2-13b-chat.ggmlv3.q8_0.bin
    70b model : https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML/blob/main/llama-2-70b-chat.ggmlv3.q8_0.zip
    (Note that the model should be aligned with your pc specs, 7b requires atleast 16Gb of RAM)
2-Put your pdf files in the data directory
3-Open a terminal windows (or powershell if you are using a windows machine) and change to the project directory
4-Run these commands in the fore-mentionned terminal :
    python -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt
5-Run this command to create the vector database:
    python ingest.py
6-Open the model.py file and change the model name to the one you are using
7-Run this command to start the bot:
    chainlit run model.py -w

""Enjoy your own private chatbot""