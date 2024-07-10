import gradio as gr
# Load Model and Tokenizer
def history_to_dialog_format(chat_history):#: list[str]):
    # Convert text to dialogue format
    return 

def response(message,history):  
    # Add input message to dialogue using history from gradio  
    ## messages=history_to_dialog_format(history)
    # Generate Response
    return


demo = gr.ChatInterface(
    response, 
    title="Skeleton Chatbot",
)

demo.launch()
