import gradio as gr
import onnxruntime_genai as og

# Set Path
PATH='.\\cpu-int4-rtn-block-32-acc-level-4'
# Load Model and Tokenizer
model = og.Model(PATH)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

## Setting generation parameters
search_options = {}
search_options['do_sample']=False
search_options['max_length']=2048
# search_options['temperature']=0
# search_options['top_p']=0.9
# search_options['top_k']=1
# search_options['repetition_penalty']=0

chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

def history_to_dialog_format(chat_history):
    # Convert text to dialogue format
    dialog=""
    for user,assistant in chat_history:
        dialog+=f'{chat_template.format(input=user)}'+assistant+"<|end|>\n"
    return dialog

def response(text,history):  
    # Add input message to dialogue using history from gradio  
    dialog=history_to_dialog_format(history)
    # Generate Response

    prompt = dialog+ f'{chat_template.format(input=text)}'
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)
    out_sentence=""

    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        word_out=tokenizer_stream.decode(new_token)
        out_sentence+=word_out
        print(word_out,end="")
    # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
    del generator
    return out_sentence


demo = gr.ChatInterface(
    response, 
    title="Phi3 Chatbot",
)

demo.launch()
