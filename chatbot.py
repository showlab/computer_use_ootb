import gradio as gr

history = [
    {"role": "assistant", "content": "I am happy to provide you that report and plot."},
    {"role": "user", "content": "hihi"},
    {"role": "assistant", "content": "hihi"}
]

with gr.Blocks() as demo:
    gr.Chatbot(history, type="messages")

demo.launch()