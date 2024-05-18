import gradio as gr
from transformers import pipeline

pipe = pipeline(task="image-classification", 
                model="microsoft/dit-base-finetuned-rvlcdip", device=0)

gr.Interface.from_pipeline(pipe, 
                           title="DIT Test",
                           ).launch(server_name='0.0.0.0')

