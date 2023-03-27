import gradio as gr

from predict import predict

# 局域网可访问，设置server_name='0.0.0.0'
gr.Interface(fn=predict, inputs=["image", "image"], outputs=["image", "number"]).launch(server_name='0.0.0.0')
