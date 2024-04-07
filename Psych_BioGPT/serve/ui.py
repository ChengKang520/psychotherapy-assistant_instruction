

import gradio as gr
from model import ModelServe1, ModelServe2
import sys
import os
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--base_model', type=str, default='no-rep', help='Base Model Path.')
    parser.add_argument('--model_type', type=str, default='wp', help='wp, wikitext')
    parser.add_argument('--load_8bit', type=bool, default=True, help='')
    parser.add_argument('--finetuned_weights', type=str, default='gpt2', help='')
    parser.add_argument('--app', type=str, default='', help='')


    args = parser.parse_args()
    print('**************************  args  ******************************')
    print(f"arg is: {args}")
    print('**************************  args  ******************************')

    model1 = ModelServe1(load_8bit=False, model_type=args.model_type, base_model=args.base_model, finetuned_weights=args.finetuned_weights)
    model2 = ModelServe2(load_8bit=False, model_type=args.model_type, base_model=args.base_model, finetuned_weights=args.finetuned_weights)


    falcon = gr.Interface(
        fn=model1.generate,
        inputs=[
            gr.components.Textbox(
                lines=4, label="Instruction", placeholder="Tell me about falcon."
            ),
            gr.components.Textbox(lines=4, label="Input", placeholder="none"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=10,
                label="Output",
            ),
        ],
        title="🦅 " + args.model_type,
        description=" 🍻 🍻 🍻 🍻 🍻 🍻 🍻",
    )


    falcon_ft = gr.Interface(
        fn=model2.generate,
        inputs=[
            gr.components.Textbox(
                lines=4, label="Instruction", placeholder="Tell me about falcon."
            ),
            gr.components.Textbox(lines=4, label="Input", placeholder="none"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=10,
                label="Output",
            ),
        ],
        title="🦅 " + args.model_type + "Fine-Tuned on Psychotherapy Data 🦉",
        description=" 🍻 🍻 🍻 🍻 🍻 🍻 🍻",
    )


    app = gr.TabbedInterface(interface_list=[falcon, falcon_ft],
                             tab_names=["Falcon inference", "Falcon-FT inference"])
    app.launch(share=True)

    # tell me about how to fight against depression

