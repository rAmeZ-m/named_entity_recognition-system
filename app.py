import gradio as gr
from transformers import pipeline

ner_pipe = pipeline("token-classification", model="rAmZ-m/distilbert-ner-conll2003",
                    aggregation_strategy="simple")

COLORS = {"PER": "#ffd166", "ORG": "#8ecae6", "LOC": "#95d5b2", "MISC": "#f4a6c1"}

def predict(text):
    if not text.strip():
        return {"text": "", "entities": []}, []
    ents = ner_pipe(text)
    highlighted = {
        "text": text,
        "entities": [{"entity": e["entity_group"], "start": e["start"], "end": e["end"]} for e in ents],
    }
    rows = [[text[e["start"]:e["end"]], e["entity_group"], round(float(e["score"]), 3)] for e in ents]
    return highlighted, rows

CSS = """
.gradio-container {max-width: 900px !important; margin: auto;}
#title {text-align: center;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CSS, title="NER App") as demo:
    gr.Markdown("# 🔎 Named Entity Recognition\n"
                "Detects **people, organizations, locations and misc** entities in English text "
                "(DistilBERT fine-tuned on CoNLL-2003).", elem_id="title")

    text = gr.Textbox(lines=5, label="Your text", placeholder="Type or paste English text here...")
    with gr.Row():
        clear_btn = gr.Button("Clear")
        submit_btn = gr.Button("Detect entities", variant="primary")

    out = gr.HighlightedText(color_map=COLORS, show_legend=True, label="Detected entities")
    table = gr.Dataframe(headers=["Entity", "Type", "Confidence"], label="Details", interactive=False)

    gr.Examples(
        examples=[
            ["Barack Obama visited Microsoft headquarters in Seattle last Monday."],
            ["Mohamed Salah scored for Liverpool against Manchester United in England."],
            ["The European Union criticized the German government over the Egyptian deal."],
        ],
        inputs=text,
    )

    submit_btn.click(predict, text, [out, table])
    text.submit(predict, text, [out, table])
    clear_btn.click(lambda: ("", {"text": "", "entities": []}, []), None, [text, out, table])

if __name__ == "__main__":
    demo.launch()