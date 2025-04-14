import joblib
import gradio as gr

# Load the model
model_path = r"C:\Users\aksha\OneDrive\Desktop\Project\Internship\Career_Prediction.pk1"
model = joblib.load(model_path)

# Prediction function
def predict_role(skills_input):
    if not skills_input:
        return "Please input at least one skill."
    # Process the input skills (split by commas and remove spaces)
    processed = ','.join(skills_input.split(",")).lower().replace(" ", "")
    prediction = model.predict([processed])[0]
    return f"{prediction}"

# Gradio UI with a textbox for skill input
interface = gr.Interface(
    fn=predict_role,
    inputs=gr.Textbox(label="Write Your Skills (comma-separated)", placeholder="e.g., Python, Data Science, SQL"),
    outputs=gr.Textbox(label="Predicted Career Role"),
    title="Career Role Predictor",
    description="Write your tech skills to predict your ideal role."
)

interface.launch()
