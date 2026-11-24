# CommentScope — Jigsaw Toxicity Lab

A Streamlit UI for the three **user-trained** Jigsaw comment classifiers in `Jigsaw_3_Model_Comparison.ipynb`:

- Word TF-IDF + One-vs-Rest Logistic Regression
- BiLSTM + Attention, trained from scratch
- TextCNN, trained from scratch

No hosted prediction API or pretrained weights are used.

## 1. Train the real models

Run your KaggleHub-enabled comparison notebook end-to-end using:

```python
import kagglehub
path = kagglehub.dataset_download('julian3833/jigsaw-toxic-comment-classification-challenge')
```

Run notebook Section 12 to export all models. Copy these files from `artifacts_comparison/` into the same folder inside this app:

```
logreg.joblib
bilstm_state.pt
textcnn_state.pt
vocab.json
model_config.json
comparison_summary.csv      # optional, recommended
comparison_per_label.csv    # optional, recommended
```

**Do not use synthetic smoke-test exports** from an earlier development run as if they were fitted on Jigsaw.

## 2. Test locally

Python 3.11 is recommended. From this directory:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app includes: single-comment comparison, CSV batch processing (up to 1,000 comments and 5 MB), per-label scores with each model's threshold, downloadable CSV results, and comparison charts sourced from your notebook's held-out metrics.

## 3. Publish online using Streamlit Community Cloud

1. Create a GitHub repository and upload all files from this folder, including your trained artifacts.
2. If model files are larger than GitHub's normal file limits, use Git LFS or a supported model-artifact storage service and add a secure startup download method. **The included app expects files locally and does not automatically download them.**
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set entrypoint to `app.py`.
5. Click **Deploy**, then verify all three models load, single-comment inference works, and metrics match your training exports.

Streamlit's deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Important scope and limitations

This system classifies harmful-language categories in comments. It does **not** identify people, infer criminal intent, or predict someone's likelihood of committing a crime. Threat and hate categories can produce false positives and false negatives. Use human review, avoid exposing private personal data, and don't use output as evidence against an individual. Scores are classifier scores, not calibrated probabilities of harm.

## Security note

`joblib` uses Python object deserialization; load only your own trusted exported models. The PyTorch files are loaded using `weights_only=True`. Restrict the repository if your dataset or artifacts contain any personal information. Published Streamlit apps may be accessible to others depending on the deployment settings.
