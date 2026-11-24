"""Inference code compatible with Jigsaw_3_Model_Comparison.ipynb exports.
Load model files only from a trusted training session.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import torch
from torch import nn

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MODEL_NAMES = {
    'Logistic Regression': 'logreg',
    'BiLSTM + Attention': 'bilstm',
    'TextCNN': 'textcnn',
}
PAD_ID, UNK_ID = 0, 1
TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[^\w\s]", flags=re.I)
CORE_FILES = ['model_config.json', 'vocab.json']
MODEL_FILES = {
    'Logistic Regression': 'logreg.joblib',
    'BiLSTM + Attention': 'bilstm_state.pt',
    'TextCNN': 'textcnn_state.pt',
}


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=128, hidden_dim=96, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, len(LABELS))

    def forward(self, ids, lengths):
        x = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=ids.size(1))
        valid = torch.arange(ids.size(1), device=ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        weights = self.attention(outputs).squeeze(-1).masked_fill(~valid, -1e9).softmax(dim=1)
        pooled = (outputs * weights.unsqueeze(-1)).sum(dim=1)
        return self.fc(self.dropout(pooled))


class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=128, channels=96, kernels=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, channels, kernel_size=k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernels) * channels, len(LABELS))
        self.kernels = kernels

    def forward(self, ids, lengths):
        x = self.embedding(ids).transpose(1, 2)
        if x.size(2) < max(self.kernels):
            x = nn.functional.pad(x, (0, max(self.kernels) - x.size(2)))
        pooled = []
        for kernel, conv in zip(self.kernels, self.convs):
            features = torch.relu(conv(x))
            positions = features.size(2)
            valid = torch.arange(positions, device=ids.device).unsqueeze(0) < torch.clamp(lengths-kernel+1, min=1).unsqueeze(1)
            features = features.masked_fill(~valid.unsqueeze(1), torch.finfo(features.dtype).min)
            pooled.append(features.max(dim=-1).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def expected_files():
    return ['model_config.json', 'vocab.json', 'logreg.joblib', 'bilstm_state.pt', 'textcnn_state.pt']


class ModelStore:
    def __init__(self, directory: str | Path):
        self.path = Path(directory)
        missing = [name for name in CORE_FILES if not (self.path / name).is_file()]
        if missing:
            raise FileNotFoundError('Missing shared model files: ' + ', '.join(missing))
        self.meta = json.loads((self.path / 'model_config.json').read_text(encoding='utf-8'))
        self.vocab = json.loads((self.path / 'vocab.json').read_text(encoding='utf-8'))
        if self.meta['labels'] != LABELS:
            raise ValueError('Unexpected training labels: models and UI may be incompatible')
        if len(self.vocab) != int(self.meta['vocab_size']):
            raise ValueError('Vocabulary size does not match model metadata')
        self.max_len = int(self.meta['max_len'])
        if self.max_len <= 0:
            raise ValueError('Invalid max_len')
        self.logreg = None
        self.bilstm = None
        self.textcnn = None
        self.available_models = []
        if (self.path / MODEL_FILES['Logistic Regression']).is_file():
            # joblib deserialization is appropriate only for trusted local exports.
            self.logreg = joblib.load(self.path / MODEL_FILES['Logistic Regression'])
            self.available_models.append('Logistic Regression')
        neural_models = [
            ('BiLSTM + Attention', 'bilstm', 'BiLSTM', 'bilstm_state.pt'),
            ('TextCNN', 'textcnn', 'TextCNN', 'textcnn_state.pt'),
        ]
        for display_name, attribute, architecture, filename in neural_models:
            if not (self.path / filename).is_file():
                continue
            obj = (
                BiLSTMClassifier(len(self.vocab), **self.meta['architecture'][architecture])
                if architecture == 'BiLSTM'
                else TextCNNClassifier(len(self.vocab), **self.meta['architecture'][architecture])
            )
            try:
                state = torch.load(self.path / filename, map_location='cpu', weights_only=True)
            except TypeError as ex:
                raise RuntimeError('Use a recent PyTorch version that supports weights_only=True') from ex
            obj.load_state_dict(state, strict=True)
            obj.eval()
            setattr(self, attribute, obj)
            self.available_models.append(display_name)
        if not self.available_models:
            raise FileNotFoundError('No trained model checkpoints found in ' + str(self.path))
        self.thresholds = self.meta['thresholds']

    def encode_many(self, comments):
        values = []
        for comment in comments:
            tokens = TOKEN_PATTERN.findall(str(comment).lower())[:self.max_len]
            ids = [self.vocab.get(tok, UNK_ID) for tok in tokens] or [UNK_ID]
            values.append(ids)
        max_size = max(len(row) for row in values)
        ids_tensor = torch.full((len(values), max_size), PAD_ID, dtype=torch.long)
        for i, row in enumerate(values):
            ids_tensor[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        lengths = torch.tensor([len(row) for row in values], dtype=torch.long)
        return ids_tensor, lengths

    @torch.inference_mode()
    def predict(self, comments: list[str], selection: list[str]) -> Dict[str, np.ndarray]:
        if not comments or any(not str(c).strip() for c in comments):
            raise ValueError('Supply nonempty comments')
        if len(comments) > 1000:
            raise ValueError('Maximum 1,000 comments per request')
        unavailable = sorted(set(selection) - set(self.available_models))
        if unavailable:
            raise ValueError('Model artifacts are not available: ' + ', '.join(unavailable))
        out = {}
        if 'Logistic Regression' in selection:
            out['Logistic Regression'] = np.asarray(self.logreg.predict_proba(comments), dtype=float)
        neural_selection = set(selection) & {'BiLSTM + Attention', 'TextCNN'}
        if neural_selection:
            chunks = {name: [] for name in neural_selection}
            for start in range(0, len(comments), 64):
                ids, lengths = self.encode_many(comments[start:start+64])
                for name in neural_selection:
                    model = self.bilstm if name == 'BiLSTM + Attention' else self.textcnn
                    chunks[name].append(torch.sigmoid(model(ids, lengths)).numpy())
            for name in neural_selection:
                out[name] = np.concatenate(chunks[name])
        for name, scores in out.items():
            if scores.shape != (len(comments), 6):
                raise RuntimeError(f'Unexpected scores shape for {name}: {scores.shape}')
        return out

    def threshold(self, model_name: str, label: str) -> float:
        return float(self.thresholds[MODEL_NAMES[model_name]][label])
