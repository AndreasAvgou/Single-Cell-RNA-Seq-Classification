import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, accuracy_score

def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion_clf = nn.CrossEntropyLoss()
    criterion_ae = nn.MSELoss()
    total_loss = 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits, rec = model(batch_x)
        loss = criterion_clf(logits, batch_y) + 0.1 * criterion_ae(rec, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, X_val, y_val, le, device, print_report=False):
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.FloatTensor(X_val).to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    if print_report:
        print(classification_report(y_val, preds, target_names=le.classes_))
    return f1_score(y_val, preds, average='macro'), accuracy_score(y_val, preds)