import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from data_utils import load_data, preprocess_pipeline
from model_arch import AlzheimerCellNet
from engine import train_epoch, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y, X_test_final = load_data('sc_alz_train_set.csv', 'sc_alz_test_set.csv')
X_tr, X_va, y_tr, y_va, X_te, le = preprocess_pipeline(X, y, X_test_final)

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)), batch_size=128, shuffle=True)

model = AlzheimerCellNet(X_tr.shape[1], len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

history = {'epoch': [], 'loss': [], 'val_f1': [], 'val_acc': []}

for epoch in range(20):
    loss = train_epoch(model, train_loader, optimizer, device)
    
    print_report = (epoch + 1) % 5 == 0
    f1, acc = evaluate_model(model, X_va, y_va, le, device, print_report=print_report)
    
    history['epoch'].append(epoch + 1)
    history['loss'].append(loss)
    history['val_f1'].append(f1)
    history['val_acc'].append(acc)
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val F1: {f1:.4f} | Val Acc: {acc:.4f}")

pd.DataFrame(history).to_csv('training_history.csv', index=False)
print("Saved training history to training_history.csv")

model.eval()
with torch.no_grad():
    test_logits, _ = model(torch.FloatTensor(X_te).to(device))
    test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()

submission = pd.DataFrame({'cell_id': range(len(test_preds)), 'predicted_label': le.inverse_transform(test_preds)})
submission.to_csv('submission_alzheimer_2026.csv', index=False)