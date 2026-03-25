import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics():
    if not os.path.exists('training_history.csv'):
        print("Error: training_history.csv not found. Please run main.py first to train the model and generate the history.")
        return

    df = pd.read_csv('training_history.csv')
    epochs = df['epoch']
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, df['loss'], marker='o', label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(epochs)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, df['val_f1'], marker='o', label='Validation F1', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Macro F1 Score')
    plt.title('Validation F1 Score')
    plt.xticks(epochs)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, df['val_acc'], marker='o', label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.xticks(epochs)
    plt.legend()
    
    plt.tight_layout()
    output_path = 'training_metrics.png'
    plt.savefig(output_path)
    print(f"Successfully generated and saved plots to {output_path}")

if __name__ == '__main__':
    plot_metrics()
