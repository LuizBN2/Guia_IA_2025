import streamlit as st
from tensorflow.keras.callbacks import Callback

class StreamlitProgressCallback(Callback):
    def __init__(self, container, total_epochs):
        super().__init__()
        self.container = container
        self.total_epochs = total_epochs
        self.progress_bar = None
        self.status_text = None
        self.epoch = 0

    def on_train_begin(self, logs=None):
        self.progress_bar = self.container.progress(0)
        self.status_text = self.container.empty()
        self.epoch = 0
        self.status_text.text("⏳ Entrenamiento iniciado...")

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)

        self.status_text.markdown(
            f"📈 **Época {self.epoch}/{self.total_epochs}**  \n"
            f"✅ Accuracy: `{acc:.4f}` | 🔍 Val_Accuracy: `{val_acc:.4f}`  \n"
            f"❌ Loss: `{loss:.4f}` | 🔎 Val_Loss: `{val_loss:.4f}`"
        )
        self.progress_bar.progress(int((self.epoch / self.total_epochs) * 100))

    def on_train_end(self, logs=None):
        self.status_text.text("🏁 Entrenamiento finalizado.")
        self.progress_bar.progress(100)
