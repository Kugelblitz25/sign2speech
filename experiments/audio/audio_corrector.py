import torch
import torchaudio
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from speechbrain.inference.vocoders import HIFIGAN
from utils.common import create_path, get_logger
from utils.config import load_config
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub



logger = get_logger("logs/audio_corrector.log")

config = load_config("Generate Audio")
checkpoint_path = create_path(config.generator.checkpoints)

device = "cuda" if torch.cuda.is_available() else "cpu"

models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/s2s-transformer-librispeech",
    arg_overrides={"fp16": False}
)
s2s_model = models[0].to(device)
s2s_model.train()  

audio_generator = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir=checkpoint_path / "tts-hifigan-ljspeech",
)

def prepare_audio_dataset(csv_path: str):
    """Convert audio files to spectrograms for training."""
    df = pd.read_csv(csv_path)
    data = []
    
    for _, row in df.iterrows():
        incorrect_wav, sr = torchaudio.load(row['Incorrect_Audio'])
        correct_wav, _ = torchaudio.load(row['Correct_Audio'])

        transform = torchaudio.transforms.MelSpectrogram()
        incorrect_spec = transform(incorrect_wav)
        correct_spec = transform(correct_wav)

        data.append([row['Incorrect_Audio'], incorrect_spec.numpy().tolist(),
                     row['Correct_Audio'], correct_spec.numpy().tolist()])
    
    columns = ["Incorrect_Audio", "Incorrect_Spec", "Correct_Audio", "Correct_Spec"]
    return pd.DataFrame(data, columns=columns)

def train_s2s_model(dataset_path: str, output_model_path: str, epochs=10, lr=1e-4):
    """Fine-tune Fairseq's Speech-to-Speech model for speech correction."""

    logger.info("Loading dataset...")
    dataset = pd.read_csv(dataset_path)

    incorrect_specs = np.array(dataset["Incorrect_Spec"].tolist())
    correct_specs = np.array(dataset["Correct_Spec"].tolist())

    incorrect_specs = torch.tensor(incorrect_specs, dtype=torch.float32).to(device)
    correct_specs = torch.tensor(correct_specs, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(s2s_model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    s2s_model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = s2s_model.forward({"source": incorrect_specs})["target"]
        loss = criterion(output, correct_specs)
        loss.backward()
        optimizer.step()
        
        logger.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(s2s_model.state_dict(), output_model_path)
    logger.info(f"Fine-tuned S2S model saved to {output_model_path}")

def correct_speech(input_audio_path: str, output_audio_path: str, model_path: str):
    """Use trained Speech-to-Speech model to correct speech."""
    
    logger.info("Loading fine-tuned model...")
    s2s_model.load_state_dict(torch.load(model_path, map_location=device))
    s2s_model.eval()

    waveform, sr = torchaudio.load(input_audio_path)
    waveform = waveform.to(device)

    sample = {
        "source": waveform,
        "source_lengths": torch.tensor([waveform.shape[1]], device=device),
    }
    
    with torch.no_grad():
        corrected_spec = s2s_model.forward(sample)["target"]
    
    corrected_audio = audio_generator.decode_batch(corrected_spec.unsqueeze(0))
    corrected_audio_np = corrected_audio.cpu().numpy()[0][0]

    sf.write(output_audio_path, corrected_audio_np, sr)
    logger.info(f"Corrected speech saved to {output_audio_path}")

def main():
    dataset_path = "audio_correct.csv"
    processed_dataset_path = "processed_audio_dataset.csv"
    output_model_path = "fine_tuned_s2s_model.pth"
    
    logger.info("Generating spectrogram dataset...")
    dataset = prepare_audio_dataset(dataset_path)
    dataset.to_csv(processed_dataset_path, index=False)
    
    logger.info("Fine-tuning Speech-to-Speech model...")
    train_s2s_model(processed_dataset_path, output_model_path)
    
    input_audio = "incorrect_speech.wav"
    output_audio = "corrected_speech.wav"
    
    logger.info("Correcting Speech...")
    correct_speech(input_audio, output_audio, output_model_path)
    
    logger.info("Audio correction pipeline complete.")

if __name__ == "__main__":
    main()
