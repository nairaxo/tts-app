import torch
from unsloth import FastLanguageModel
from snac import SNAC
import numpy as np
from scipy import signal
import io
import wave
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Literal
import uvicorn

# Token constants
TOKENISER_LENGTH = 128256
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_HUMAN = TOKENISER_LENGTH + 3
END_OF_HUMAN = TOKENISER_LENGTH + 4
START_OF_AI = TOKENISER_LENGTH + 5
END_OF_AI = TOKENISER_LENGTH + 6

GEN_START_TOKEN = 128259
GEN_EOS_TOKEN = 128258
GEN_END_EXTRA_TOKEN = 128260
GEN_REMOVE_TOKEN = 128258
CODE_OFFSET = 128266

# Pydantic models for request/response
class TTSRequest(BaseModel):
    text: str
    language: Literal["swahili", "wolof"] = "swahili"

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.snac_model = None
        self.device = None
        self.hf_token = "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"
        
    def load_snac_model(self):
        """Load SNAC model once for all languages"""
        if self.snac_model is None:
            self.snac_model = SNAC.from_pretrained(
                "hubertsiuzdak/snac_24khz",
                token=self.hf_token
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.snac_model = self.snac_model.to(self.device)
    
    def load_language_model(self, language: str):
        """Load language-specific model and tokenizer"""
        if language not in self.models:
            model_name = f"nairaxo/orpheus_lora_{language}"
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=512,
                token=self.hf_token
            )
            FastLanguageModel.for_inference(model)
            
            self.models[language] = model.to(self.device)
            self.tokenizers[language] = tokenizer
            
        return self.models[language], self.tokenizers[language]

# Global model manager
model_manager = ModelManager()

# FastAPI app
app = FastAPI(
    title="Orpheus TTS API",
    description="Text-to-Speech API supporting Swahili and Wolof languages",
    version="1.0.0"
)

def redistribute_codes(code_list, snac_model, device):
    """Redistribute quantized codes into SNAC layers"""
    if len(code_list) == 0:
        raise ValueError("Empty code list")
    
    if len(code_list) % 7 != 0:
        print(f"Warning: Code list length {len(code_list)} is not divisible by 7")
        # Trim to make it divisible by 7
        code_list = code_list[:(len(code_list) // 7) * 7]
        if len(code_list) == 0:
            raise ValueError("No valid codes after trimming")
    
    layer_1, layer_2, layer_3 = [], [], []
    num_groups = len(code_list) // 7
    
    for i in range(num_groups):
        group = code_list[7 * i: 7 * i + 7]
        
        # Validate code values to prevent CUDA indexing errors
        for j, code in enumerate(group):
            if code < 0:
                print(f"Warning: Negative code {code} at position {j} in group {i}")
                group[j] = 0
        
        layer_1.append(max(0, group[0]))
        layer_2.append(max(0, group[1] - 4096))
        layer_3.append(max(0, group[2] - (2 * 4096)))
        layer_3.append(max(0, group[3] - (3 * 4096)))
        layer_2.append(max(0, group[4] - (4 * 4096)))
        layer_3.append(max(0, group[5] - (5 * 4096)))
        layer_3.append(max(0, group[6] - (6 * 4096)))
    
    # Additional validation for layer values
    def validate_layer(layer, layer_name, max_val=4096):
        validated = []
        for val in layer:
            if val >= max_val or val < 0:
                print(f"Warning: Invalid value {val} in {layer_name}, clipping to valid range")
                val = max(0, min(val, max_val - 1))
            validated.append(val)
        return validated
    
    layer_1 = validate_layer(layer_1, "layer_1", 4096)
    layer_2 = validate_layer(layer_2, "layer_2", 4096)  
    layer_3 = validate_layer(layer_3, "layer_3", 4096)
    
    codes = [
        torch.tensor(layer_1, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(layer_2, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(layer_3, dtype=torch.long).unsqueeze(0).to(device)
    ]

    try:
        audio_waveform = snac_model.decode(codes)
        return audio_waveform
    except Exception as e:
        print(f"Error in SNAC decode: {e}")
        print(f"Layer shapes: {[c.shape for c in codes]}")
        print(f"Layer ranges: {[(c.min().item(), c.max().item()) for c in codes]}")
        raise

def tts_pipeline(prompt: str, language: str):
    """Text-to-speech pipeline"""
    try:
        # Load models
        model_manager.load_snac_model()
        model, tokenizer = model_manager.load_language_model(language)
        
        # Tokenize input prompt
        input_ids_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(model_manager.device)
        print(f"Input shape: {input_ids_tensor.shape}")

        # Prepare generation tokens
        start_token = torch.tensor([[GEN_START_TOKEN]], dtype=torch.int64, device=model_manager.device)
        end_tokens = torch.tensor([[END_OF_TEXT, GEN_END_EXTRA_TOKEN]], dtype=torch.int64, device=model_manager.device)
        modified_input_ids = torch.cat([start_token, input_ids_tensor, end_tokens], dim=1)
        attention_mask = torch.ones_like(modified_input_ids, device=model_manager.device)

        print(f"Modified input shape: {modified_input_ids.shape}")

        # Generate quantized codes
        with torch.no_grad():  # Ensure no gradients are computed
            generated_ids = model.generate(
                input_ids=modified_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=800,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                num_return_sequences=1,
                eos_token_id=GEN_EOS_TOKEN,
                use_cache=True,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            )

        print(f"Generated shape: {generated_ids.shape}")
        print(f"Generated tokens range: {generated_ids.min().item()} to {generated_ids.max().item()}")

        # Crop after last speaker marker
        marker_token = 128257
        token_indices = (generated_ids == marker_token).nonzero(as_tuple=True)
        if len(token_indices[1]) > 0:
            last_marker = token_indices[1][-1].item()
            cropped = generated_ids[:, last_marker + 1:]
            print(f"Cropped after marker at position {last_marker}")
        else:
            cropped = generated_ids
            print("No marker found, using full generation")

        print(f"Cropped shape: {cropped.shape}")

        # Remove unwanted tokens
        processed = cropped[cropped != GEN_REMOVE_TOKEN]
        print(f"After removing unwanted tokens: {processed.shape}")
        
        # Ensure 2D for trimming
        if processed.dim() == 1:
            processed = processed.unsqueeze(0)

        # Trim length to a multiple of 7
        total = processed.size(1)
        trimmed_length = (total // 7) * 7
        
        if trimmed_length == 0:
            raise ValueError("No valid audio codes generated")
            
        trimmed = processed[:, :trimmed_length]
        print(f"Trimmed to length: {trimmed_length}")

        # Convert to codes and validate
        codes_tensor = trimmed - CODE_OFFSET
        print(f"Codes tensor range: {codes_tensor.min().item()} to {codes_tensor.max().item()}")
        
        codes = codes_tensor.flatten().tolist()
        print(f"Number of codes: {len(codes)}")
        
        # Validate codes before sending to SNAC
        valid_codes = []
        for code in codes:
            if code < 0:
                print(f"Warning: Negative code {code}, setting to 0")
                valid_codes.append(0)
            elif code >= 28672:  # 7 * 4096 = maximum expected code value
                print(f"Warning: Code {code} too large, clipping")
                valid_codes.append(28671)
            else:
                valid_codes.append(code)
        
        audio_waveform = redistribute_codes(valid_codes, model_manager.snac_model, model_manager.device)
        return audio_waveform
        
    except Exception as e:
        print(f"Error in TTS pipeline: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

def numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 8000) -> bytes:
    """Convert numpy array to WAV bytes"""
    # Ensure audio is in the right format
    if audio_array.dtype != np.int16:
        # Normalize to [-1, 1] and convert to int16
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Loading SNAC model...")
    model_manager.load_snac_model()
    print("API ready!")

@app.get("/")
async def root():
    return {"message": "Orpheus TTS API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": model_manager.device}

@app.post("/generate-speech")
async def generate_speech(request: TTSRequest):
    """Generate speech from text"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        print(f"Generating speech for: '{request.text}' in {request.language}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate audio
        waveform = tts_pipeline(request.text, request.language)
        
        if waveform is None or waveform.numel() == 0:
            raise HTTPException(status_code=500, detail="Generated audio is empty")
            
        array = waveform.detach().cpu().numpy()
        array = np.squeeze(array)
        
        print(f"Generated audio shape: {array.shape}")
        print(f"Audio range: {array.min()} to {array.max()}")

        # Check if audio is valid
        if len(array) == 0:
            raise HTTPException(status_code=500, detail="Generated audio has zero length")

        # Resample to 8kHz
        target_sr = 8000
        original_sr = 24000
        
        if len(array) > 0:
            num_samples = int(len(array) * target_sr / original_sr)
            if num_samples > 0:
                array = signal.resample(array, num_samples)
            else:
                raise HTTPException(status_code=500, detail="Resampled audio has zero length")

        print(f"Resampled audio shape: {array.shape}")

        # Convert to WAV bytes
        wav_bytes = numpy_to_wav_bytes(array, target_sr)
        
        print(f"WAV file size: {len(wav_bytes)} bytes")
        
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_{request.language}.wav"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/supported-languages")
async def get_supported_languages():
    return {"languages": ["swahili", "wolof"]}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )